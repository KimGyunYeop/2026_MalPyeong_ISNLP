"""Fail-closed contracts for draining one live mixed lane at a case boundary.

The live GPU0 runner was launched with three remaining cases in one shell.  A
plain ``screen -X quit`` would kill the current training process, while polling
``status.txt`` and then killing the screen races with the next case.  This
module resolves the *outer* runner through an exact process ancestry and stops
only that PID.  Its current child can therefore finish train/inference, but the
outer shell cannot advance to the next case.

No PID is a configuration value.  Every mutating command revalidates the Linux
process start tick, command line, environment, cwd, screen ancestry, run leaf,
and boot id immediately before signalling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from main_code.quantized_queue_contract import (
    BarrierWaiting,
    ContractError,
    EXPECTED_ROWS,
    MIXED_CONFIG_IDS,
    MIXED_GPU1_CASES,
    MIXED_GROUP,
    expected_run_dir,
    validate_case,
)


DRAIN_CASE = "m06_avg025_aux010"
DRAIN_SCREEN = "mixed_metric_c02_gpu0"
ORIGINAL_GPU1_SCREEN = "mixed_metric_c02_gpu1"
MIGRATED_SCREEN = "mixed_metric_single_gpu1_continuation"
OLD_COORDINATOR_SCREEN = "quantized_after_mixed_queue"
NEW_COORDINATOR_SCREEN = "single_gpu_after_mixed_queue"
DEFAULT_QUARANTINE_RELATIVE = Path(
    "tmp_trashcan2/gpu_queue_migration_20260814T0055"
)

EXPECTED_GPU0_RUN_CASES = (
    "m00_c02_control",
    "m02_seed44",
    "m04_avg_mse030",
    DRAIN_CASE,
    "m08_avg025_avg_rank020",
    "m11_avg025_avg_pair005",
)
ORIGINAL_GPU1_CASES = MIXED_GPU1_CASES
MIGRATED_CASES = (
    "m08_avg025_avg_rank020",
    "m11_avg025_avg_pair005",
)
ALREADY_COMPLETE_BEFORE_DRAIN = (
    "m00_c02_control",
    "m01_seed42",
    "m02_seed44",
    "m03_avg_mse020",
    "m04_avg_mse030",
    "m05_avg025_list000",
)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    state: str
    ppid: int
    cmdline: tuple[str, ...]


@dataclass(frozen=True)
class DrainTarget:
    schema_version: int
    boot_id: str
    host: str
    project_root: str
    results_root: str
    run_dir: str
    screen_name: str
    screen: ProcessIdentity
    outer: ProcessIdentity
    runner: ProcessIdentity
    expected_gpu: str
    expected_run_cases: tuple[str, ...]
    state: str = "resolved"


@dataclass(frozen=True)
class PredrainedEvidence:
    schema_version: int
    state: str
    host: str
    project_root: str
    results_root: str
    run_dir: str
    screen_name: str
    quarantine_root: str
    quarantined_run_dir: str
    quarantined_lock: str
    quarantined_status: str
    quarantined_exit_code: int
    evidence_sha256: dict[str, str]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _target_as_json(target: DrainTarget) -> dict[str, Any]:
    value = asdict(target)
    # JSON arrays are accepted by ``load_target`` and normalized back to tuples.
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"evidence hash 읽기 실패: {path}: {exc}") from exc
    return digest.hexdigest()


def _quarantined_paths(quarantine_root: Path) -> tuple[Path, Path]:
    model_dir = quarantine_root / "m08_avg025_avg_rank020" / "ax4_light"
    return model_dir / MIXED_CONFIG_IDS["m08_avg025_avg_rank020"], Path(
        f"{model_dir / MIXED_CONFIG_IDS['m08_avg025_avg_rank020']}.lock"
    )


def validate_predrained_state(
    project_root: Path,
    results_root: Path,
    quarantine_root: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
    proc_root: Path = Path("/proc"),
) -> PredrainedEvidence:
    """Validate a completed external drain without trusting a handwritten flag."""

    project_root = project_root.resolve()
    results_root = results_root.resolve()
    quarantine_root = quarantine_root.resolve()
    validate_case(results_root, DRAIN_CASE, expected_rows=expected_rows)
    validate_not_started(results_root, MIGRATED_CASES)
    try:
        _screen_pid(proc_root, DRAIN_SCREEN)
    except ContractError as exc:
        # Exactly no matching screen is required.  Distinguish it from duplicate
        # or unreadable matches by inspecting the message produced by _screen_pid.
        if not str(exc).endswith(": []"):
            raise
    else:
        raise ContractError(f"predrained 상태인데 screen이 아직 존재함: {DRAIN_SCREEN}")

    quarantined_run, quarantined_lock = _quarantined_paths(quarantine_root)
    if not quarantine_root.is_relative_to(project_root):
        raise ContractError(
            f"quarantine은 project root 내부여야 함: {quarantine_root}"
        )
    if not quarantined_run.is_dir() or not quarantined_lock.is_file():
        raise ContractError(
            "격리된 m08 leaf/lock 누락: "
            f"run={quarantined_run.is_dir()}, lock={quarantined_lock.is_file()}"
        )
    try:
        siblings = sorted(item.name for item in quarantined_run.parent.iterdir())
    except OSError as exc:
        raise ContractError(f"quarantine sibling 읽기 실패: {exc}") from exc
    expected_siblings = sorted((quarantined_run.name, quarantined_lock.name))
    if siblings != expected_siblings:
        raise ContractError(
            f"quarantine model folder에 예상 밖 항목 존재: {siblings} != {expected_siblings}"
        )

    experiment = _read_object(quarantined_run / "experiment.json")
    required_experiment = {
        "suite": "new_proposed",
        "base_name": MIXED_GROUP,
        "case_name": "m08_avg025_avg_rank020",
        "config_id": MIXED_CONFIG_IDS["m08_avg025_avg_rank020"],
        "model_argument": "ax4_light",
    }
    mismatched = {
        key: (experiment.get(key), expected)
        for key, expected in required_experiment.items()
        if experiment.get(key) != expected
    }
    if mismatched:
        raise ContractError(f"격리 m08 experiment identity 불일치: {mismatched}")
    try:
        status = (quarantined_run / "status.txt").read_text(encoding="utf-8").strip()
        raw_exit = (quarantined_run / "train_exit_code.txt").read_text(
            encoding="utf-8"
        ).strip()
    except OSError as exc:
        raise ContractError(f"격리 m08 status/exit 읽기 실패: {exc}") from exc
    if status != "train_failed" or raw_exit != "130":
        raise ContractError(
            f"격리 m08 실패 계약 불일치: status={status!r}, exit={raw_exit!r}"
        )
    context = _read_object(quarantined_run / "status_context.json")
    required_context = {
        "status": "train_failed",
        "phase": "train",
        "exit_code": 130,
        "host": socket.gethostname(),
    }
    context_mismatch = {
        key: (context.get(key), expected)
        for key, expected in required_context.items()
        if context.get(key) != expected
    }
    if context_mismatch:
        raise ContractError(f"격리 m08 context 불일치: {context_mismatch}")
    resolved = _read_object(quarantined_run / "resolved_config.json")
    required_config = {
        "model_slug": "ax4_light",
        "model_id": "skt/A.X-4.0-Light",
        "seed": 43,
        "ranking_target": "trait_average",
        "listwise_loss_weight": 0.2,
        "score_postprocess": "average_matched",
    }
    config_mismatch = {
        key: (resolved.get(key), expected)
        for key, expected in required_config.items()
        if resolved.get(key) != expected
    }
    if config_mismatch:
        raise ContractError(f"격리 m08 config 불일치: {config_mismatch}")

    evidence_files = (
        "experiment.json",
        "resolved_config.json",
        "status.txt",
        "status_context.json",
        "train_exit_code.txt",
        "train_console.log",
    )
    missing = [name for name in evidence_files if not (quarantined_run / name).is_file()]
    if missing:
        raise ContractError(f"격리 m08 evidence 누락: {missing}")
    hashes = {name: _sha256(quarantined_run / name) for name in evidence_files}
    return PredrainedEvidence(
        schema_version=1,
        state="externally_drained",
        host=socket.gethostname(),
        project_root=str(project_root),
        results_root=str(results_root),
        run_dir=str(expected_run_dir(results_root, DRAIN_CASE)),
        screen_name=DRAIN_SCREEN,
        quarantine_root=str(quarantine_root),
        quarantined_run_dir=str(quarantined_run),
        quarantined_lock=str(quarantined_lock),
        quarantined_status=status,
        quarantined_exit_code=int(raw_exit),
        evidence_sha256=hashes,
    )


def record_predrained_state(
    project_root: Path,
    results_root: Path,
    quarantine_root: Path,
    manifest: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
    proc_root: Path = Path("/proc"),
) -> PredrainedEvidence:
    evidence = validate_predrained_state(
        project_root,
        results_root,
        quarantine_root,
        expected_rows=expected_rows,
        proc_root=proc_root,
    )
    _atomic_json(manifest, asdict(evidence))
    # Recompute after writing; the manifest itself is never sufficient evidence.
    observed = validate_predrained_state(
        project_root,
        results_root,
        quarantine_root,
        expected_rows=expected_rows,
        proc_root=proc_root,
    )
    if observed != evidence:
        raise ContractError("predrained evidence가 manifest 기록 중 변경됨")
    return evidence


def verify_predrained_manifest(
    manifest: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
    proc_root: Path = Path("/proc"),
) -> PredrainedEvidence:
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"predrained manifest 읽기 실패: {manifest}: {exc}") from exc
    if not isinstance(value, dict) or value.get("state") != "externally_drained":
        raise ContractError(f"predrained manifest state 불일치: {manifest}")
    required_strings = (
        "host",
        "project_root",
        "results_root",
        "run_dir",
        "screen_name",
        "quarantine_root",
        "quarantined_run_dir",
        "quarantined_lock",
        "quarantined_status",
    )
    if any(not isinstance(value.get(key), str) for key in required_strings):
        raise ContractError("predrained manifest string field 손상")
    expected = validate_predrained_state(
        Path(value["project_root"]),
        Path(value["results_root"]),
        Path(value["quarantine_root"]),
        expected_rows=expected_rows,
        proc_root=proc_root,
    )
    if value != asdict(expected):
        raise ContractError("predrained manifest와 재계산 evidence 불일치")
    return expected


def load_target(path: Path) -> DrainTarget:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"drain manifest 읽기 실패: {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ContractError(f"drain manifest schema 불일치: {path}")

    def identity(name: str) -> ProcessIdentity:
        item = value.get(name)
        if not isinstance(item, dict):
            raise ContractError(f"drain manifest {name} identity 누락")
        try:
            return ProcessIdentity(
                pid=int(item["pid"]),
                start_ticks=int(item["start_ticks"]),
                state=str(item["state"]),
                ppid=int(item["ppid"]),
                cmdline=tuple(str(part) for part in item["cmdline"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"drain manifest {name} identity 손상") from exc

    try:
        return DrainTarget(
            schema_version=1,
            boot_id=str(value["boot_id"]),
            host=str(value["host"]),
            project_root=str(value["project_root"]),
            results_root=str(value["results_root"]),
            run_dir=str(value["run_dir"]),
            screen_name=str(value["screen_name"]),
            screen=identity("screen"),
            outer=identity("outer"),
            runner=identity("runner"),
            expected_gpu=str(value["expected_gpu"]),
            expected_run_cases=tuple(str(item) for item in value["expected_run_cases"]),
            state=str(value.get("state", "resolved")),
        )
    except (KeyError, TypeError) as exc:
        raise ContractError(f"drain manifest 필드 손상: {path}") from exc


def _proc_path(proc_root: Path, pid: int, name: str = "") -> Path:
    path = proc_root / str(pid)
    return path / name if name else path


def _process_stat(proc_root: Path, pid: int) -> ProcessIdentity:
    base = _proc_path(proc_root, pid)
    try:
        stat_text = (base / "stat").read_text(encoding="utf-8")
        command_bytes = (base / "cmdline").read_bytes()
    except FileNotFoundError as exc:
        raise BarrierWaiting(f"process가 종료됨: pid={pid}") from exc
    except OSError as exc:
        raise ContractError(f"process 읽기 실패: pid={pid}: {exc}") from exc
    close = stat_text.rfind(")")
    if close < 0:
        raise ContractError(f"/proc stat 형식 손상: pid={pid}")
    remainder = stat_text[close + 2 :].split()
    if len(remainder) <= 19:
        raise ContractError(f"/proc stat 필드 부족: pid={pid}")
    try:
        state = remainder[0]
        ppid = int(remainder[1])
        start_ticks = int(remainder[19])
    except (IndexError, ValueError) as exc:
        raise ContractError(f"/proc stat 필드 손상: pid={pid}") from exc
    cmdline = tuple(
        part.decode("utf-8", errors="surrogateescape")
        for part in command_bytes.split(b"\0")
        if part
    )
    if not cmdline:
        raise ContractError(f"빈 process cmdline: pid={pid}")
    return ProcessIdentity(pid, start_ticks, state, ppid, cmdline)


def _environment(proc_root: Path, pid: int) -> dict[str, str]:
    try:
        raw = _proc_path(proc_root, pid, "environ").read_bytes()
    except FileNotFoundError as exc:
        raise BarrierWaiting(f"process가 종료됨: pid={pid}") from exc
    except OSError as exc:
        raise ContractError(f"process environ 읽기 실패: pid={pid}: {exc}") from exc
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="surrogateescape")] = value.decode(
            errors="surrogateescape"
        )
    return result


def _cwd(proc_root: Path, pid: int) -> Path:
    try:
        return _proc_path(proc_root, pid, "cwd").resolve(strict=True)
    except FileNotFoundError as exc:
        raise BarrierWaiting(f"process cwd가 사라짐: pid={pid}") from exc
    except OSError as exc:
        raise ContractError(f"process cwd 읽기 실패: pid={pid}: {exc}") from exc


def _boot_id(proc_root: Path) -> str:
    path = proc_root / "sys/kernel/random/boot_id"
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ContractError(f"boot_id 읽기 실패: {path}: {exc}") from exc
    if not value:
        raise ContractError("boot_id가 비어 있음")
    return value


def _is_shell_runner(identity: ProcessIdentity) -> bool:
    if len(identity.cmdline) < 2:
        return False
    return Path(identity.cmdline[0]).name == "bash" and identity.cmdline[1].endswith(
        "main_code/run_method.sh"
    )


def _screen_pid(proc_root: Path, screen_name: str) -> int:
    matches: list[int] = []
    for child in proc_root.iterdir():
        if not child.name.isdigit():
            continue
        try:
            identity = _process_stat(proc_root, int(child.name))
        except (BarrierWaiting, ContractError):
            continue
        command = identity.cmdline
        if len(command) >= 3 and Path(command[0]).name == "SCREEN":
            for index, part in enumerate(command[:-1]):
                if part == "-dmS" and command[index + 1] == screen_name:
                    matches.append(identity.pid)
                    break
    if len(matches) != 1:
        raise ContractError(
            f"exact screen process 수 불일치: {screen_name}: {matches}"
        )
    return matches[0]


def _is_descendant(proc_root: Path, pid: int, ancestor: int) -> bool:
    seen: set[int] = set()
    current = pid
    while current > 1 and current not in seen:
        seen.add(current)
        identity = _process_stat(proc_root, current)
        if identity.ppid == ancestor:
            return True
        current = identity.ppid
    return False


def _descendants(proc_root: Path, ancestor: int) -> list[ProcessIdentity]:
    identities: list[ProcessIdentity] = []
    for child in proc_root.iterdir():
        if not child.name.isdigit():
            continue
        pid = int(child.name)
        if pid == ancestor:
            continue
        try:
            if _is_descendant(proc_root, pid, ancestor):
                identities.append(_process_stat(proc_root, pid))
        except (BarrierWaiting, ContractError):
            continue
    return identities


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BarrierWaiting(f"artifact 없음: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"JSON 손상: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON object가 아님: {path}")
    return value


def validate_not_started(results_root: Path, cases: Iterable[str]) -> None:
    for case in cases:
        if case not in MIXED_CONFIG_IDS:
            raise ContractError(f"알 수 없는 mixed case: {case}")
        run_dir = expected_run_dir(results_root, case)
        lock_path = Path(f"{run_dir}.lock")
        if run_dir.exists() or lock_path.exists():
            raise ContractError(
                f"migration 대상 case가 이미 시작됐거나 lock이 존재함: {case}: "
                f"run={run_dir.exists()}, lock={lock_path.exists()}"
            )
        model_dir = run_dir.parent
        if model_dir.is_dir():
            siblings = sorted(item.name for item in model_dir.iterdir())
            if siblings:
                raise ContractError(
                    f"migration 대상 model folder가 비어 있지 않음: {case}: {siblings}"
                )


def validate_named_cases(
    results_root: Path,
    cases: Sequence[str],
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> tuple[list[Path], list[str]]:
    complete: list[Path] = []
    waiting: list[str] = []
    for case in cases:
        try:
            complete.append(
                validate_case(results_root, case, expected_rows=expected_rows)
            )
        except BarrierWaiting as exc:
            waiting.append(f"{case}: {exc}")
    return complete, waiting


def resolve_drain_target(
    project_root: Path,
    results_root: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> DrainTarget:
    project_root = project_root.resolve()
    results_root = results_root.resolve()
    validate_not_started(results_root, MIGRATED_CASES)
    run_dir = expected_run_dir(results_root, DRAIN_CASE)
    experiment = _read_object(run_dir / "experiment.json")
    required_experiment = {
        "suite": "new_proposed",
        "base_name": MIXED_GROUP,
        "case_name": DRAIN_CASE,
        "config_id": MIXED_CONFIG_IDS[DRAIN_CASE],
        "model_argument": "ax4_light",
    }
    mismatched = {
        key: (experiment.get(key), expected)
        for key, expected in required_experiment.items()
        if experiment.get(key) != expected
    }
    if mismatched:
        raise ContractError(f"drain experiment identity 불일치: {mismatched}")
    try:
        status = (run_dir / "status.txt").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BarrierWaiting(f"drain status 읽기 실패: {run_dir}: {exc}") from exc
    if status not in {"running", "trained"}:
        raise ContractError(
            f"drain은 실행 중 m06에서만 arm할 수 있음: status={status!r}"
        )
    context = _read_object(run_dir / "status_context.json")
    runner_pid = context.get("runner_pid")
    if not isinstance(runner_pid, int) or runner_pid <= 1:
        raise ContractError("m06 status_context runner_pid가 유효하지 않음")
    if context.get("host") != socket.gethostname():
        raise ContractError(
            f"m06 runner host 불일치: {context.get('host')!r} != {socket.gethostname()!r}"
        )

    screen_pid = _screen_pid(proc_root, DRAIN_SCREEN)
    screen = _process_stat(proc_root, screen_pid)
    runner = _process_stat(proc_root, runner_pid)
    if not _is_shell_runner(runner):
        raise ContractError(f"m06 runner cmdline 불일치: {runner.cmdline}")
    outer = _process_stat(proc_root, runner.ppid)
    if not _is_shell_runner(outer):
        raise ContractError(f"m06 outer cmdline 불일치: {outer.cmdline}")
    if not _is_descendant(proc_root, outer.pid, screen.pid):
        raise ContractError(
            f"m06 outer가 exact screen descendant가 아님: {outer.pid} !< {screen.pid}"
        )
    if runner.ppid != outer.pid:
        raise ContractError("m06 runner의 direct parent가 outer가 아님")
    if outer.state in {"T", "t", "Z", "X"}:
        raise ContractError(f"m06 outer가 arm 가능한 상태가 아님: {outer.state}")
    if runner.state in {"T", "t", "Z", "X"}:
        raise ContractError(f"m06 runner가 진행 중 상태가 아님: {runner.state}")

    environment = _environment(proc_root, outer.pid)
    required_environment = {
        "GPU": "0",
        "RUN_STAGE": "mixed_traitrmse_avgrho_v1",
        "RUN_MODELS": "ax4_light",
        "RESULTS_ROOT": str(results_root),
    }
    env_mismatch = {
        key: (environment.get(key), expected)
        for key, expected in required_environment.items()
        if environment.get(key) != expected
    }
    if env_mismatch:
        raise ContractError(f"m06 outer environment 불일치: {env_mismatch}")
    observed_cases = tuple(environment.get("RUN_CASES", "").split())
    if observed_cases != EXPECTED_GPU0_RUN_CASES:
        raise ContractError(
            f"m06 outer RUN_CASES 불일치: {observed_cases} != {EXPECTED_GPU0_RUN_CASES}"
        )
    if _cwd(proc_root, outer.pid) != project_root:
        raise ContractError(
            f"m06 outer cwd 불일치: {_cwd(proc_root, outer.pid)} != {project_root}"
        )
    if _cwd(proc_root, runner.pid) != project_root:
        raise ContractError("m06 runner cwd 불일치")

    return DrainTarget(
        schema_version=1,
        boot_id=_boot_id(proc_root),
        host=socket.gethostname(),
        project_root=str(project_root),
        results_root=str(results_root),
        run_dir=str(run_dir),
        screen_name=DRAIN_SCREEN,
        screen=screen,
        outer=outer,
        runner=runner,
        expected_gpu="0",
        expected_run_cases=EXPECTED_GPU0_RUN_CASES,
    )


def verify_target(
    target: DrainTarget,
    *,
    proc_root: Path = Path("/proc"),
    require_stopped: bool = False,
    allow_runner_gone: bool = False,
) -> None:
    if target.boot_id != _boot_id(proc_root):
        raise ContractError("boot_id가 바뀌어 저장된 PID identity를 사용할 수 없음")
    if target.host != socket.gethostname():
        raise ContractError("drain manifest host 불일치")
    screen = _process_stat(proc_root, target.screen.pid)
    outer = _process_stat(proc_root, target.outer.pid)
    if screen.start_ticks != target.screen.start_ticks:
        raise ContractError("screen PID가 재사용됨")
    if outer.start_ticks != target.outer.start_ticks:
        raise ContractError("outer PID가 재사용됨")
    if screen.cmdline != target.screen.cmdline or outer.cmdline != target.outer.cmdline:
        raise ContractError("screen/outer cmdline identity가 바뀜")
    if not _is_descendant(proc_root, outer.pid, screen.pid):
        raise ContractError("outer가 더 이상 exact screen descendant가 아님")
    if require_stopped and outer.state not in {"T", "t"}:
        raise ContractError(f"outer가 stopped 상태가 아님: {outer.state}")

    try:
        runner = _process_stat(proc_root, target.runner.pid)
    except BarrierWaiting:
        if not allow_runner_gone:
            raise
    else:
        if runner.start_ticks != target.runner.start_ticks:
            raise ContractError("runner PID가 재사용됨")
        if runner.ppid != outer.pid or runner.cmdline != target.runner.cmdline:
            raise ContractError("runner ancestry/cmdline identity가 바뀜")
    environment = _environment(proc_root, outer.pid)
    if environment.get("GPU") != target.expected_gpu:
        raise ContractError("outer GPU environment가 바뀜")
    if tuple(environment.get("RUN_CASES", "").split()) != target.expected_run_cases:
        raise ContractError("outer RUN_CASES environment가 바뀜")
    if _cwd(proc_root, outer.pid) != Path(target.project_root):
        raise ContractError("outer cwd identity가 바뀜")


def stop_outer_at_boundary(
    project_root: Path,
    results_root: Path,
    manifest: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> DrainTarget:
    target = resolve_drain_target(
        project_root, results_root, proc_root=proc_root
    )
    _atomic_json(manifest, _target_as_json(target))
    # Re-read and revalidate the persisted identity immediately before the one
    # permitted signal.  os.kill receives a positive PID, never a process group.
    persisted = load_target(manifest)
    verify_target(persisted, proc_root=proc_root)
    os.kill(persisted.outer.pid, signal.SIGSTOP)
    deadline = time.monotonic() + 3.0
    while True:
        try:
            verify_target(
                persisted,
                proc_root=proc_root,
                require_stopped=True,
                allow_runner_gone=True,
            )
            break
        except (BarrierWaiting, ContractError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    stopped_value = _target_as_json(persisted)
    stopped_value["state"] = "outer_stopped"
    _atomic_json(manifest, stopped_value)
    return load_target(manifest)


def assert_drain_ready(
    results_root: Path,
    manifest: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
    proc_root: Path = Path("/proc"),
) -> Path:
    run_dir = validate_case(results_root, DRAIN_CASE, expected_rows=expected_rows)
    target = load_target(manifest)
    verify_target(
        target,
        proc_root=proc_root,
        require_stopped=True,
        allow_runner_gone=True,
    )
    try:
        runner = _process_stat(proc_root, target.runner.pid)
    except BarrierWaiting:
        runner = None
    if runner is not None and runner.state != "Z":
        raise BarrierWaiting(
            f"m06 lifecycle shell이 아직 종료 전: pid={runner.pid}, state={runner.state}"
        )
    active_descendants = [
        item
        for item in _descendants(proc_root, target.outer.pid)
        if item.state not in {"Z", "X"}
    ]
    if active_descendants:
        raise BarrierWaiting(
            "stopped outer 아래 active child가 남음: "
            + ", ".join(f"{item.pid}:{item.state}" for item in active_descendants)
        )
    validate_not_started(results_root, MIGRATED_CASES)
    return run_dir


def finalize_live_drain(
    results_root: Path,
    manifest: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
    proc_root: Path = Path("/proc"),
) -> None:
    """Terminate only the stopped outer after m06 is immutably complete.

    SIGTERM/HUP can remain pending while a process is job-control stopped.  An
    exact positive-PID SIGKILL is used only after ``assert_drain_ready`` proves
    that m06 and both inference surfaces are complete, no active child remains,
    and m08/m11 have not started.  This avoids resuming the shell into m08.
    """

    assert_drain_ready(
        results_root,
        manifest,
        expected_rows=expected_rows,
        proc_root=proc_root,
    )
    target = load_target(manifest)
    verify_target(
        target,
        proc_root=proc_root,
        require_stopped=True,
        allow_runner_gone=True,
    )
    validate_not_started(results_root, MIGRATED_CASES)
    os.kill(target.outer.pid, signal.SIGKILL)
    deadline = time.monotonic() + 3.0
    while _proc_path(proc_root, target.outer.pid).exists():
        if time.monotonic() >= deadline:
            raise ContractError(
                f"exact stopped outer 종료 확인 실패: pid={target.outer.pid}"
            )
        time.sleep(0.05)


def resume_outer(
    results_root: Path,
    manifest: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> None:
    target = load_target(manifest)
    verify_target(target, proc_root=proc_root, require_stopped=True)
    validate_not_started(results_root, MIGRATED_CASES)
    try:
        validate_case(results_root, DRAIN_CASE)
    except BarrierWaiting:
        pass
    else:
        raise ContractError(
            "m06이 complete라 resume하면 m08이 즉시 시작됨; resume를 거부함"
        )
    # Positive exact PID only.  Revalidate immediately before SIGCONT.
    verify_target(target, proc_root=proc_root, require_stopped=True)
    os.kill(target.outer.pid, signal.SIGCONT)


def _print_case_status(
    results_root: Path, cases: Sequence[str], expected_rows: int
) -> int:
    try:
        complete, waiting = validate_named_cases(
            results_root, cases, expected_rows=expected_rows
        )
    except ContractError as exc:
        print(f"CONTRACT ERROR: {exc}")
        return 2
    print(f"CASES complete={len(complete)}/{len(cases)} waiting={len(waiting)}")
    for message in waiting:
        print(f"WAITING {message}")
    return 3 if waiting else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cases = subparsers.add_parser("cases")
    cases.add_argument("--results-root", type=Path, required=True)
    cases.add_argument("--case", action="append", required=True)
    cases.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)

    not_started = subparsers.add_parser("not-started")
    not_started.add_argument("--results-root", type=Path, required=True)
    not_started.add_argument("--case", action="append", required=True)

    resolve = subparsers.add_parser("resolve-drain")
    resolve.add_argument("--project-root", type=Path, required=True)
    resolve.add_argument("--results-root", type=Path, required=True)

    predrained = subparsers.add_parser("check-predrained")
    predrained.add_argument("--project-root", type=Path, required=True)
    predrained.add_argument("--results-root", type=Path, required=True)
    predrained.add_argument("--quarantine-root", type=Path, required=True)
    predrained.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)

    record_predrained = subparsers.add_parser("record-predrained")
    record_predrained.add_argument("--project-root", type=Path, required=True)
    record_predrained.add_argument("--results-root", type=Path, required=True)
    record_predrained.add_argument("--quarantine-root", type=Path, required=True)
    record_predrained.add_argument("--manifest", type=Path, required=True)
    record_predrained.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)

    verify_predrained = subparsers.add_parser("verify-predrained")
    verify_predrained.add_argument("--manifest", type=Path, required=True)
    verify_predrained.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)

    stop = subparsers.add_parser("stop-drain")
    stop.add_argument("--project-root", type=Path, required=True)
    stop.add_argument("--results-root", type=Path, required=True)
    stop.add_argument("--manifest", type=Path, required=True)

    verify = subparsers.add_parser("verify-drain")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--require-stopped", action="store_true")
    verify.add_argument("--allow-runner-gone", action="store_true")

    ready = subparsers.add_parser("drain-ready")
    ready.add_argument("--results-root", type=Path, required=True)
    ready.add_argument("--manifest", type=Path, required=True)
    ready.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)

    finalize = subparsers.add_parser("finalize-drain")
    finalize.add_argument("--results-root", type=Path, required=True)
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)

    resume = subparsers.add_parser("resume-drain")
    resume.add_argument("--results-root", type=Path, required=True)
    resume.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "cases":
            return _print_case_status(args.results_root, args.case, args.expected_rows)
        if args.command == "not-started":
            validate_not_started(args.results_root, args.case)
            print("NOT STARTED: " + " ".join(args.case))
            return 0
        if args.command == "resolve-drain":
            target = resolve_drain_target(args.project_root, args.results_root)
            print(json.dumps(_target_as_json(target), ensure_ascii=False, indent=2))
            return 0
        if args.command == "check-predrained":
            evidence = validate_predrained_state(
                args.project_root,
                args.results_root,
                args.quarantine_root,
                expected_rows=args.expected_rows,
            )
            print(json.dumps(asdict(evidence), ensure_ascii=False, indent=2))
            return 0
        if args.command == "record-predrained":
            evidence = record_predrained_state(
                args.project_root,
                args.results_root,
                args.quarantine_root,
                args.manifest,
                expected_rows=args.expected_rows,
            )
            print(
                f"RECORDED PREDRAINED: m06 complete, canonical m08/m11 absent, "
                f"quarantine={evidence.quarantine_root}"
            )
            return 0
        if args.command == "verify-predrained":
            evidence = verify_predrained_manifest(
                args.manifest, expected_rows=args.expected_rows
            )
            print(
                f"PREDRAINED EVIDENCE OK: quarantine={evidence.quarantine_root}"
            )
            return 0
        if args.command == "stop-drain":
            target = stop_outer_at_boundary(
                args.project_root, args.results_root, args.manifest
            )
            print(
                f"STOPPED outer pid={target.outer.pid} start_ticks={target.outer.start_ticks}; "
                f"runner pid={target.runner.pid} continues m06"
            )
            return 0
        if args.command == "verify-drain":
            verify_target(
                load_target(args.manifest),
                require_stopped=args.require_stopped,
                allow_runner_gone=args.allow_runner_gone,
            )
            print("DRAIN IDENTITY OK")
            return 0
        if args.command == "drain-ready":
            run_dir = assert_drain_ready(
                args.results_root,
                args.manifest,
                expected_rows=args.expected_rows,
            )
            print(f"DRAIN READY: {run_dir}")
            return 0
        if args.command == "finalize-drain":
            target = load_target(args.manifest)
            finalize_live_drain(
                args.results_root,
                args.manifest,
                expected_rows=args.expected_rows,
            )
            print(
                f"FINALIZED exact stopped outer: pid={target.outer.pid}, "
                "m08/m11 remain not-started"
            )
            return 0
        if args.command == "resume-drain":
            resume_outer(args.results_root, args.manifest)
            print("RESUMED exact outer runner")
            return 0
    except BarrierWaiting as exc:
        print(f"WAITING: {exc}")
        return 3
    except ContractError as exc:
        print(f"CONTRACT ERROR: {exc}")
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
