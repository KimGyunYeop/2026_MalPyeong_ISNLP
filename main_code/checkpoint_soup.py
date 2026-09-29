"""여러 체크포인트의 **가중치**를 평균해 하나의 체크포인트로 만든다.

왜 예측 앙상블이 아니라 가중치 평균인가
---------------------------------------
예측 앙상블은 이미 측정했고 전부 졌다. 형제 8개의 오차 상관이 0.9981~0.9993,
백본을 바꿔도 0.9488~0.9574였다. 상관이 이렇게 높으면 평균해도 분산이 줄지 않는다.
가중치 평균은 다른 연산이다. 같은 초기값에서 출발한 run들이 손실 표면의 **같은
분지(basin)** 안에 있으면, 그 가중치들의 평균이 각각보다 평평한 지점에 놓여
일반화가 좋아질 수 있다(model soup). 예측 앙상블처럼 추론 비용이 늘지도 않는다.

전제: 같은 seed에서 나온 run이어야 한다. LoRA A는 seed로 초기화되므로 seed가
다르면 서로 다른 분지에 있고 평균이 무의미하다. `--require-same-seed`가 기본이다.

LoRA를 정확히 평균하는 법
-------------------------
델타는 ``delta_i = (alpha/r) · B_i @ A_i``다. A와 B를 따로 평균하면
``mean(B) @ mean(A) != mean(B @ A)``라 원하는 값이 아니다. 대신 이어 붙인다.

    A_cat = concat([A_1, ..., A_k], dim=0)            shape (r·k, in)
    B_cat = concat([B_1/k, ..., B_k/k], dim=1)        shape (out, r·k)
    r' = r·k,  alpha' = alpha·k     (비율 alpha/r가 보존된다)

    delta_cat = (alpha'/r') · B_cat @ A_cat = (1/k) · Σ (alpha/r) · B_i @ A_i

즉 델타의 **정확한** 산술평균이며, 결과물은 여전히 평범한 LoRA 어댑터다.
헤드(`heads.pt`)는 중첩 dict이므로 float 텐서만 재귀적으로 평균한다.

주의: 복사해 오는 ``config.json``의 ``lora_r``은 멤버 값(32) 그대로 남는다.
추론 경로는 ``PeftModel.from_pretrained``가 어댑터 폴더의 ``adapter_config.json``
(r = 32·k)을 읽으므로 문제가 없고, ``verify_loaded_peft_adapter_exact``가 저장값과
메모리값을 대조한다. ``config.json``의 ``lora_r``은 **학습 시 LoRA를 새로 만들 때만**
쓰이므로, 이 수프 체크포인트에서 이어 학습하면 안 된다. 평가·서빙 전용이다.

사용:
    python -m main_code.checkpoint_soup --output <dir> <ckpt> <ckpt> [<ckpt> ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file, save_file


class SoupError(ValueError):
    """수프를 만들 수 없는 입력."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_directory(checkpoint: Path) -> Path | None:
    """체크포인트가 속한 run 디렉터리(= resolved_config.json이 있는 곳)."""

    current = checkpoint.resolve()
    for _ in range(4):
        if (current / "resolved_config.json").is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def checkpoint_provenance(checkpoint: Path) -> dict[str, Any]:
    run = _run_directory(checkpoint)
    resolved = _read_json(run / "resolved_config.json") if run else {}
    selection_path = checkpoint / "selection.json"
    selection = _read_json(selection_path) if selection_path.is_file() else {}
    return {
        "checkpoint": str(checkpoint.resolve()),
        "run": str(run) if run else None,
        "seed": resolved.get("seed"),
        "lora_r": resolved.get("lora_r"),
        "lora_alpha": resolved.get("lora_alpha"),
        "max_train_steps": resolved.get("max_train_steps"),
        "global_step": selection.get("global_step"),
        "selection_metric": selection.get("metric"),
        "selection_metric_value": selection.get("metric_value"),
    }


def _assert_compatible(
    provenances: Sequence[dict[str, Any]], *, require_same_seed: bool
) -> None:
    seeds = {item["seed"] for item in provenances}
    if require_same_seed and len(seeds) > 1:
        raise SoupError(
            f"서로 다른 seed {sorted(map(str, seeds))}를 섞으려 한다. LoRA A는 seed로 "
            "초기화되므로 다른 분지에 있고 가중치 평균이 무의미하다. 정말 원하면 "
            "--allow-mixed-seeds를 준다"
        )
    for field in ("lora_r", "lora_alpha"):
        values = {item[field] for item in provenances}
        if len(values) > 1:
            raise SoupError(f"{field}가 서로 다르다: {sorted(map(str, values))}")


def _average_head_node(nodes: Sequence[Any], *, path: str) -> Any:
    """heads.pt는 평평한 state_dict가 아니라 중첩 dict다.

    실제 구조는 ``{"schema_version": 2, "scoring_state": {...44 tensors...}}``이다.
    float 텐서만 평균하고, 나머지는 멤버끼리 **같은지 확인한 뒤** 그대로 둔다.
    스칼라를 조용히 첫 멤버 값으로 덮으면 서로 다른 계약의 체크포인트를 섞은
    사실이 결과에 드러나지 않는다.
    """

    first = nodes[0]
    if isinstance(first, dict):
        keys = set(first)
        for node in nodes[1:]:
            if not isinstance(node, dict) or set(node) != keys:
                raise SoupError(f"heads.pt 구조가 멤버마다 다르다: {path or '<root>'}")
        return {
            key: _average_head_node(
                [node[key] for node in nodes], path=f"{path}.{key}" if path else key
            )
            for key in sorted(keys)
        }
    if isinstance(first, torch.Tensor):
        shapes = {tuple(node.shape) for node in nodes}
        if len(shapes) > 1:
            raise SoupError(f"헤드 {path!r}의 shape가 서로 다르다: {shapes}")
        if not first.is_floating_point():
            # 정수 버퍼(예: step 카운터)는 평균이 의미 없다.
            return first.clone()
        stacked = torch.stack([node.float() for node in nodes], dim=0)
        return stacked.mean(dim=0).to(first.dtype)
    for node in nodes[1:]:
        if node != first:
            raise SoupError(
                f"heads.pt의 비텐서 값 {path!r}이 멤버마다 다르다: {first!r} != {node!r}"
            )
    return first


def _averaged_heads(checkpoints: Sequence[Path]) -> Any:
    loaded = [
        torch.load(checkpoint / "heads.pt", map_location="cpu")
        for checkpoint in checkpoints
    ]
    return _average_head_node(loaded, path="")


def _concatenated_adapter(
    checkpoints: Sequence[Path],
) -> tuple[dict[str, torch.Tensor], int]:
    """델타의 정확한 평균을 담은 rank r·k 어댑터."""

    count = len(checkpoints)
    states = [
        load_file(str(checkpoint / "adapter" / "adapter_model.safetensors"))
        for checkpoint in checkpoints
    ]
    names = set(states[0])
    for index, state in enumerate(states[1:], start=1):
        if set(state) != names:
            missing = sorted(names.symmetric_difference(state))[:5]
            raise SoupError(
                f"어댑터 텐서 목록이 다르다 ({checkpoints[index]}): {missing}"
            )
    merged: dict[str, torch.Tensor] = {}
    for name in sorted(names):
        tensors = [state[name] for state in states]
        if ".lora_A." in name:
            merged[name] = torch.cat([tensor.float() for tensor in tensors], dim=0)
        elif ".lora_B." in name:
            merged[name] = torch.cat(
                [tensor.float() / count for tensor in tensors], dim=1
            )
        else:
            # lora_embedding_* 나 modules_to_save 같은 dense 항목은 그냥 평균한다.
            stacked = torch.stack([tensor.float() for tensor in tensors], dim=0)
            merged[name] = stacked.mean(dim=0)
        merged[name] = merged[name].to(tensors[0].dtype)
    return merged, count


def build_soup(
    checkpoints: Sequence[Path],
    output: Path,
    *,
    require_same_seed: bool = True,
) -> dict[str, Any]:
    if len(checkpoints) < 2:
        raise SoupError("수프에는 체크포인트가 둘 이상 필요하다")
    provenances = [checkpoint_provenance(path) for path in checkpoints]
    _assert_compatible(provenances, require_same_seed=require_same_seed)

    base = checkpoints[0]
    adapter_config = _read_json(base / "adapter" / "adapter_config.json")
    if adapter_config.get("use_rslora"):
        # rsLoRA는 alpha/sqrt(r)로 스케일하므로 r을 k배 하면 비율이 보존되지 않는다.
        raise SoupError("use_rslora=true 어댑터는 이 이어 붙이기로 평균할 수 없다")

    output.mkdir(parents=True, exist_ok=True)
    (output / "adapter").mkdir(exist_ok=True)
    merged, count = _concatenated_adapter(checkpoints)
    save_file(merged, str(output / "adapter" / "adapter_model.safetensors"))

    soup_config = dict(adapter_config)
    soup_config["r"] = int(adapter_config["r"]) * count
    soup_config["lora_alpha"] = int(adapter_config["lora_alpha"]) * count
    # rank_pattern/alpha_pattern이 있으면 위 스칼라와 충돌한다. 이 저장소의
    # 어댑터는 둘 다 비어 있고, 비어 있지 않으면 조용히 틀리므로 막는다.
    for key in ("rank_pattern", "alpha_pattern"):
        if adapter_config.get(key):
            raise SoupError(f"{key}가 비어 있지 않은 어댑터는 지원하지 않는다")
    with (output / "adapter" / "adapter_config.json").open("w", encoding="utf-8") as handle:
        json.dump(soup_config, handle, ensure_ascii=False, indent=2)

    torch.save(_averaged_heads(checkpoints), output / "heads.pt")
    shutil.copy2(base / "config.json", output / "config.json")
    if (output / "tokenizer").exists():
        shutil.rmtree(output / "tokenizer")
    shutil.copytree(base / "tokenizer", output / "tokenizer")

    manifest = {
        "schema_version": 1,
        "kind": "checkpoint_soup",
        "members": provenances,
        "member_count": count,
        "lora_r": soup_config["r"],
        "lora_alpha": soup_config["lora_alpha"],
        "delta_semantics": "exact_arithmetic_mean_of_member_deltas",
        "head_semantics": "recursive_arithmetic_mean_of_float_tensors",
        "config_source": str(base.resolve()),
    }
    with (output / "soup_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-mixed-seeds",
        action="store_true",
        help="seed가 다른 체크포인트를 섞는다(보통 성능이 무너진다)",
    )
    args = parser.parse_args(argv)
    manifest = build_soup(
        args.checkpoints, args.output, require_same_seed=not args.allow_mixed_seeds
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
