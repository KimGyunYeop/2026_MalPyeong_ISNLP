# 검증 데이터

`official_validation_400.jsonl`은 타 서버 Docker 검증기가 사용하는 공개 validation 400편
fixture다. bundle checksum과 평가 입력 순서의 기준이므로 내용을 수정하거나 재직렬화하지 않는다.

이 데이터는 공개 endpoint·parser·점수열 동일성을 재현하기 위한 것이며 hidden test나 공식 LLM
Judge를 포함하지 않는다. 현행 c02+Gemma-teacher 증류 A.X image는 local 400/400과 push를
완료했으며, 이 fixture를 사용한 외부 단일 L40S digest 검증은 아직 미완료다.
