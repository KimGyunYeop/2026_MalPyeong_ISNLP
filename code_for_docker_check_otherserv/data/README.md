# 검증 데이터

`official_validation_400.jsonl`은 타 서버 Docker 검증기가 사용하는 공개 validation 400편
fixture다. 평가 입력 순서의 기준이므로 내용을 수정하거나 재직렬화하지 않는다.
대회 데이터이므로 Git에는 올리지 않는다.

이 데이터는 공개 endpoint·parser·점수열 동일성을 재현하기 위한 것이며 hidden test나 공식 LLM
Judge를 포함하지 않는다.
