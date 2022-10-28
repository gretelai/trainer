.PHONY: test
test:
	python -m pytest

.PHONY: type
type:
	python -m pyright src/gretel_trainer/benchmark tests/test_benchmark.py
