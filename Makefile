.PHONY: test
test:
	python -m pytest

.PHONY: type
type:
	python -m pyright src/gretel_trainer/benchmark tests/test_benchmark.py

.PHONY: multi
multi:
	python -m pyright src/gretel_trainer/relational tests/test_relational.py && python -m pytest tests/test_relational.py
