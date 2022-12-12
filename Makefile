.PHONY: test
test:
	python -m pytest

.PHONY: type
type:
	python -m pyright src/gretel_trainer/benchmark src/gretel_trainer/relational tests

.PHONY: multi
multi:
	-python -m pyright src/gretel_trainer/relational tests/test_relational.py
	python -m pytest tests/test_relational.py
