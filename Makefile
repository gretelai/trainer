.PHONY: test
test:
	python -m pytest

.PHONY: type
type:
	python -m pyright src tests

.PHONY: multi
multi: type
	python -m pytest tests/relational/

.PHONY: bench
bench: type
	python -m pytest tests/benchmark/
