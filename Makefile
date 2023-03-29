.PHONY: test
test:
	python -m pytest

.PHONY: type
type:
	python -m pyright src/gretel_trainer/benchmark src/gretel_trainer/relational tests

.PHONY: multi
multi:
	-python -m pyright src/gretel_trainer/relational tests/relational/
	python -m pytest tests/relational/

.PHONY: multilint
multilint:
	python -m isort src/gretel_trainer/relational tests/relational
	python -m black src/gretel_trainer/relational tests/relational

.PHONY: bench
bench:
	python -m pyright src/gretel_trainer/benchmark tests/benchmark
	python -m pytest tests/benchmark/

.PHONY: benchlint
benchlint:
	python -m isort src/gretel_trainer/benchmark tests/benchmark
	python -m black src/gretel_trainer/benchmark tests/benchmark
