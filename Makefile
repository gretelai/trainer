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

.PHONY: b2
b2:
	python -m pyright src/gretel_trainer/b2 tests/b2
	python -m pytest tests/b2/

.PHONY: b2lint
b2lint:
	python -m isort src/gretel_trainer/b2 tests/b2
	python -m black src/gretel_trainer/b2 tests/b2
