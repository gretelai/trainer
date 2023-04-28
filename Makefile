.PHONY: test
test:
	python -m pytest

.PHONY: type
type:
	python -m pyright src tests

.PHONY: multi
multi:
	-python -m pyright src/gretel_trainer/relational tests/relational/
	python -m pytest tests/relational/

.PHONY: multilint
multilint:
	python -m isort src/gretel_trainer/relational tests/relational
	python -m black src/gretel_trainer/relational tests/relational
