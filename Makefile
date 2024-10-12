install-client:
	poetry install --only main,client

install-server:
	poetry install --only main,server

# ---------
# Dev rules
# ---------

dev-install:
	poetry install --with dev

dev-lint:
	ruff format; \
	ruff check --fix; \
	mypy .
