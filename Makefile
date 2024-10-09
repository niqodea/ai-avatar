install-client:
	poetry install --only main,client

install-server:
	poetry install --only main,server

run-server:
	uvicorn ai_avatar.server:app $(filter-out $@,$(MAKECMDGOALS))

run-client:
	python3 -m ai_avatar.client $(filter-out $@,$(MAKECMDGOALS))

# ---------
# Dev rules
# ---------

dev-install:
	poetry install --with dev

dev-lint:
	ruff format; \
	ruff check --fix; \
	mypy .
