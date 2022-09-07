.DEFAULT_GOAL := help
.PHONY: coverage deps help lint publish push test tox

coverage:  ## Run tests with coverage
	python3 -m coverage erase
	python3 -m coverage run --include=mixupy/* -m pytest -ra
	python3 -m coverage report -m

deps:  ## Install dependencies
	python3 -m pip install --upgrade pip
	python3 -m pip install black coverage flake8 flit mccabe pylint pytest tox tox-gh-actions

lint:  ## Lint and static-check
	python3 -m flake8 mixupy
	python3 -m pylint mixupy

publish:  ## Publish to PyPi
	python3 -m flit publish

push:  ## Push code with tags
	git push && git push --tags

test:  ## Run tests
	python3 -m pytest -ra

tox:   ## Run tox
	python3 -m tox

help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
