# ==============================================================
#					RAG AGAINST THE MACHINE
# ==============================================================

NAME = rag_against_the_machine
PYTHON = python3
UV = $(shell command -v uv 2> /dev/null || echo $(HOME)/.local/bin/uv)
UV_PROJECT_ENVIRONMENT ?= .venv

SRC = src

DEPS =	bm25s[full] \
		chromadb \
		dspy \
		fire \
		flake8 \
		langchain \
		langchain-text-splitters \
		mypy \
		ollama \
		pudb \
		pydantic \
		PyStemmer \
		transformers \
		tqdm

all: install

install:
	@if [ ! -e $(UV) ]; then \
		echo "installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; \
	fi
	@$(UV) add $(DEPS)
	@$(MAKE) sync --no-print-directory

sync:
	@echo "Syncing dependencies..."
	@$(UV) sync

start-server:
	@curl http://localhost:11434 >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then \
		echo "Starting ollama server..."; \
		gnome-terminal -- bash -c "ollama serve; exec bash"; \
	fi

run: start-server
	@$(UV) run python -m $(SRC)


debug:
	@$(UV) run python -m pudb -m $(SRC)

lint:
	@echo "Running flake8..."
	@$(UV) run flake8 $(SRC)
	@echo "Running mypy..."
	@$(UV) run mypy $(SRC)


lint-strict:
	@echo "Running flake8..."
	@$(UV) run flake8 $(SRC)
	@echo "Running mypy --strict..."
	@$(UV) run mypy $(SRC) --strict

clean:
	@if [ -n "$$(find . -type d \( -name ".mypy_cache" -o -name "__pycache__" \
	-o -name ".uv_cache" -o -name ".pytest_cache" \) -print -quit)" ]; then \
		echo "Cleaning cache files..."; \
		find . -type d \( -name ".mypy_cache" -o -name "__pycache__" -o -name \
		".uv_cache" -o -name ".pytest_cache" \) -exec rm -rf {} +; \
	fi

fclean: clean
	@echo "Removing virtual environment..."
	$(RM) -r $(UV_PROJECT_ENVIRONMENT)

re: fclean all

.PHONY: all install run debug lint lint-strict clean fclean re sync start-server
.DEFAULT_GOAL = all
