# ==============================================================
#					RAG AGAINST THE MACHINE
# ==============================================================

NAME = rag_against_the_machine
PYTHON = python3
UV = $(shell command -v uv 2> /dev/null || echo $(HOME)/.local/bin/uv)
UV_PROJECT_ENVIRONMENT ?= .venv

SRC = src

K=10
CONTEXT_LENGTH=2000

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
	@curl -s http://localhost:11434 >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then \
		echo "Starting ollama server..."; \
		export OLLAMA_NUM_PARALLEL=2; \
		export OLLAMA_FLASH_ATTENTION=1; \
		ollama serve > /dev/null 2>&1 & \
		until curl -s http://localhost:11434 >/dev/null 2>&1; do sleep 1; done; \
	fi

stop-server:
	@curl -s http://localhost:11434 >/dev/null 2>&1; \
	if [ $$? -eq 0 ]; then \
		echo "Stopping ollama server..."; \
		pkill ollama; \
	fi

test-all:
	@$(MAKE) test-index --no-print-directory
	@$(MAKE) test-search --no-print-directory
	@$(MAKE) test-answer --no-print-directory
	@$(MAKE) test-evaluate --no-print-directory

test-index:
	@echo "Testing index..."
	@$(UV) run python -m $(SRC) index --max_chunk_size=$(CONTEXT_LENGTH)

test-search:
	@echo "Testing search..."
	@$(UV) run python -m $(SRC) search_dataset \
	--dataset_path data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
	--k $(K) \
	--save_directory data/output/search_results

test-answer: start-server
	@echo "Testing answer..."
	@$(UV) run python -m $(SRC) answer_dataset \
	--student_search_results_path data/output/search_results/dataset_docs_public.json \
	--save_directory data/output/search_results_and_answer

test-evaluate:
	@echo "Testing evaluate..."
	@$(UV) run python -m $(SRC) evaluate \
	--student_answer_path data/output/search_results/dataset_docs_public.json \
	--dataset_path data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
	--k $(K) \
	--max_context_length $(CONTEXT_LENGTH)

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

.PHONY: all install test debug lint lint-strict clean fclean re sync start-server
.DEFAULT_GOAL = all
