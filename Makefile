MAKEFLAGS += --no-print-directory

BUILD_DIR ?= cmake-build
BUILD_TYPE ?= Release
FFX_BUILD_TESTING ?= ON

PYTHON_DIR = python/ffx

.PHONY: all configure build install install-cpp install-python test test-cpp test-python lint lint-cpp lint-python format format-cpp format-python

$(BUILD_DIR)/CMakeCache.txt:
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_INSTALL_PREFIX=install -DFFX_BUILD_TESTING=$(FFX_BUILD_TESTING)

configure: $(BUILD_DIR)/CMakeCache.txt

build:
	@cmake --build $(BUILD_DIR)

install-cpp: build
	@cmake --install $(BUILD_DIR)

install-python:
	@python3 -m pip install --upgrade pip
	@python3 -m pip install -e "$(PYTHON_DIR)/compiler[development]"

install: install-cpp install-python

test-cpp:
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

test-nn: build
	@ctest --test-dir $(BUILD_DIR) -L "nn" --output-on-failure

test-algorithm: build
	@ctest --test-dir $(BUILD_DIR) -L "algorithm" --output-on-failure

test-framework: build
	@ctest --test-dir $(BUILD_DIR) -L "framework" --output-on-failure

test-python:
	@python3 -m pytest $(PYTHON_DIR) --no-header -v

test: test-cpp test-python

lint-cpp:
	@git ls-files "**/*.cc" "**/*.h" | xargs -r clang-format --dry-run --Werror
	@{ git ls-files "**/CMakeLists.txt" "*.cmake"; } | xargs -r cmake-format --check

lint-python:
	@python3 -m ruff check $(PYTHON_DIR)
	@python3 -m ruff format --check $(PYTHON_DIR)

lint: lint-cpp lint-python

format-cpp:
	@git ls-files "**/*.cc" "**/*.h" | xargs -r clang-format -i
	@{ git ls-files "**/CMakeLists.txt" "*.cmake"; } | xargs -r cmake-format -i

format-python:
	@python3 -m ruff check $(PYTHON_DIR) --fix
	@python3 -m ruff format $(PYTHON_DIR)

format: format-cpp format-python
