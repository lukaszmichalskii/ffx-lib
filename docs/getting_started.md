# Quickstart Guide

> ⚠️ **Release Tag: `1.0.0-alpha`**

---

### 1. Installation

Clone the repository and install the Python frontend compiler in editable mode:

```bash
git clone https://github.com/lukaszmichalskii/ffx-lib.git
cd ffx-lib

# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install ffx-compiler
pip install -e python/ffx/compiler

# Install PyTorch for demo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnxscript
```

---

### 2. Export VGG11 Model

Run the export script to generate the sample ONNX computational graph:

```bash
python3 scripts/vgg11.py
```

*Output artifact:* `models/vgg11.onnx`

---

### 3. AOT Compilation & Optimization

Compile the ONNX model into optimized header-only C++ source code. Passing **`-O1`** enables static shape folding and activation kernel fusion:

```bash
# -O1: run Level 1 optimization (shape folding & kernel fusion)
ffx-compiler --model models/vgg11.onnx -O1 --verbose
```

#### Command Flags

* `--model <path>`: Path to input ONNX model.
* `-O1`: Enables kernel fusion and static shape folding passes.
* `--verbose`: Displays graph IR summaries and compiler debug logs.

> 💡 *Run `ffx-compiler --help` for full CLI option details.*

---

### 4. Generated Artifacts

The AOT compiler outputs standalone, dependency-less C++ files directly into `./codegen`:

```text
codegen/
├── model.h     # Synthesized C++ execution graph header
└── model.data  # Binary packed weight arrays
```

---

### 5. C++ Runtime Build and Install

The `1.0.0-alpha` release introduces a standardized `main.cc` entry point to streamline local debugging and runtime execution. It wraps generated model headers (`codegen/model.h`) inside a portable CLI runner, allowing developers to quickly verify weight loading, run forward passes, and inspect performance without writing custom boilerplate.


```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install -DFFX_BUILD_TESTING=OFF
cmake --install build
```

---

### 6. C++ Runtime Execution

Copy the generated artifacts into `codegen_test/` and build:
```bash
cp codegen/main.cc codegen_test/main.cc
cp codegen/model.h codegen_test/model.h
cp codegen/model.data codegen_test/model.data

cd codegen_test
cmake -S . -B build -DCMAKE_PREFIX_PATH=../install
cmake --build build
./build/bin/main_<backend>  # cuda, hip, serial, tbb, omp2
```

