# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Service:** Oulu University Lehmus AI
- **Model:** Gemma4 26B MoE

# Purpose
This document serves as a starting point for code analysis using Lehmus AI. The purpose of this project is to explore the limits of locally hosted LLM-based code analysis.

# Compilation and Building
To build Astaroth, follow these steps in the base directory:

1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make -j`

### Dependencies
Requires `gcc`, `cmake`, `flex`, `bison`, and either `nvcc` (NVIDIA) or `hipcc` (AMD).

### Common CMake Options
| Option | Description | Default |
|--------|-------------|---------|
| `DOUBLE_PRECISION` | Generates double precision code | `ON` |
| `MPI_ENABLED` | Enables MPI support | `OFF` |
| `USE_HIP` | Use HIP instead of CUDA | `OFF` |
| `BUILD_SAMPLES` | Builds projects in samples subdirectory | `ON` |
| `BUILD_TESTS` | Builds Astaroth test samples | `OFF` |

# Root Directory File Descriptions
* **CMakeLists.txt**: Main build configuration file for the project.
* **CMakePresets.json**: Configuration file for defining CMake build presets.
* **CONTRIBUTING.md**: Guidelines and instructions for contributing to the project.
* **Dockerfile**: Instructions for building a container image for the environment.
* **LICENCE.md**: The legal licensing terms for the codebase.
* **README.md**: Overview and introductory information about the project.
* **bitbucket-pipelines.yml**: CI/CD configuration for Bitbucket Pipelines.
* **doxyfile**: Configuration file for generating documentation using Doxygen.
* **paper.bib**: BibTeX file containing bibliographic references for academic papers.
* **paper.md**: Markdown version or draft of an academic paper related to the project.
* **pyproject.toml**: Configuration for Python project management and build systems.
* **requirements.txt**: List of Python dependencies required for the project.
* **sourceme.sh**: Shell script used to set up environment variables or paths.
* **uv.lock**: Lockfile for the `uv` Python package manager to ensure reproducible environments.

# Project Directory Tree
```text
.
├── 3rdparty
│   └── eigen
│       ├── blas
│       ├── ci
│       ├── cmake
│       ├── unsupported
│       ├── Eigen
│       ├── test
│       └── ...
├── acc-comm
├── acc-runtime
│   ├── acc
│   ├── api
│   ├── built-in
│   ├── Pencil
│   ├── samples
│   ├── LICENCE.md
│   ├── CMakeLists.txt
│   ├── README.md
│   └── stdlib
├── analysis
│   ├── python
│   ├── test_tools
│   └── viz_tools
├── cmake
├── config
├── deprecated
├── doc
├── include
├── pilot
├── runtime_compilation
├── samples
├── src
│   ├── core
│   └── utils
├── stdlib
├── test
├── tools
├── CMakeLists.txt
├── CMakePresets.json
├── CONTRIBUTING.md
├── Dockerfile
├── LICENCE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── sourceme.sh
├── uv.lock
└── ...
```
*(Note: This is a high-level summary. Use `find .` for the full exhaustive list.)*
