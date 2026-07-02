# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Service:** Oulu University Lehmus AI
- **Model:** Gemma4 26B MoE

# Purpose
This document serves as a starting point for code analysis using Lehmus AI. The purpose of this project is to explore the limits of locally hosted LLM-based code analysis.

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
