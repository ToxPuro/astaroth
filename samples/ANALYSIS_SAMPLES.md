# ⚠️ WARNING ⚠️
# **This is a LLM based analysis.**
# **THE BRANCH SHOULD NEVER BE MERGED TO master OR develop BRANCHES!**

# LLM Information
- **Service:** Oulu University Lehmus AI
- **Model:** Gemma4 26B MoE

# Purpose
This document provides a collection of curated analysis tasks, prompts, and case studies specifically focused on exploring the variety, implementation, and patterns found within the `samples/` directory.

# Note
> **Note:** The `samples/standalone_mpi/` directory is the main standalone implementation. The `samples/standalone/` directory is considered deprecated.

# Analysis Use Cases & Templates (Samples Focus)

## 1. DSL & Stencil Implementation Analysis
**Goal:** Understand how the domain-specific language (DSL) is used to define physical simulations.
**Target Directories:** `samples/advection-example/`, `samples/les/`, `samples/heat-equation/`
**Sample Prompt:** *"Examine the `.ac` files in `samples/advection-example/DSL/`. How are the stencil coefficients and grid dimensions defined for this specific simulation?"*

## 2. Parallelism & Scalability Patterns
**Goal:** Identify how different parallelization paradigms (MPI vs. Single-GPU/Multi-GPU) are implemented in the samples.
**Target Directories:** `samples/mpi-io/`, `samples/standalone_mpi/`, `samples/multikerneltest/`
**Sample Prompt:** *"Compare the MPI initialization and communication patterns in `samples/mpi-io/` versus `samples/standalone_mpi/`. How does the responsibility of data distribution differ between them?"*

## 3. Benchmark & Performance Profiling
**Goal:** Understand the structure and setup of performance-testing samples.
**Target Directories:** `samples/benchmark-thrust/`, `samples/microbenchmark/`, `samples/benchmark-device/`
**Sample Prompt:** *"Analyze `samples/benchmark-thrust/main.cu`. What specific kernel operations are being benchmarked, and how is the timing measurement implemented?"*

## 4. Simulation Workflow & Configuration
**Goal:** Trace how input configurations and runtime parameters drive a simulation.
**Target Directories:** `samples/plasma-meets-ai-workshop/`, `samples/tfm-mpi/`
**Sample Prompt:** *"In `samples/tfm-mpi/`, how are the simulation parameters loaded from configuration files? Trace the path from file read to kernel parameter application."*

# Running Analysis Samples
To use these templates effectively:
1. Use the `Project Directory Tree` in `ANALYSIS_BASE.md` to locate the target sample subdirectory.
2. Use `grep` or `Read` to extract specific code snippets relevant to the prompt.
3. Execute the analysis task to understand the implementation pattern.

# Repository Reference
- For codebase overview: `ANALYSIS_BASE.md`
- For build instructions: `README.md`
