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

## Subdirectory Reference

### Benchmarks & Library Tests
- `samples/benchmark/`: Basic performance benchmarking for Astaroth core.
- `samples/benchmark-device/`: Benchmarking kernel operations on individual devices.
- `samples/benchmark-node/`: Node-level performance benchmarking.
- `samples/benchmark-thrust/`: Benchmarking using NVIDIA Thrust library integration.
- `samples/microbenchmark/`: Fine-grained microbenchmarking of specific operations.
- `samples/microbenchmark-nn/`: Microbenchmarking neural network backend integrations (cuDNN/miopen).
- `samples/mpi_reduce_bench/`: Benchmarking MPI reduction operations.
- `samples/genbenchmarkscripts/`: Utilities for generating benchmark scripts.
- `samples/cubtest/`: Tests CUB/HIPCU library integration for segmented reductions.
- `samples/cpptest/`: Validates Astaroth core functionality using C++ integration steps.
- `samples/ctest/`: A basic C-based integration test for verifying mesh loading and reduction.
- `samples/fortrantest/`: Fortran language interoperability tests.

### DSL & Stencil Examples
- `samples/advection-example/`: Demonstrates basic advection of a concentration field using the DSL.
- `samples/blur/`: Simple stencil-based blurring example.
- `samples/convection_kramers/`: Simulation of convection using the Kramers model.
- `samples/heat-equation/`: Implementation of the heat equation using both native and deep learning backends.
- `samples/les/`: Large Eddy Simulation (LES) implementations.
- `samples/stencil-loader/`: Tests for the stencil loading mechanism.

### Diagnostics, Tracing & Testing
- `samples/boundcond_test/`: Tests for boundary condition implementations.
- `samples/devicetest/`: Basic device availability and functionality tests.
- `samples/dconst-race-condition-test/`: Tests for potential race conditions in device constants.
- `samples/mpi_fullgriderror_test/`: Tests for error handling in full-grid MPI operations.
- `samples/mpitest/`: General MPI integration and functionality tests.
- `samples/multikerneltest/`: Tests for multi-kernel execution patterns.
- `samples/stress/`: Stress testing for Astaroth components.
- `samples/taskgraph_example/`: Basic example of Astaroth task graph usage.
- `samples/taskgraph_print/`: Utility for printing the task graph structure.
- `samples/taskgraph_test/`: Tests for the task graph execution engine.
- `samples/taskgraph_trace/`: Traces the Astaroth task graph and dependencies for MPI-enabled builds.

### TFM (Test Field Model) Simulations
- `samples/tfm/`: A comprehensive suite for Test Field Model simulations, including MHD and pipeline benchmarking.
- `samples/tfm/mhd/`: Contains DSL-based Magnetohydrodynamics (MHD) definitions for the TFM pipeline.
- `samples/tfm-mpi/`: MPI-enabled implementation of the TFM pipeline.

### Python & Workshop Tools
- `samples/pyastaroth/`: Python bindings and related tools for Astaroth.
- `samples/pyastaroth_generated/`: Auto-generated files for Python bindings.
- `samples/plasma-meets-ai-workshop/`: Collection of simulation examples and models for workshop use.

### Parallelism & Scalability
- `samples/bwtest-mpi/`: Bandwidth testing in MPI environments.
- `samples/mpi-io/`: Tests for MPI-based I/O operations.
- `samples/mpi-io-multithreaded/`: Tests for multi-threaded MPI-based I/O.
- `samples/standalone/`: Deprecated standalone implementation.
- `samples/standalone_mpi/`: The primary standalone MPI implementation.
- `samples/stress/DSL/`: DSL-specific stress testing.

## Subdirectory Reference

### Benchmarks & Library Tests
- `samples/cubtest/`: Tests CUB/HIPCU library integration for segmented reductions.
- `samples/cpptest/`: Validates Astaroth core functionality using C++ integration steps.
- `samples/ctest/`: A basic C-based integration test for verifying mesh loading and reduction.

### Diagnostics & Tracing
- `samples/taskgraph_trace/`: Traces the Astaroth task graph and dependencies for MPI-enabled builds.

### TFM (Test Field Model) Simulations
- `samples/tfm/`: A comprehensive suite for Test Field Model simulations, including MHD and pipeline benchmarking.
- `samples/tfm/mhd/`: Contains DSL-based Magnetohydrodynamics (MHD) definitions for the TFM pipeline.
