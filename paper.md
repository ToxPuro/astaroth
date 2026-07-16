---
title: 'Astaroth: A scientific computing framework for accelerating stencil computations.'
tags:
  - GPU
  - DSL
  - C/C++
  - scientific computing
  - high performance computing
  - astrophysics

authors:
  - name: Touko Puro
    orcid: 0009-0008-8632-0385
    corresponding: true
    affiliation: 1
  - name: Johannes Pekkilä
    orcid: 0000-0002-1974-7150
    affiliation: 1
  - name: Miikka S. Väisälä
    orcid: 0000-0002-8782-4664
    affiliation: 2
  - name: Oskar Lappi
    orcid: 0000-0003-3182-8161
    affiliation: 3
  - name: Matthias Rheinhardt
    orcid: 0000-0001-9840-5986
    affiliation: 1
  - name: Hsien Shang
    orcid: 0000-0001-8385-9838
    affiliation: 4
  - name: Maarit Korpi-Lagg
    orcid: 0000-0002-9614-2200
    affiliation: 1

affiliations:
 - name:  Aalto University, Finland
   index: 1
 - name:  University of Oulu, Finland
   index: 2
 - name:  University of Helsinki, Finland
   index: 3
 - name:  Academia Sinica Institute of Astronomy and Astrophysics, Taiwan
   index: 4

date: 20 May 2026
bibliography: paper.bib

---

# Summary

Stencil computations[^stencil_footnote] are one of the bedrocks of high-performance scientific simulations, forming the core of many partial differential equation (PDE) and numerical linear algebra solvers.
In recent years, GPUs have become the primary compute platform for data-parallel applications in high-performance computing, and it is difficult to run large simulations without them.
`Astaroth` is a GPU framework for stencil computations, that has been developed to address this problem of scalable scientific computing.
`Astaroth` provides its own domain specific language (DSL), in which researchers can express such computations without having to focus on technical implementation details.
It can run efficiently both on CUDA- and HIP-based environments, and for testing purposes on CPUs.
While stencils are the core of `Astaroth`, it also accelerates other operations like reductions (e.g. sums), simple ray-tracing, and integrates with libraries performing GPU-accelerated Fourier transforms, all of which are important for simulations on structured grids.
`Astaroth` is optimized for multiphysics use cases and has primarily been used for turbulent astrophysical plasma simulations.

# Statement of need

Much of the software used for scientific computing is written for CPUs, and has
to be ported to GPUs to run larger problems with decent times-to-solution.
`Astaroth` has been developed to solve this problem for the subset of
scientific software that relies heavily on stencil computations and compared to existing solutions is
specifically optimized for computations needed in multiphysics simulations.
`Astaroth`'s domain specific language (DSL) is designed to enable researchers to rewrite existing PDE solvers or to write
completely new ones.

As examples of this, `Astaroth` has been used to write a partial differential equation (PDE) solver for astrophysical
plasma simulations [@vaisala2023exploring], which scales to thousands of GPUs with a weak scaling
efficiency \>90% [@pekkila_graphicsprocessors_2026, and has also been used to accelerate the widely used astrophysics
framework `Pencil Code`.

Of course, `Astaroth`'s PDE solver is not limited to astrophysics, and neither
is `Astaroth` limited to PDEs.
As an example, many image processing techniques, like edge detection and
convolutions, are stencil operations and stencil operations are widely used in numerical linear algebra.


# State of the field

Several approaches have been proposed to improve the performance, portability, and productivity dimensions of stencil computations.
Single-process approaches include domain-specific languages (DSLs), e.g., `Halide` [@ragan2013halide], `PolyMage` [@mullapudi2015polymage], `Delite` [@sujeeth_delitecompiler_2014], and `Lift` [@steuwer_liftfunctional_2017], and general parallel processing abstractions, e.g., `Kokkos` [@trott2021kokkos] and `RAJA` [@beckingsale2019raja].
The primary benefit of a DSL is that assumptions about the structure of computations can be made to improve the performance of generated code while maintaining a high-level representation for the user.
Performance-portability is typically augmented with automated tuning and algorithm selection (e.g., `PATUS` [@christen_patuscode_2011], `PARTANS` [@lutz_partansautotuning_2013]).
These approaches are also adopted in `Astaroth`, which introduces a DSL, a code generator, and implements automated tile-size optimization.
A distinctive feature of Astaroth is its algorithmic specialization for cache-constrained use cases in multiphysics, where the working set required to update interdependent physical fields is too large to fit into on-chip caches.
This is addressed by rearranging the computations into linear and nonlinear stages on-chip [@pekkila2025stencil].
`Astaroth` not only consider stencils in isolation, but also their combinations with other operations inside the same kernel, such as the automatic fusion of distributed reductions with other computations.

Methods to improve the performance-portability-productivity dimensions in multiprocess applications include topology-aware workload distribution and abstractions for distributed operations.
`Chapel` [@callahan_cascadehigh_2004], `Charm++` [@kale_charmportable_1993] improve the performance-portability-productivity dimensions by providing general-purpose distributed programming models, whereas `Cactus` [@goodale_cactusframework_2003], `PETSc` [@mills_towardperformance_2021] implement scientific computation toolkits.
Further productivity advances can be gained with domain-specialized frameworks, which provide ready-made solutions for specific use cases and a simplified application programming interface.
`Astaroth` also belongs to this class of frameworks.
Examples closest to `Astaroth` are `Parthenon` [@grete_parthenonperformance_2023] and `AMReX` [@zhang_amrexframework_2019].
Both are frameworks for distributed adaptive mesh refinement (AMR), where `Parthenon` uses `Kokkos` as the compute backend and `AMReX` provides compute features with parallel function wrappers and user-written C++ lambdas.
In contrast to these projects, `Astaroth`'s distributed abstraction layer focuses on structured grid computations without mesh refinement, and can thus make simplifying assumptions about the underlying data-movement patterns to better address on-chip data movement, batching, and data-processing pipelines.
Similar to `Parthenon` and others [@pearson_movementplacement_2021], `Astaroth` implements its modification of topology-aware domain decomposition and rank reordering for improved portability across systems, and performs fused packing to alleviate communication overheads.
Furthermore, Astaroth implements a task scheduler for compute and communication tasks [@lappi2021task].

In the field, `Astaroth` stands out as a CUDA/HIP stencil-computing framework focused on addressing the performance-productivity trade-off in cache-heavy multiphysics applications with a domain-specific language, automated tile-size optimization, and topology handling, taking ownership of data structures and movement throughout the computational science pipeline.
This enables holistic optimizations of full scientific workflows and offers a lower barrier of entry to experimentation with optimization techniques spanning traditionally decoupled tasks (e.g., extensive kernel fusion of operations across the stack), which would not be practical with libraries utilizing opaque submodules for compute and communication.


# Software design

`Astaroth` consists of three main components: 1) `acc`, a compiler and runtime system for a domain-specific language (DSL) for stencil computations, 2) an API for executing stencil applications on multi-GPU platforms, and 3) a standalone PDE solver.
Below, we present a quick overview of these components. Extensive documentation is available at [@astaroth_doc].

## `acc` compiler and runtime system

`Astaroth` has a DSL for stencil-based computations, designed to be used by domain scientists without having to consider technical implementation details.
The main operations, such as stencils, are written in a declarative syntax, and the kernels that use them are written in an imperative syntax.[^paradigm_footnote]
The implementation is left to `Astaroth`'s DSL compiler `acc`, which applies a number of specialized optimizations.
An especially important optimization is the unrolled and reordered computation of all required stencils at the start of the kernels, which enables instruction-level parallelism and efficient usage of caches [@pekkila_graphicsprocessors_2026].
In addition to stencils, the DSL supports two other operations: 1) multi-GPU reductions -- which are commonly needed for stencil-based solvers; and 2) simplified distributed ray-tracing, where rays cannot change directions and are restricted to move through neighbouring grid points -- which is necessary for simulations incorporating radiative transfer [@heinemann2006radiative].
`Astaroth`'s DSL also includes a standard library, providing, inter alia: derivative operators used in PDE solvers, implemented for generally spaced Cartesian, spherical or cylindrical grids; and Poisson solvers, e.g. for self-gravity [@krasnopolsky2026iterative].

`acc` transpiles the DSL source into CUDA or HIP source code, which is further compiled into machine code using a native CUDA or HIP compiler.
The program thus produced is executed in the `acc` runtime system, which further optimizes the kernels by autotuning the thread block sizes for kernel execution.
`acc` also supports run-time compilation, because run-time configuration parameters may change the evaluation of conditional statements, thereby changing the branches taken at run-time.
With run-time compilation, `acc` compiles for a given configuration only those parts of the DSL source that will be executed.
The information of what code gets executed also allows `Astaroth` to optimize run-time behaviour more precisely, e.g. memory allocations or communication patterns.

## Multi-GPU runtime system and API

In the DSL, using the `ComputeSteps` language construct, users can define a list of compute steps specifying a sequence of kernels and boundary conditions.
Kernels defined in `ComputeSteps` may be fused to reduce memory reads.
Based on the overall domain decomposition and the stencils' data access patterns, `acc` infers the dependency relationships between the steps, and constructs a directed acyclic graph (DAG) of dependent tasks.
Each step is split into many tasks, one per region: there are 26 smaller regions at each subdomain's boundaries, which are dependent on communicated data from neighbors; and one big region at the core of the subdomain, which is not.
Communication tasks are inserted where needed.

`Astaroth`'s task scheduler executes these DAGs, asynchronously launching computation and communication tasks when prerequisite tasks are completed.
This improves performance in communication-bound cases, especially for higher process counts [@lappi2021task].
For fast data transfers and to support all possible hardware, both GPU-to-GPU remote direct memory access (RDMA) and CPU-to-CPU communication are supported.

This runtime system can be accessed through `Astaroth`'s runtime API.
The API is C-ABI compatible, supporting foreign function interfaces to external applications written in any programming language.
The API is organized into two layers: the `Device` layer and the `Grid` layer.
The `Device` layer provides access to single-GPU functionality, such as: moving data between CPU and GPU, launching kernels, and loading/storing snapshots from/to disk.
The `Grid` layer provides access to multi-GPU functionality, such as: executing DAGs, distributed initialization, and distributed loading/storing of snapshots.
Other special functionality is also provided through the API, such as distributed Fourier transforms.

## Solver

`Astaroth` also includes a standalone finite-difference PDE solver [@pekkila2022scalable], which takes full advantage of the DSL and the multi-GPU API, and can be used to write new simulation models.
It also works as a testbed for performance research.
This solver uses an astrophysical magnetohydrodynamical setup (`acc-runtime/samples/mhd_modular`) by default, but can be configured to run any DSL code.
The samples directory also includes other production-ready setups, e.g. `tfm-mpi` for the test-field method [@pekkila_graphicsprocessors_2026].

The solver handles distributed initial conditions, domain decomposition, simulation diagnostics, and logging.
It is also designed to react to a number of events, such as NaNs in the simulation data, simulation time limits, and a stop signal given through the file system.
The directory `analysis/` contains Python-based data analysis tools, which can be used to process and work with the data produced by the standalone solver.


# Research impact statement

`Astaroth` has already been used in a number of papers as the core PDE-solver, mainly for astrophysical plasma simulations [@vaisala2021interaction; @vaisala2023exploring; @gent2026asymptotic], but also in seismology [@ladino2025acoustic].
Additionally, it has been used for research on performance optimization methods[@pekkila_graphicsprocessors_2026;@pekkila2025stencil;@pekkila2017methods], communication techniques [@pekkila2022scalable;@lappi2021task], compiler techniques [@pekkila_masters_2019;@puro2023programmatic] and other topics [@yokelson2024soma; @puro2025gpu].
We expect that the recent GPU-acceleration of `Pencil Code`, which was done by embedding `Astaroth`'s DSL and runtime system into it, will increase the number of `Astaroth` users.
The associated speedup factor of 20-60 [@pekkila2022scalable] will enable more realistic astrophysical simulations in a wide range of use cases from modelling small-scale dynamos [@warnecke2025small] to processes producing primordial gravitational waves and their propagation [@roper2020numerical].


# Acknowledgements

We acknowledge the contributions of all developers and early users of `Astaroth` who have been instrumental in its evolution. These include Petr Bém, Jörn Warnecke, Frederick Gent, Ruben Krasnopolsky, Wei-Wen Li, Mordecai Mac Low, Chun-Fan Liu, Man Hei Li, Tzu-Chun Hsu and Indrani Das.
We acknowledge the computational resources and services provided by CSC — IT Center for Science, the Aalto Science-IT project, ASIAA High-Performance Computing, and National Center for High-Performance Computing (NCHC), National Applied Research Laboratories (NARLabs) in Taiwan, the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, and resources from LUMI-G through the Euro-HPC joint undertaking. Furthermore, we appreciate the important technical assistance provided by CSC, by people like Fredrik Robertsén and others.
The development of `Astaroth`  has received funding from the Academy of Finland, ReSoLVE Centre of Excellence, Grant/Award Number: 307411;
The European Research Council, the European Union's Horizon 2020 research and innovation program, project UniSDyn, Grant/Award Number: 818665; KAUTE Foundation, Grant/Award Numbers: 20240173 and 20250154; Research Council of Finland, project MomEnt, Grant/Award Number: 373416.
The authors acknowledge support for the CompAS Project from the Institute of Astronomy and Astrophysics, Academia Sinica (ASIAA), the Academia Sinica grant AS-IAIA-114-M01, and the National Science and Technology Council (NSTC) in Taiwan through grants 112-2112-M-001-030, 113-2112-M-001-008, and 114-2112-M-001-001-; the International Collaboration and Cooperation grant for COSMAGG that supports the exchanges between Taiwan and Finland: 113-2927-I-001-513-, 114-2927-I-001-506-, and Research Council of Finland project 359462.
MSV thanks the support of Jenny and Antti Wihuri Foundation and Finnish Cultural Foundation in his doctoral thesis work, and therefore the early development in Astaroth prototype.

# AI usage disclosure

AI tools have not been used in any step of software creation, documentation or in the authoring of this paper.

# References

[^stencil_footnote]: Stencil computations are computations on structured grids where a given point is updated using a fixed neighborhood pattern. Examples are convolutions in image processing and convolutional neural networks, and different schemes for spatial derivatives like the finite-difference method.
[^paradigm_footnote]: In declarative programming, computations are defined by describing what the results look like; in imperative programming, by describing the steps to perform.
