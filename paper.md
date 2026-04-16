---
title: 'Astaroth: A scientific computing framework for accelerating stencil codes.'
tags:
  - C/C++
  - scientific computing
  - high performance computing
  - astrophysics

authors:
  - name: Johannes Pekkilä
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1 
  - name: Touko Puro
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    corresponding: true
    affiliation: 1 
  - name: Miikka Väisälä
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 2
  - name: Oskar Lappi 
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 3
  - name: Maarit Korpi-Lagg 
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Matthias Rheinhardt
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1

affiliations:
 - name:  Aalto University, Finland 
   index: 1
 - name:  University of Oulu, Finland 
   index: 2
 - name:  University of Helsinki, Finland 
   index: 3
 
date: 15 April 2026
bibliography: paper.bib

---

# Summary

Stencil computations are one the bedrocks of high performance scientific simulations,
with them being the most important primitive for a large number of PDE and numerical linear algebra solvers. 
At the same time GPU computing is an ever increasing part of high performance computing given its speed and energy efficiency. 
Thus GPU acceleration of stencil computations, and of the codes using them, is essential in the pursuit of increasingly larger simulations, which are needed in ubiquitous scientific fields to push scientific knowledge forward.

# Statement of need

`Astaroth` is a GPU framework primarily for scientific computing and for acceleration of scientific software where the most important primitives are stencil operations.
Astaroth provides its own domain specific language (DSL) and a runtime for it that allows
one to easily express their required computations and leaving the gritty details to the compiler and runtime.
While stencils are the core of Astaroth, it also provides other important primitives like reductions (e.g. sums), simple ray-tracing and has integrations for GPU accelerated Fourier transforms all of which are important for simulations on structured grids.

`Astaroth`'s main use has been accelerating PDE solvers based on high-order Finite Differences. Thus it cames together with its own stand-alone PDE solver and a standard library that has a large spectrum of different operators implemented for PDE simulations, making it easy to write new simulations with it and port existing codes.

The performance increase enabled by `Astaroth` compared to previous generation CPU based solvers and frameworks enables simulations of increased resolution, pushing the achievable numerical parameters closer to those found in nature.

# State of the field                                                                                                                  

Would Johannes be the best to write something about other DSL etc. approaches for GPU stencil computations?
I mean if he would have the time to slap something quickly here? I can polish whatever he has time to quickly sketch.

Compared to the state of the field Astaroth excels in the robustness of the integration of its domain specific language to its runtime and with its integration to existing codes (make the meaning of this more precise).

# Software design

TODO: Based on Oskar's comments larger emphasis should be made here about the integration of Astaroth to existing codes.
At the same time the granularity of the multi-layer approach could be cut down.


`Astaroth`'s core design flows from the DSL and stencil computations and is multi-layered.
A core design principle is that the whole library, including its runtime, is compiled based on the DSL written by the user.
This allows `Astaroth` to make the usage experience as smooth as possible and at the same time optimize itself for the specific use case at hand. This principle is taken to the logical conclusion in the feature of runtime-compilation where the whole library is compiled from scratch at the runtime of the application after all possible information about the application are known. This is particularly important for large multi-physics PDE solvers which contain many control variables describing the simulation setup at hand.

Depending on the users wants and needs `Astaroth` exposes multiple different layers from which the user can choose from.
Importantly the layer of highest abstraction (the so called `Grid`-layer) is as descriptive as possible.
In the DSL the user specifies what computation they want to perform on a structured grid by using stencils and the `Grid`-layer executes user's intent in as a streamlined way as possible.
One example of this are `ComputeSteps`; The user describes steps of computations (kernel invocations) and `Astaroth` handles all execution details of performing the computations such as the required computation for multi-GPU stencil computations and reductions.
This declarative nature allows `Astaroth` to automatically to handle the execution details and leaves enough freedom for `Astaroth` to choose the most performant way to perform the computations.
Alternatively for use cases where the user wants granular control of every specific detail `Astaroth` exposes the `Device`-layer from which the user can configure details down to hardware settings.

The external C-based API of `Astaroth` is kept as simple as possible allowing easy integration of it to existing codes.
An important design choice with respect to this its global configuration structure which holds all of the important global state of `Astaroth`, which enables one to easily to transfer data from existing codes to `Astaroth`.


# Research impact statement

`Astaroth` has already been used in many papers as the core PDE solver, mainly in astrophysical settings, and in performance focused papers
as a platform to study optimization techniques. Recently the acceleration of `Pencil Code` via `Astaroth` is expected to 
increase the number of people relying on `Astaroth` as the core execution engine and is expected to enable more realistic astrophysical simulations in a wide range of scientific applications from modelling small-scale dynamos to the propagation and processes producing primordial gravitational waves.


List of papers: (if something is missing please add)

Stencil computations on amd and nvidia graphics processors: Performance and tuning strategies
Interaction of large-and small-scale dynamos in isotropic turbulent flows from GPU-accelerated simulations
Scalable communication for high-order stencil computations using CUDA-aware MPI
Exploring the formation of resistive pseudodisks with the GPU code astaroth
Methods for compressible fluid simulation on GPUs using high-order finite differences
Observability, monitoring, and in situ analytics in exascale applications (used Astaroth standalone)
Asymptotic behaviour of galactic small-scale dynamos at modest magnetic Prandtl number
SOMA: Observability, monitoring, and in situ analytics for exascale applications
GPU acceleration of average gradient method for solving partial differential equations



# Acknowledgements

We acknowledge the contributions of every committer to Astaroth and the early users of it who have been instrumental 
in its evolution. (Jörn Warnecke, Frederick Gent, Indrani Das, Ruben Krasnopolsky, Hsien Shang, who else)?

# References
