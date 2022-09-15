# Developing MAM4xx

Welcome to the MAM4xx development team! MAM4xx is a function-by-function
port of MAM4 (the 4-mode Modal Aerosol Model package) from Fortran to
"performance portable" C++ code that can run on GPU accelerators as well as
traditional CPUs.

This C++ port of MAM4 provides a much-needed prognostic aerosol capability to
the [Energy Exascale Earth System Model (E3SM)](https://e3sm.org), which is
designed to run on the Department of Energy's
[leadership-class computing facilities](https://www.doeleadershipcomputing.org).

## Getting Help

You're part of a team. That means you shouldn't have to sit alone with a problem
and scratch your head until the solution magically appears inside it. If you're
stuck, organize your thoughts and ask another member of the team for their
input. Not only can this save you time, but it can also help you build
productive relationships with other team members.

If you're a member of the MAM4xx development team on the EAGLES Project, the
easiest way to seek help is to post a Slack message to the appropriate channel.
We have access to an `ESMD-BER` Slack workspace with some useful channels for
asynchronous team communication:

* `eagles-mam-cpp`: This is the best channel for discussing and troubleshooting
  issues related to MAM4xx development.
* `eagles_haero`: This channel is for discussing the HAERO aerosol package
  "toolbox" used by MAM4xx.
* `eagles_mamrefactor`: In this channel, you can ask questions about the
  [MAM4 box model](https://github.com/eagles-project/mam_refactor) used for
  porting MAM4's aerosol microphysics parameterizations.

If you're not already on the `ESMD-BER` Slack workspace, please ask a team
member to send you an invitation.

If you like, you can also create issues in the [MAM4xx repository](https://github.com/eagles-project/mam4xx)
itself.

## The Big Picture

MAM4xx makes the 4-mode modal aerosol model ("MAM4") available to
[EAMxx (aka "SCREAM")](https://github.com/E3SM-Project/scream), which is written
in C++ and uses [Kokkos](https://github.com/kokkos/kokkos) to achieve good
parallel performance on platforms of interest to the DOE's Office of Science.
EAMxx has a lot of moving parts, but MAM4xx mainly interacts only with a data
structure called the Atmosphere Driver (AD).

### Terminology

* A **host model** is an atmospheric model containing a dynamical core that
  solves a transport equation for mass in the atmosphere, and several physics
  packages that parameterize important atmospheric processes that can't be
  resolved by the underlying grid. EAMxx is the atmospheric host model used
  by E3SM version 4, so when we refer to the "host model", we refer to EAMxx.
* An **aerosol package** is a physics package that provides a representation of
  aerosols (prognostic and diagnostic variables, and tendencies for evolving
  the prognostics) for use by a host model. The aerosol package of interest for
  us, of course, is MAM4xx.
* An **aerosol process** is a set of functions associated with a specific part
  of the aerosol lifecycle (e.g. nucleation, coagulation, aging) that calculate
  updates to aerosol-related quantities.
* An **aerosol parameterization** is a function that computes one or more
  quantities needed to update aerosol-related quantities. An aerosol process can
  contain one or more related parameterizations.
* A **prognostic variable** is a quantity in the atmosphere whose evolution is
  described by a differential equation. Prognostic variables cannot be obtained
  using closed-form (algebraic) equations.
* A **diagnostic variable** is a quantity in the atmosphere that can be
  expressed in terms of prognostic variables in closed form (usually some
  algebraic expression).
* A **tendency** is a time derivative ("rate of change") associated with a
  prognostic variable. An aerosol process computes tendencies given a set of
  prognostic variables.
* An **atmospheric state** is a complete quantitative description of the
  atmosphere according to a host model. This description consists entirely of
  prognostic and diagnostic variables.

### EAMxx's atmosphere driver

In essence, an atmospheric host model does the following things:
1. it initializes the state of the atmosphere at the beginning of the simulation
2. it advances the state of the atmosphere and the simulation time in a sequence
   of discrete "time steps". Within each time step of length `dt`, tendencies
   are computed for each of the prognostic variables, and the atmospheric state
   is updated from time `t` to time `t + dt` by integrating these tendencies

### EAMxx's atmosphere processes
### HAERO and aerosol processes

## MAM4xx Code Structure

### Aerosol processes and parameterizations
### Aerosol process structure

## C++ Guidelines

C++ is a large, multi-paradigm programming language. The way C++ is used has
changed so many times over the years that it's crucial for us to decide how
much of the language we use, and how we'll use it.

Much of this section is up for debate/discussion, but here are some guiding
principles that are unlikely to change:

1. **Favor clarity over cleverness.** Anyone can write code that no one else
   can understand. Writing simple code that is intelligible to people of various
   skill levels is challenging, but worth the investment in time and effort.
   _"Don't be clever."_ -Bjarne Stroustrup
2. **Avoid frivolous use of C++ features.** The language is huge, and our job is
   not to maximize our use of it, but to use it effectively.
   _"Every new powerful feature will be overused and misused."_
   -Bjarne Stroustrup
3. **Remember your audience.** We're writing science codes, so what we write
   must be intelligible to scientists and others without formal training in
   software engineering. Don't use `int x{};` when `int x = 0;` does the same
   thing with more clarity.
   _"Only half of the C++ community is above average."_ -Bjarne Stroustrup

The astute reader may notice a certain redundancy or even repetition in these
principles. It is left as an exercise to ponder why that might be so.

### Style

We adhere somewhat loosely to [LLVM's C++ Style Guide](https://llvm.org/docs/CodingStandards.html),
with a few notable _exceptions_:

* We allow the use of C++ exceptions, since simulation codes have rather
  simplistic error handling requirements.
* Names of functions, methods, and variables use `snake_case`, not
  `camelCase` or `UpperCamelCase`.
* We typically use braces to enclose logic for all `if`/`else`/loop statements
  even if they are only a single line, for consistency and readability.
* We use `EKAT_ASSERT` instead of `assert` to ensure that all MPI processes
  are properly shut down when a program terminates because of a violated
  assertion.

### Best practices

The bullets in the [LLVM C++ Style Guide](https://llvm.org/docs/CodingStandards.html)
provide good guidelines for best practices. Here are some additional
tips/opinions:

* **Avoid inheritance where possible**: C++'s model of inheritance is easy to
  use but often hard to understand. When possible organize things so that
  objects belong to other objects instead of inheriting from them. Sometimes
  this approach is articulated as [**composition over inheritance**](https://en.wikipedia.org/wiki/Composition_over_inheritance).
  In particular, when you are tempted to use inheritance to bestow the
  capabilities of one type upon another, pause for a moment to think about
  whether there's a better way to accomplish what you're trying to do.
* **Make effective use of the standard library, but don't overdo it**: Sometimes
  the clearest expression of an algorithm uses a `for` loop and not a devilishly
  clever concoction of esoteric STL types, traits, and functional wizardry.
  _This is particularly true when writing code that runs on a GPU, for which the
  standard template library is largely unavailable!_

## Kokkos, EKAT, Haero: Intra-node Parallelism

MAM4xx is written in "performance-portable" C++ code using [Kokkos](https://kokkos.github.io/kokkos-core-wiki/)
to dispatch parallelizable workloads to threads on CPUs or GPUs on a compute
node. Kokkos allows developers to write code that is very nearly standard C++
that can run on GPU accelerators, which makes it unnecessary to learn
specialized accelerator languages like CUDA and HIP.

Because MAM4xx is based on column physics, it operates on sets of independent
vertical atmospheric columns and can do all of its work within a single compute
node. In other words, a MAM4xx instance on a compute node has no specific need
to communicate with other nodes. However, the host model that uses MAM4xx almost
certainly needs inter-nodal communication, for which [MPI](https://www.mpi-forum.org)
is used.

The high-performance data types in MAM4xx used for these parallel dispatches are
all provided by Kokkos. Kokkos is a general-purpose parallel programming model,
and is accordingly complex, with many elaborate features and options. In order
to reduce this complexity and focus on decisions and logic related to earth
system models (ESMs) in general and aerosols in particular, we make use of a
couple of additional layers:

* [**E3SM/Kokkos Application Toolkit (EKAT)**](https://github.com/E3SM-Project/EKAT):
  A library that defines specific Kokkos-based data structures relevant to
  E3SM-related projects, and some useful bundled external libraries:
    * `yamlcpp`: a C++ YAML parser for handling configuration files
    * `spdlog`: a fancy C++ logging system that provides multiple loggers and
                extensible logging levels
    * `fmt`: a fancy C++ formatting system that provides Python-like formatting
             for C++ strings
* [**High performance AEROsol interface (HAERO)**](https://github.com/eagles-project/haero):
  A library that defines data types for aerosol packages. HAERO relies heavily
  upon EKAT, but makes a lot of choices appropriate for aerosol column physics
  so we can focus on solving relevant problems and not reinventing the wheel
  over and over.

EKAT includes an MPI configuration in its build system, and this configuration
is passed along to HAERO and MAM4xx. This means you'll need some implementation
of MPI on your system, like [OpenMPI](https://www.open-mpi.org) or
[MPICH](https://www.mpich.org).

In this section, we describe the data structures provided by HAERO (via EKAT
and Kokkos). The [Kokkos documentation](https://kokkos.github.io/kokkos-core-wiki/)
and [tutorials](https://github.com/kokkos/kokkos-tutorials) are fantastic
resources for understanding the most important data structures and
techniques we use.

### Views: C++ multidimensional arrays

Fortran programmers have long been skeptical about using C++ as a scientific
programming language because C++ doesn't have multidimensional arrays. (This has
also frustrated a lot of C++ programmers in the HPC community!)

Kokkos provides a solution to this problem: the [View](https://kokkos.github.io/kokkos-core-wiki/API/core/view/view.html)
data structure. A `View` is basically a multidimensional array that lives in a
specific memory location (either on a CPU or a GPU). The `View` type has several
template parameters that dictate what it stores, where it stores things, and
how it indexes them.

As a multidimensional array, a `View` has a **rank** that indicates the number
of indices it possesses. For example, a rank-1 `View` `V` has a single index,
allowing you to retrieve the `i`th value with the syntax `V(i)`. A rank-3 `View`
`T` has three indexes, providing access to an element with the syntax
`T(i, j, k)`.

Some people refer to the rank of a View as its **dimension**, but this term
actually refers to the number of elements for a specific index. For example, the
dimension of the second index of `T` above is the valid number of values of `j`
that can be used in the expression `T(i, j, k)`. Indices in a `View` run from
`0` to `dim-1`, where `dim` is the dimension relative to the index in question.
The **shape** of a `View` is the set of dimensions of its indices. For example,
the rank-3 `View` `T` may have a shape of `(100, 100, 100)`.

The `View` type is very flexible, so it can be complicated to work with
directly. HAERO provides a few useful types that nail down the various
parameters according to the needs of aerosol column physics:

* `ColumnView`: a rank-1 `View` whose index (typically written `k`) identifies
  a specific vertical level in a column of "air" in the atmosphere. This type
  of `View` is used to represent all quantities of interest in an aerosol
  parameterization.
* `TracersView`: a rank-3 `View` with indices `n`, `i`, `k`, that identify a
  specific tracer (advected quantity) `n` in a specific column `i` at a specific
  vertical level `k`. This `View` type is used to extract prognostic aerosol
  data from an atmospheric host model (e.g. EAMxx) so it can be advanced by
  MAM4xx.
* `DiagnosticsView`: a rank-3 `View` similar to `TracersView`, used to index
  diagnostic aerosol data from an atmosphereic host model for use and updating
  by MAM4xx.

These three `View` types should be all you need to implement aerosol processes
and their parameterizations. In fact, the aerosol processes themselves really
only use the `ColumnView` type.

### Parallel dispatch: host and device

MAM4xx runs within a single process runs on an entire compute node, no matter
how many CPUs or GPUs are available to that node. Within MAM4xx, processes on
different compute nodes typically don't communicate directly with each other.
Instead, the host model coordinates communication between these processes using
MPI, and MAM4xx relies on the host model to get consistent data.

To understand the _intranodal_ parallelism used by MAM4xx, we need some
terminology:

* The **compute host** is the CPU running the process containing the atmospheric
  host model and MAM4xx. The compute host manages the control flow of the host
  nmodel and MAM4xx--it can also do numerical calculations, but such
  calculations can't be done in parallel on the host.
* The **compute device** is where numerical calculations are performed in
  parallel. On a node with only CPUs, the role of the compute device is played
  by the same CPU as that for the compute host. On a node with access to GPUs,
  the compute device is the GPU, which has its own memory and (very different!)
  processing hardware. Logically, the compute device is distinct from the
  compute host, because only the compute device can execute code in parallel.

Strictly speaking, a machine can have more than one compute device. For example,
a many-core CPU with a GPU has two potential compute devices: the CPU and the
GPU. We will ignore this possibility and assume that all calculations are done
on a single compute device, which we call the **device**.

To make use of the **device** on a compute node, MAM4xx uses the
[parallel dispatch](https://kokkos.github.io/kokkos-core-wiki/API/core/ParallelDispatch.html)
capabilities provided by Kokkos. MAM4xx's "column physics" approach allows it
to take advantage of a specific parallel dispatch approach based on the Kokkos
[TeamPolicy](https://kokkos.github.io/kokkos-core-wiki/API/core/policies/TeamPolicy.html).
Here's how it works.

_Describe hierarchical parallel dispatch here_

### Frequently Asked Questions

## Packs and Vectorization

### Why Packs?

### Masks and predicates

### Frequently Asked Questions

## Resources

* [EKAT repository](https://github.com/E3SM-Project/EKAT)
* [E3SM website](https://e3sm.org)
* [Kokkos documentation](https://kokkos.github.io/kokkos-core-wiki/)
* [HAERO repository](https://github.com/eagles-project/haero)
* [LLVM C++ Style Guide](https://llvm.org/docs/CodingStandards.html)
* [MAM4 box model repository](https://github.com/eagles-project/mam_refactor)
* [SCREAM (EAMxx) repository](https://github.com/E3SM-Project/scream)
* [Skywalker documentation](https://eagles-project.github.io/skywalker/)
