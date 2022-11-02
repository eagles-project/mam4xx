# MAM4xx

This repository contains the source code for a performance-portable C++
implementation of the MAM4 modal aerosol model (with 4 fixed modes).

## Required Software

To build MAM4xx, you need:

* [CMake v3.12+](https://cmake.org/)
* GNU Make
* reliable C and C++ compilers
* optionally, a working MPI installation (like [OpenMPI](https://www.open-mpi.org/)
  or [Mpich](https://www.mpich.org/)), if you want to use MAM4xx in a
  multi-node parallel environment
* the [HAERO](https://github.com/eagles-project/haero) aerosol package interface,
  which provides necessary libraries and settings.

You can obtain all of these freely on the Linux and Mac platforms. On Linux,
just use your favorite package manager. On a Mac, you can get the Clang C/C++
compiler by installing XCode, and then use a package manager like
[Homebrew](https://brew.sh/) or [MacPorts](https://www.macports.org/) to get the
rest.

For example, to download the relevant software on your Mac using Homebrew, type

```
brew install cmake openmpi
```

## Building MAM4xx

### Installing HAERO

Before you get started with MAM4xx, you'll need a working installation of the
HAERO high-performance aerosol interface library. For your convenience, we have
provided the `build-haero.sh` script, which can be used to quickly and easily
install HAERO in a desired configuration. The script allows you to set a number
of parameters. Check out the comments at the top of `build-haero.sh`.

You can build a CPU-capable version of HAERO with some defaults set by typing

```
./build-haero.sh <path>
```

where `<path>` is a directory to which HAERO will be installed. If you'd rather
install HAERO yourself, you can follow the instructions in the
[HAERO repository](https://github.com/eagles-project/haero). Make sure you run
all the steps, including `make install`.

### Initializing submodules

Before you start working with the repo, make sure you initialize its submodules:

```
git submodule update --init --recursive
```

### Configuring and Building MAM4xx

To build MAM4xx:

1. Create a build directory by running the `setup` script from the top-level
   source directory:
   ```
   ./setup build
   ```
2. Change to your build directory and edit the `config.sh` file to select
   configuration options. Then run `./config.sh` to configure the model. MAM4xx
   gets most of its configuration information from HAERO, so there aren't many
   options here.
3. From the build directory, type `make -j` to build the library. (If you're
   building MAM4xx for GPUs, place a number after the `-j` flag, as in
   `make -j 8`).
4. To run tests for the library (and the driver, if configured), type
   `make test`.
5. To install the model to the location indicated by `PREFIX` in your
   `config.sh` script, type `make install`. By default, products are installed
   in `include`, `lib`, `bin`, and `share` ѕubdirectories within your build
   directory.

### Making code changes and rebuilding

This project uses **build trees** that are separate from source trees. This
is standard practice in CMake-based build systems, and it allows you to build
several different configurations without leaving generated and compiled files
all over your source directory. However, you might have to change the way you
work in order to be productive in this kind of environment.

When you make a code change, make sure you build from the build directory that
you created in step 1 above:

```
cd /path/to/mam4xx/build
make -j
```

You can also run tests from this build directory with `make test`.

This is very different from how some people like to work. One method of making
this easier is to use an editor in a dedicated window, and have another window
open with a terminal, sitting in your `build` directory.

The build directory has a structure that mirrors the source directory, and you
can type `make` in any one of its subdirectories to do partial builds. In
practice, though, it's safest to always build from the top of the build tree.

### Checking C++ formatting, and auto-formatting

Our C++ style rules are described in the [MAM4xx developer guide](https://github.com/eagles-project/mam4xx/blob/main/docs/development.md).
We enforce them using `clang-format`. If you have the correct version of
`clang-format` installed, you can use the following targets to check and fix
all C++ code in the `src` subdirectory:

* `make format-cxx-check`: checks C++ formatting in all source files and reports
  any non-conforming code
* `make format-cxx`: applies C++ formatting rules to all source files, editing
  them in place. Try to do this in a separate commit from your other work.

You can run either of these targets from your build directory. If you have a
different version of `clang-format` than the one we support, you'll get an error
message telling you the correct version to install when you use either of these
targets.

## Generating Documentation

Documentation for MAM4xx can be built using
[`mkdocs`](https://squidfunk.github.io/mkdocs-material/).
In order to build and view the
documentation, you must download `mkdocs` and its Material theme:

```pip3 install mkdocs mkdocs-material```

Then, run `mkdocs serve` from the root directory of your MAM4xx repo,
and point your browser to [`http://localhost:8000`](http://localhost:8000).

You can also view the [MAM4xx developer guide](https://github.com/eagles-project/mam4xx/blob/main/docs/development.md)
on GitHub.
