# Installation

MAM4xx builds and runs on the following platforms:

* Mac and Linux laptops and workstations
* NERSC Cori
* Compy and Constance at PNNL

## Required Software

To build MAM4xx, you need:

* [CMake v3.17+](https://cmake.org/)
* GNU Make
* reliable C and C++ compilers
* optionally, a working MPI installation (like [OpenMPI](https://www.open-mpi.org/)
  or [Mpich](https://www.mpich.org/)), if you want to use MAM4xx in a
  multi-node parallel environment
* the [HAERO](https://github.com/eagles-project/haero) aerosol package interface,
  which provides necessary libraries and settings.

If you want to compare MAM4xx's processes and parameterizations with the
[MAM4 microphysics box model](https://github.com/eagles-project/mam_refactor),
you also need decent Fortran compiler, such as GNU `gfortran` or Intel's `ifort`
compiler.

You can obtain all of these (except perhaps your favorite Fortran compiler)
freely on the Linux and Mac platforms. On Linux, just use your favorite package
manager. On a Mac, you can get the Clang C/C++ compiler by installing XCode, and
then use a package manager like [Homebrew](https://brew.sh/) or
[MacPorts](https://www.macports.org/) to get the rest.

For example, to download the relevant software on your Mac using Homebrew, type

```
brew install cmake gfortran openmpi
```

## Build and Install HAERO

MAM4xx uses data structures and libraries provided by Haero. To build Haero, you
can either

* use the [build-haero.sh script](https://github.com/eagles-project/mam4xx/blob/main/build-haero.sh), or
* follow the directions in the [Haero repository](https://github.com/eagles-project/haero) itself

## Clone the MAM4xx Repository

First, go get the [source code](https://github.com/eagles-project/mam4xx)
at GitHub:

=== "SSH"
    ```
    git clone git@github.com:eagles-project/mam4xx.git
    ```
=== "HTTPS"
    ```
    git clone https://github.com/eagles-project/mam4xx.git
    ```

This places a `mam4xx` folder into your current path.

## Configure MAM4xx

MAM4xx uses CMake, and accepts a number of options that specify how it should be
built. In order to simplify the build process, we've provided a simple `setup`
script that generates a shell script you can run to invoke CMake with the
appropriate options set.

To configure MAM4xx:

1. Create a build directory by running the `setup` script from the top-level
   source directory:
   ```
   ./setup build
   ```
2. Change to your build directory and edit the `config.sh` file to select
   configuration options. Then run `./config.sh` to configure the model.

If you prefer, you can fish the options out of the `setup` script (or your
generated `config.sh` file) and feed them directly to CMake.

## Build, Test, and Install MAM4xx

After you've configured MAM4xx, you can build it:

1. From the build directory, type `make -j` to build the library. (If you've
   configured your build for a GPU, place a number after the `-j` flag, as in
   `make -j 8`).
4. To run tests for the library (and the driver, if configured), type
   `make test`.
5. To install the model to the location indicated by `PREFIX` in your
   `config.sh` script (or `CMAKE_INSTALL_PREFIX`, if you specified it manually),
   type `make install`. By default, products are installed in `include`, `lib`,
   `bin`, and `share` subdirectories within your build directory.

## Making code changes and rebuilding

Notice that you must build MAM4xx in a  **build tree**, separate from its source
trees. This is standard practice in CMake-based build systems, and it allows you
to build several different configurations without leaving generated and compiled
files all over your source directory. However, you might have to change the way
you work in order to be productive in this kind of environment.

When you make a code change, make sure you build from the build directory that
you created in step 1 above:

```
cd /path/to/mam4xx/build
make -j
```

You can also run tests from this build directory with `make test`.

This is very different from how some people like to work. One method of making
this easier is to use an editor in a dedicated window, and have another window
open with a terminal, sitting in your `build` directory. If you're using a fancy
modern editor, it might have a CMake-based workflow that handles all of this for
you.

The build directory has a structure that mirrors the source directory, and you
can type `make` in any one of its subdirectories to do partial builds. In
practice, though, it's safest to always build from the top of the build tree.

