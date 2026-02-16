#!/usr/bin/env bash

# This script builds the Haero high-performance aerosool toolkit with specific
# settings so mam4xx can be linked against it. Run it like this:
#
# `./build-haero.sh <prefix> <device> <precision> <build_type> [gpu_type] [gpu_arch]
#
# where
# * <prefix> is the installation prefix (e.g. /usr/local) in which you
#   would like Haero installed
# * <device> (either `cpu` or `gpu`), identifies the device type for which Haero
#   is built.
# * <precision> (either `single` or `double`) determines the precision of
#   floating point numbers used in Haero. Default: double
# * <build_type> (either `Debug` or `Release`) determines whether Haero is built
#   optimized or for debugging. Default: Debug
# OPTIONAL Args (REQUIRED for GPU builds)
# * [gpu_type] (`nvidia`, `amd`, `intel`) this is the "brand" of GPU
# * [gpu_arch] (e.g., HOPPER90, AMD_GFX90A, INTEL_PVC) this is the "model" of
#              GPU, indicating the generation, compute capability, etc.
#   * A list of many options is found in $HAERO_ROOT/setup
#   * The full list of options may be found on the kokkos docs site:
# https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#architectures
#
# NOTE: This script disables MPI, since the mam4xx team is focused on single-
# NOTE: node parallelism. If you need an MPI-enabled build of Haero, please
# NOTE: follow the installation directions in that repo.

PREFIX=$1
DEVICE=$2
PRECISION=$3
BUILD_TYPE=$4
GPU_TYPE=$5
DEVICE_ARCH=$6
# Turn off search for yaml libraries. EKAT will build yaml-cpp from submodules.
SKIP_FIND_YAML_CPP=ON

if [[ "$PREFIX" == "" ]]; then
  echo "Haero installation prefix was not specified!"
  echo "Usage: $0 <prefix> <device> <precision> <build_type> [gpu_type] [gpu_arch]"
  exit
fi

# Set defaults.
if [[ "$DEVICE" == "" ]]; then
  DEVICE=cpu
  echo "No device specified. Selected cpu."
fi
if [[ "$PRECISION" == "" ]]; then
  PRECISION=double
  echo "No floating point precision specified. Selected double."
fi
if [[ "$BUILD_TYPE" == "" ]]; then
  BUILD_TYPE=Debug
  echo "No build type specified. Selected Debug."
fi

# Validate options
if [[ "$DEVICE" != "cpu" && "$DEVICE" != "gpu" ]]; then
  echo "Invalid device specified: $DEVICE"
  exit
fi
if [[ "$PRECISION" != "single" && "$PRECISION" != "double" ]]; then
  echo "Invalid precision specified: $PRECISION (must be single or double)"
  exit
fi
# This should format the string as required (first letter capitalized)
BUILD_TYPE="${BUILD_TYPE^}"
if [[ "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Release" ]]; then
  echo "Invalid optimization specified: $BUILD_TYPE"
  echo "Must be Debug or Release (case-sensitive)"
  exit
fi

# Clone a fresh copy Haero in the current directory. Delete any existing copy.
if [[ -d $(pwd)/.haero ]]; then
  rm -rf $(pwd)/.haero
fi
echo "Cloning Haero repository into $(pwd)/.haero..."
git clone git@github.com:eagles-project/haero.git .haero || exit
cd .haero || exit
git submodule update --init --recursive || exit

cd ..

# ==============================================================================
# below are the default compiler choices for gpu builds and are set based on
# the command line args.
# ==============================================================================
if [[ "$DEVICE" == "gpu" ]]; then
  ENABLE_GPU=ON
  if [[ "$GPU_TYPE" == "" ]]; then
    echo "ERROR: GPU_TYPE not provided for GPU build."
    echo "Must be nvidia, amd, or intel (case-sensitive)"
    exit
  elif [[ "$GPU_TYPE" != "nvidia" && "$GPU_TYPE" != "amd" && "$GPU_TYPE" != "intel" ]]; then
    echo "Device provided as 'gpu', but invalid GPU_TYPE specified: ${GPU_TYPE}."
    echo "Must be nvidia, amd, or intel (case-sensitive)"
    exit
  fi
  if [[ "$DEVICE_ARCH" == "" ]]; then
    echo "ERROR: DEVICE_ARCH not provided for GPU build."
    exit
  fi
  # ==============================================================================
  # NVIDIA GPU + gcc
  # NOTE: if CXX is set to nvcc_wrapper, then this must be the same path used
  # in the `sed` command below
  # This happens by default via the $nvcw variable and should typically work
  # unless your environment differs from the default
  # ==============================================================================
  if [[ "$GPU_TYPE" == "nvidia" ]]; then
    echo "GPU_TYPE given as nvidia (${DEVICE_ARCH})--setting default compilers."
    echo "and modifying nvcc_wrapper with correct architecture flag."
    CXX="$(pwd)/.haero/ext/ekat/extern/kokkos/bin/nvcc_wrapper"
    nvcw=$CXX
    CC=gcc
    echo "C++ compiler: ${CXX}"
    echo "C compiler: ${CC}"
    # ==========================================================================
    if [[ -x "$nvcw" ]]; then
      # FIXME: this will not work for compute capbility 10.0
      CUDA_GEN=${DEVICE_ARCH:(-2)}
      # FIXME: this assumes a default value in nvcc_wrapper
      sed -i s/default_arch=\"sm_70\"/default_arch=\"sm_"$CUDA_GEN"\"/g "${nvcw}"
      echo "===================================================================="
      echo "nvcc_wrapper modified--verify that default_arch=sm_${CUDA_GEN}"
      echo "===================================================================="
      grep -i "default_arch=" "${nvcw}"
      echo "===================================================================="
    else
      echo "ERROR: nvcc_wrapper not found at expected location for nvidia gpu build."
      exit
    fi
# ==============================================================================
# AMD GPU + HIP and clang
# ==============================================================================
  elif [[ "$GPU_TYPE" == "amd" ]]; then
    echo "GPU_TYPE given as amd--setting default compilers."
    CXX=hipcc
    CC=amdclang
    echo "C++ compiler: ${CXX}"
    echo "C compiler: ${CC}"
# ==============================================================================
# Intel GPU + intel compilers
# ==============================================================================
  elif [[ "$GPU_TYPE" == "intel" ]]; then
    echo "GPU_TYPE given as intel--setting default compilers."
    CXX=icpx
    CC=icx
    echo "C++ compiler: ${CXX}"
    echo "C compiler: ${CC}"
# ==============================================================================
  else
    echo "Device provided as 'gpu', but invalid GPU_TYPE specified: ${GPU_TYPE}."
    echo "Must be nvidia, amd, or intel (case-sensitive)"
    echo "More importantly... how did we get here?"
    exit
  fi
else
  ENABLE_GPU=OFF
fi

if [[ "$DEVICE" == "cpu" ]]; then
  # Default cpu compilers (can be overridden by environment variables)
  echo "Setting compilers for CPU device."
  if [[ -z $CC ]]; then
    echo "'CC' environment variable not found--setting to default (gcc)"
    CC=gcc
  fi
  if [[ -z $CXX ]]; then
    echo "'CXX' environment variable not found--setting to default (g++)"
    CXX=g++
  fi
fi

# ==============================================================================
#     DON'T CHANGE ANYTHING BELOW HERE UNLESS YOU KNOW WHAT YOU'RE DOING
# ==============================================================================

echo "Configuring Haero with the given selections (WITHOUT MPI)..."
cmake -S ./.haero -B ./.haero/build \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER=$CC \
  -DHAERO_SKIP_FIND_YAML_CPP=$SKIP_FIND_YAML_CPP \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DHAERO_ENABLE_MPI=OFF \
  -DHAERO_ENABLE_GPU=$ENABLE_GPU \
  -DHAERO_PRECISION=$PRECISION \
  -DKokkos_ARCH_$DEVICE_ARCH:BOOL=ON \
  -DHAERO_DEVICE_ARCH=$DEVICE_ARCH \
  -G "Unix Makefiles" \
  || exit

echo "Building and installing Haero in $PREFIX..."
cd .haero/build || exit
make -j8 install || exit

cd ../../
echo "Haero has been installed in $PREFIX. Set HAERO_DIR to this directory in"
echo "your config.sh script after running setup."
