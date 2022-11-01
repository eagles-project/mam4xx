#!/usr/bin/env bash

# This script builds the Haero high-performance aerosool toolkit with specific
# settings so mam4xx can be linked against it. Run it like this:
#
# `./build-haero.sh <prefix> <device> <precision> <packsize> <opt>
#
# where
# * <prefix> is the installation prefix (e.g. /usr/local) in which you
#   would like Haero installed
# * <device> (either cpu:ARCH or gpu:ARCH), identifies the device ARCHitecture
#   for which Haero is built. Some examples:
#   `cpu:AMDAVX`   - AMD CPU with AVX instructions (default)
#   `gpu:AMPERE80` - NVIDIA Ampere GPU
# * <precision> (either `single` or `double`) determines the precision of
#   floating point numbers used in Haero. Default: double
# * <packsize> (an integer such as 1, 4, 8) determines the number of values in
#   a Pack used for vectorization, mainly on CPUs (most GPUs use 1). Default: 1
# * <build_type> (either `Debug` or `Release`) determines whether Haero is built
#   optimized or for debugging. Default: Debug
#
# NOTE: This script disables MPI, since the mam4xx team is focused on single-
# NOTE: node parallelism. If you need an MPI-enabled build of Haero, please
# NOTE: follow the installation directions in that repo.

PREFIX=$1
DEVICE=$2
PRECISION=$3
PACKSIZE=$4
BUILD_TYPE=$5

# Default compilers (can be overridden by environment variables)
if [[ -z $CC ]]; then
  CC=cc
fi
if [[ -z $CXX ]]; then
  CXX=c++
fi

if [[ "$PREFIX" == "" ]]; then
  echo "Haero installation prefix was not specified!"
  echo "Usage: $0 <prefix> <device> <precision> <packsize> <build_type>"
  exit
fi

# Set defaults.
if [[ "$DEVICE" == "" ]]; then
  DEVICE=cpu:AMDAVX
  echo "No device specified. Selected cpu:AMDAVX."
fi
if [[ "$PRECISION" == "" ]]; then
  PRECISION=double
  echo "No floating point precision specified. Selected double."
fi
if [[ "$PACKSIZE" == "" ]]; then
  PACKSIZE=1
  echo "No pack size specified. Selected 1."
fi
if [[ "$BUILD_TYPE" == "" ]]; then
  BUILD_TYPE=Debug
  echo "No build type specified. Selected Debug."
fi

# Validate options
if [[ "$DEVICE" != "cpu:"* && "$DEVICE" != "gpu:"* ]]; then
  echo "Invalid device specified: $DEVICE"
  exit
fi
if [[ "$PRECISION" != "single" && "$PRECISION" != "double" ]]; then
  echo "Invalid precision specified: $PRECISION (must be single or double)"
  exit
fi
# FIXME: pack size?
if [[ "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Release" ]]; then
  echo "Invalid optimization specified: $BUILD_TYPE (must be Debug or Release)"
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

# Configure Haero with the given selections.
if [[ "$DEVICE" == "cpu:"* ]]; then
  ENABLE_GPU=OFF
  DEVICE_ARCH="${DEVICE/cpu:/}"
else
  ENABLE_GPU=ON
  DEVICE_ARCH="${DEVICE/gpu:/}"
fi

echo "Configuring Haero with the given selections (WITHOUT MPI)..."
cmake -S ./.haero -B ./.haero/build \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DHAERO_ENABLE_MPI=OFF \
  -DHAERO_ENABLE_GPU=$ENABLE_GPU \
  -DHAERO_DEVICE_ARCH=$DEVICE_ARCH \
  -DKokkos_ARCH_$DEVICE_ARCH:BOOL=ON \
  -DHAERO_PRECISION=$PRECISION \
  -DHAERO_PACK_SIZE=$PACKSIZE \
  || exit

echo "Building and installing Haero in $PREFIX..."
cd .haero/build || exit
make -j8 install

cd ../../
echo "Haero has been installed in $PREFIX. Set HAERO_DIR to this directory in"
echo "your config.sh script after running setup."
