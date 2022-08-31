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
# * <opt> (either `debug` or `release`) determines whether Haero is built
#   optimized or for debugging. Default: debug

PREFIX=$1
DEVICE=$2
PRECISION=$3
PACKSIZE=$4
OPT=$5

# Default compilers (can be overridden by environment variables)
if [[ -z $CC ]]; then
  CC=cc
fi
if [[ -z $CXX ]]; then
  CXX=c++
fi

if [[ "$PREFIX" == "" ]]; then
  echo "Haero installation prefix was not specified!"
  echo "Usage: $0 <prefix> <device> <precision> <packsize> <opt>"
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
if [[ "$OPT" == "" ]]; then
  OPT=debug
  echo "No optimization level specified. Selected debug."
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
if [[ "$OPT" != "debug" && "$OPT" != "release" ]]; then
  echo "Invalid optimization specified: $OPT (must be debug or release)"
  exit
fi

# Capitalize the optimization setting
OPT=${OPT^}

# Clone Haero in the current directory (if needed).
if [[ ! -d $(pwd)/.haero ]]; then
  echo "Cloning Haero repository into $(pwd)/.haero..."
  git clone git@github.com:eagles-project/haero.git .haero || exit
  cd .haero || exit
  git submodule update --init --recursive || exit
  cd ..
fi

# Configure Haero with the given selections.
if [[ "$DEVICE" == "cpu:"* ]]; then
  ENABLE_GPU=OFF
  DEVICE_ARCH="${DEVICE/cpu:/}"
else
  ENABLE_GPU=ON
  DEVICE_ARCH="${DEVICE/gpu:/}"
fi

echo "Configuring Haero with the given selections..."
cmake -S ./.haero -B ./.haero/build \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
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
