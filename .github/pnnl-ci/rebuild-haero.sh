#!/bin/bash

exit() {
  # Clear all trap handlers so this isn't echo'ed multiple times, potentially
  # throwing off the CI script watching for this output
  trap - `seq 1 31`

  # If called without an argument, assume not an error
  local ec=${1:-0}

  # Echo the snippet the CI script is looking for
  echo BUILD_STATUS:${ec}

  # Actually exit with that code, although it won't matter in most cases, as CI
  # is only looking for the string 'BUILD_STATUS:N'
  builtin exit ${ec}
}

# This will be the catch-all trap handler after arguments are parsed.
cleanup() {
  # Clear all trap handlers
  trap - `seq 1 31`

  # When 'trap' is invoked, each signal handler will be a curried version of
  # this function which has the first argument bound to the signal it's catching
  local sig=$1

  echo
  echo Exit code $2 caught in build script triggered by signal ${sig}.
  echo

  exit $2
}

echo $BUILD_TYPE "detected for BUILD_TYPE"
echo $HAERO_INSTALL "detected for HAERO install location"
echo $PRECISION "detected for PRECISION"
echo $SYSTEM_NAME "is the target cluster"

PREFIX=$HAERO_INSTALL
DEVICE=gpu
PRECISION=$PRECISION
BUILD_TYPE=$BUILD_TYPE

# Default compilers (can be overridden by environment variables)
if [[ -z $CC ]]; then
  CC=cc
fi
if [[ -z $CXX ]]; then
  CXX=c++
fi

if [[ "$PREFIX" == "" ]]; then
  echo "Haero installation prefix was not specified!"
  echo "Usage: $0 <prefix> <device> <precision> <build_type>"
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
if [[ "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Release" ]]; then
  echo "Invalid optimization specified: $BUILD_TYPE (must be Debug or Release)"
  exit
fi

# We need to keep a copy of HAERO src code present for the mometn
# TODO - remove this `cd` and install without referencing source code
cd $PREFIX

# Clone a fresh copy Haero in the current directory. Delete any existing copy.
if [[ -d $(pwd)/.haero ]]; then
  rm -rf $(pwd)/.haero
fi
echo "Cloning Haero repository into $(pwd)/.haero..."

# Need to clone HAERO using HTTPS instead of SSH
git clone https://github.com/eagles-project/haero.git .haero || exit
cd .haero || exit

# Need to modify .gitmodules file before cloning
perl -i -p -e 's|git@(.*?):|https://\1/|g' .gitmodules || exit
# Update just haero submodules
git submodule update --init || exit

# Go through and repeat the process for submodules with submodules
declare -a arr=("ekat")
for subm in "${arr[@]}"
do
  pushd ./ext/$subm
  perl -i -p -e 's|git@(.*?):|https://\1/|g' .gitmodules || exit
  git submodule update --init || exit
  popd
done

# Are we on a special machine?
cd machines
echo $(pwd)
for MACHINE_FILE in $(ls)
do
  MACHINE=${MACHINE_FILE/\.sh/}
  echo $MACHINE
  echo `hostname` | grep -q "$MACHINE"
  host_match=$?
  echo $SYSTEM_NAME | grep -q "$MACHINE"
  sys_match=$?
  if  [ $host_match -eq 0 ] || [ $sys_match -eq 0 ]; then
    echo "Found machine file $MACHINE_FILE. Setting up environment for $MACHINE..."
    source ./$MACHINE.sh
  fi
done

cd ../..

# Configure Haero with the given selections.
if [[ "$DEVICE" == "gpu" ]]; then
  ENABLE_GPU=ON
else
  ENABLE_GPU=OFF
fi

echo "Configuring Haero with the given selections (WITHOUT MPI)..."
cmake -S ./.haero -B ./.haero/build \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DHAERO_ENABLE_MPI=OFF \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DHAERO_ENABLE_GPU=$ENABLE_GPU \
  -DKokkos_ARCH_PASCAL60=ON \
  -DHAERO_PRECISION=$PRECISION

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  exit $EXIT_CODE
fi

cd ./.haero/build || exit
echo "Building and installing Haero in $PREFIX..."
make -j8 install

EXIT_CODE=$?

cd ../../
echo "Haero has been installed in $PREFIX. Set HAERO_DIR to this directory in"
echo "your config.sh script after running setup."

echo "Cleaning up HAERO git clone in $PWD/.haero"
rm -rf $(pwd)/.haero

exit $EXIT_CODE
