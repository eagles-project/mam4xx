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

set -xv

# TODO - add more verification to ensure variables are set before proceeding
echo $BUILD_TYPE "detected for BUILD_TYPE"
echo $HAERO_INSTALL "detected for HAERO install location"
echo $PRECISION "detected for PRECISION"

. /etc/profile.d/modules.sh
module purge
module load cmake
module load gcc/9.1.0
module load cuda/11.4
module load python/3.7.0

# Need to set env variables to get compiler set correctly
export CC=$(which gcc) CXX=$(which g++) FC=$(which gfortran)

# Need to clone in validation submodule as we are unable to clone automatically
perl -i -p -e 's|git@(.*?):|https://\1/|g' .gitmodules || exit
git submodule update --init || exit

cmake \
  -DMAM4XX_HAERO_DIR=$HAERO_INSTALL \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/install \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -B build -S $(pwd) \
  -G "Unix Makefiles" && \

cmake --build build -- -j && \
cd build && ctest -V -E ndrop

EXIT_CODE=$?

set +xv

exit $EXIT_CODE
