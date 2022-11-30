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

echo $BUILD_TYPE " detected for BUILD_TYPE"
echo $HAERO_INSTALL " detected for HAERO install location"
echo $PRECISION " detected for PRECISION"

# Since we are using CI, we must set environment variable
# This clones submodules manually over HTTPS instead of using SSH
export CI_HTTPS_INSTALL=1
# SYSTEM_NAME seems to work on login nodes, but not in compute instances, so we will export manually for PNNL
export SYSTEM_NAME=deception
# Similarly in CI we need to configure the right module path
# Perhpas this should just be added to HAERO directly
. /etc/profile.d/modules.sh
module purge

./build-haero.sh \
  $HAERO_INSTALL \
  gpu \
  $PRECISION \
  $BUILD_TYPE

EXIT_CODE=$?
exit $EXIT_CODE
