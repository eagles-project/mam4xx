#!/usr/bin/env bash

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Description:
#   This script checks out two branches/tags/commits locally, building them
#   in a specific configuration and running their validation tests, producing
#   a visually-interpretable comparison of their bit-for-bit (BFB) hashes.
#
# Usage:
#   ./check_bfb_bashes.sh <branch1> <branch2> <device> <precision> <build_type> [gpu_type] [gpu_arch]
# For now, this script accepts the same arguments as the build-haero.sh script,
# apart from the first two arguments, which specify branches/tags/commits.
#-------------------------------------------------------------------------------

# command line arguments 
branch1=$1
branch2=$2
device=$3
precision=$4
build_type=$5
gpu_type=$7
device_arch=$7

usage() {
  echo "$0: usage:"
  echo "$0 <branch1> <branch2> <cpu|gpu> <single|double> <Debug|Release> [nvidia|amd|intel] [device-arch]"
}

main() {
  # check arguments
  if [ "$branch1" == "" ] || [ "$branch2" == "" ] || [ "$device" == "" ]; then
    usage
    exit 1
  fi

  # validate and set defaults
  if [ "$precision" == "" ]; then
    precision=double
  elif [ "$precision" != "single" ] && [ "$precision" != "double" ]; then
    usage
    exit 1
  fi
  if [ "$build_type" == "" ]; then
    build_type=Release
  elif [ "$build_type" != "Debug" ] && [ "$build_type" != "Release" ]; then
    usage
    exit 1
  fi

  # check GPU-related arguments
  if [ "$device" != "cpu" ] && [ "$device" != "gpu" ]; then
    usage
    exit 1
  fi
  if [ "$device" == "gpu" ]; then
    if [ "$gpu_type" != "intel" ] && [ "$gpu_type" == "amd" ] && [ "$gpu_type" == "nvidia" ]; then
      usage
      echo "(Missing gpu type: must be intel, amd, or nvidia)"
      exit 1
    fi
    if [ "$device_arch" == "" ]; then
      usage
      echo "(Missing device arch)"
      exit 1
    fi
  fi

  temp_dir="$(mktemp -d bfb-comparison-XXXXXX)"
  echo "Comparing bit-for-bit hashes in temporary directory $temp_dir."

  # clone repos into the temporary directory

  clone_repo eagles-project/mam4xx $branch1 $temp_dir/$branch1 &
  pid1=$!
  clone_repo eagles-project/mam4xx $branch2 $temp_dir/$branch2 &
  pid2=$!

  wait $pid1
  status1=$?
  wait $pid2
  status2=$?

  if [ $status1 -ne 0 ] || [ $status2 -ne 0 ]; then
    echo "Error cloning branches from GitHub"
    exit 1
  fi

  # configure and build the branches

  build_branch $branch1 $temp_dir/$branch1 &
  pid1=$!
  build_branch $branch2 $temp_dir/$branch2 &
  pid2=$!
   
  wait $pid1
  status1=$?
  wait $pid2
  status2=$?

  if [ $status1 -ne 0 ] || [ $status2 -ne 0 ]; then
    echo "Error building branches"
    exit 1
  fi

  # run the validation tests (serially)
  run_branch $branch1 $temp_dir/$branch1
  status1=$?
  if [ $status1 -ne 0 ]; then
    echo "Error running validation tests for branch $branch1"
    exit 1
  fi

  run_branch $branch2 $temp_dir/$branch2
  status2=$?
  if [ $status2 -ne 0 ]; then
    echo "Error running validation tests for branch $branch2"
    exit 1
  fi

  compare_bfb_hashes $temp_dir/$branch1 $temp_dir/$branch2

  #rm -rf $temp_dir
}

#---------------------
# Function Definitions
#---------------------

clone_repo() {    
  local repo=$1 
  local branch=$2 
  local path=$3 

  echo "Cloning code from $branch branch of github.com/$repo to $path..."

  git clone git@github.com:$repo.git -b $branch $path >& clone.log
  if [ "$branch" != "jeff-cohere/merge-haero" ]; then
    pushd $path
    git submodule update --init --recursive >& submodules.log
    popd
  fi
} 

build_branch() {
  local branch=$1 
  local path=$2 

  echo "Building branch $branch in $path/build..."

  pushd $path

  if [ "$branch" != "jeff-cohere/merge-haero" ]; then
    local haero_dir=build/haero
    ./build-haero.sh $haero_dir $device $precision $build_type $gpu_type $device_arch >& haero.log
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=$build_type \
      -DMAM4XX_HAERO_DIR=`pwd`/$haero_dir \
      -DENABLE_SKYWALKER=ON \
      -G "Unix Makefiles" >& cmake.log
  else
    if [ "$device" == "gpu" ]; then
      cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=$build_type \
        -DMAM4XX_ENABLE_MPI=OFF \
        -DMAM4XX_ENABLE_GPU=ON \
        -DMAM4XX_DEVICE_ARCH=$device_arch \
        -DMAM4XX_PRECISION=$precision \
        -DMAM4XX_ENABLE_SKYWALKER=ON \
        -G "Unix Makefiles" >& cmake.log
    else
      cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=$build_type \
        -DMAM4XX_ENABLE_MPI=OFF \
        -DMAM4XX_ENABLE_GPU=OFF \
        -DMAM4XX_PRECISION=$precision \
        -DMAM4XX_ENABLE_SKYWALKER=ON \
        -G "Unix Makefiles" >& cmake.log
    fi
  fi
  cd build
  make -j8 > build.log

  popd
}

run_branch() {
  local branch=$1 
  local path=$2 

  pushd $path/build

  make test >& test.log

  popd
}

compare_bfb_hashes() {

  echo "Comparing $branch1 hashes to $branch2 hashes..."

  local path1=$1 
  local build_dir1=$path1/build
  local path2=$2
  local build_dir2=$path2/build

  grep -A 2 "mam4xx hash" $build_dir1/Testing/Temporary/LastTest.log > $build_dir1/bfb-hashes.txt
  grep -A 2 "mam4xx hash" $build_dir2/Testing/Temporary/LastTest.log > $build_dir2/bfb-hashes.txt
  diff $build_dir1/bfb-hashes.txt $build_dir2/bfb-hashes.txt
}

# Silent versions of popd and pushd
pushd() {
    command pushd "$@" > /dev/null
}
popd() {
    command popd "$@" > /dev/null
}

# Run the script
main 
