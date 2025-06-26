#!/usr/bin/env bash
set -euo pipefail

#===============================================================================
# Script: quick_ne4pg2_cime_test_frontier.sh
#
# Summary:
# This script sets up and launches a test run of the E3SM model
# with the MAM4xx aerosol scheme on Frontier using a specified
# compiler and project ID.
#
# Main Tasks:
# - Clones the E3SM repository and checks out the specified branches
# - Updates submodules, including the external MAM4XX component
# - Creates a uniquely named temporary run directory in the scratch filesystem
# - Edits the build configuration for faster compilation (GMAKE_J = 80)
# - Creates and launches an E3SM test case (e.g., SMS_Ln5.ne4pg2...)
# - Prints instructions for monitoring the test result
#
# Notes:
# - This script aborts if the E3SM source directory already exists
# - Intended for single-use test runs; temporary directories are timestamped
#
# Usage:
#   $ bash quick_ne4pg2_cime_test_frontier.sh
#===============================================================================

#---------------------------------------------------------------
# Config / User Inputs
#---------------------------------------------------------------
readonly E3SM_BRANCH="overfelt/eamxx/diagnostics_AQ_and_GS"
readonly MAM4XX_BRANCH="overfelt/diagnostics_AQ_and_GS"

#Less common user inputs:
readonly PROJID="cli115"
readonly COMPILER="craygnu-hipcc"
readonly SCRATCH_DIR="/lustre/orion/$PROJID/proj-shared/$USER/mam4xx_eamxx_runs"

#---------------------------------------------------------------
# Setup
#---------------------------------------------------------------
ulimit -d unlimited
ulimit -s unlimited
ulimit -c unlimited

# Load environment modules
source /opt/cray/pe/lmod/lmod/init/bash

#color coding
readonly RED='\e[1;31m'
readonly GREEN='\e[1;32m'
readonly BLUE='\e[1;34m'
readonly NC='\e[0m' # No Color

#---------------------
# Main
#---------------------
main() {

    mkdir -p "$SCRATCH_DIR"
    require_dir_exists "$SCRATCH_DIR"

    cd "$SCRATCH_DIR"
    echo "Working in: $SCRATCH_DIR"
    newline && time_elapsed_min
    
    readonly SRC_NAME="E3SM"
    if [[ ! -d "$SRC_NAME" ]]; then
        echo "Cloning E3SM repository..."
        git clone git@github.com:E3SM-Project/E3SM.git > /dev/null 2>&1
        newline && time_elapsed_min
    else
        echo -e "${GREEN}**WARNING** ${NC}Using existing E3SM repo in: ${BLUE}$SCRATCH_DIR/$SRC_NAME ${NC}"
        newline
    fi
    require_dir_exists "$SRC_NAME"
    readonly CODE_ROOT="$SCRATCH_DIR/$SRC_NAME"
    cd "$CODE_ROOT"

    echo "Fetch latest from the repo and Check out the E3SM branch: $E3SM_BRANCH"
    git fetch origin > /dev/null 2>&1
    git checkout "$E3SM_BRANCH" > /dev/null 2>&1

    echo -e "${GREEN}**HARD resetting** ${NC} E3SM repository in: ${BLUE}$SCRATCH_DIR/$SRC_NAME ${NC} to branch ${BLUE}$E3SM_BRANCH${NC}"
    newline
    git reset --hard origin/"$E3SM_BRANCH" > /dev/null 2>&1
    
    echo "Initializing submodules..."
    git submodule deinit -f . > /dev/null 2>&1
    git submodule update --init --recursive > /dev/null 2>&1
    newline && time_elapsed_min

    echo "Switching MAM4xx to branch: $MAM4XX_BRANCH"
    require_dir_exists "externals/mam4xx"
    cd externals/mam4xx
    git fetch origin > /dev/null 2>&1
    git checkout "$MAM4XX_BRANCH" > /dev/null 2>&1
    newline && time_elapsed_min

    echo -e "${GREEN}**HARD resetting**${NC} MAM4xx repository in: ${BLUE}`pwd` ${NC} to branch ${BLUE}$MAM4XX_BRANCH${NC}"
    newline
    git reset --hard origin/"$MAM4XX_BRANCH" > /dev/null 2>&1

    local temp_dir="test_$(date +'%m-%d-%Y__%H_%M_%S')"
    readonly temp_dir
    cd "$SCRATCH_DIR"
    exit_if_dir_exists "$SCRATCH_DIR/$temp_dir"

    echo "Creating temporary directory: $SCRATCH_DIR/$temp_dir"
    mkdir "$temp_dir"
    cd "$temp_dir"

    cd "$CODE_ROOT"
    newline && time_elapsed_min

    echo "Updating GMAKE_J to 80 for faster builds"
    sed -i 's|<GMAKE_J>[0-9]*</GMAKE_J>|<GMAKE_J>80</GMAKE_J>|' "$CODE_ROOT/cime_config/machines/config_machines.xml"

    require_dir_exists "cime/scripts"
    cd cime/scripts

    module load cray-python/3.11.7

    newline && time_elapsed_min

    local test_ne4="SMS_Ln5.ne4pg2_ne4pg2.F2010-EAMxx-MAM4xx"
    readonly test_ne4
    echo "Launching test: $test_ne4"

    ./create_test "$test_ne4" --compiler "$COMPILER" -p "$PROJID" --walltime 00:05:00 \
        -t "${temp_dir}_mam4xx" -r "$SCRATCH_DIR/$temp_dir" \
        --output-root "$SCRATCH_DIR/$temp_dir" \
        > /dev/null

    newline && time_elapsed_min

    echo "================================================================================="
    echo "Results can be viewed by issuing: $SCRATCH_DIR/$temp_dir/cs.status.${temp_dir}_mam4xx"
    newline
    echo "Here are the results:"
    $SCRATCH_DIR/$temp_dir/cs.status.${temp_dir}_mam4xx
    echo "================================================================================="
    newline && time_elapsed_min
}

# Helper: Newline
newline() { echo ''; }

# Helper: Elapsed time
time_elapsed_min() {
    local endtime
    endtime=$(date +%s)
    local elapsed=$(( endtime - starttime ))
    printf "Time elapsed: %02d:%02d\n" $((elapsed / 60)) $((elapsed % 60))
}

# Helper: Exit if dir exists
exit_if_dir_exists() {
    local path="$1"
    if [[ -d "$path" ]]; then
        echo -e "${RED}Error: Directory '$path' already exists. Please remove it or choose a different location."
        newline
        exit 1
    fi
}

# Helper: Check if dir exists (required)
require_dir_exists() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        echo "${RED}Error: Required directory '$path' does not exist."
        newline
        exit 1
    fi
}

# Capture start time
readonly starttime=$(date +%s)
echo "Start Time: $(date +%T)"

#call the main function
main
