#!/bin/bash

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Description:
#   This script automates the process of comparing model performance between master and development branches of E3SM.
#
# Usage:
#   Customize the user input section below before running the script.
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

main() {

    #---------------------------------------------------------------
    # User-defined configuration
    #---------------------------------------------------------------

    # Perlmutter
    machine=pm-cpu     #pm-gpu 
    compiler=gnu       #gnugpu 
    project=e3sm 
    workdir=/pscratch/sd/m/meng/compare_model_performance 
    plotdir=/global/cfs/cdirs/e3sm/www/meng/compare_performance  # make sure this a www in your Community directory to check the plot as a html page
    html_address=https://portal.nersc.gov/cfs/e3sm/meng/compare_performance
    module load python

    # Compy
    # machine=compy
    # compiler=intel
    # project=e3sm 
    # workdir=/compyfs/huan967/compare_model_performance 
    # plotdir=/compyfs/www/huan967/compare_performance
    # html_address=https://compy-dtn.pnl.gov/huan967/compare_performance  
    # module load python/miniconda4.12.0 

    if [ ! -d $plotdir ]; then
        mkdir -p $plotdir 
    fi

    compset=F2010-EAMxx-MAM4xx   #F2010-SCREAMv1 
    resolution=ne4pg2_ne4pg2 
    pe=16
    # resolution=ne30pg2_ne30pg2  
    # pe=32 
    runtime=Ln5 
    queue=debug
    wallclock_time=00:10:00

    # SMS test run
    case=SMS_P${pe}_${runtime}.${resolution}.${compset}.${machine}_${compiler}

    user1=E3SM-Project 
    branch1=master
    user2=odiazib
    branch2=eamxx/mo_photo_cloud_mod

    casename1=e3sm-master
    casename2=odiazib-mo_photo_cloud_mod 

    code_root1=$workdir/E3SM-$casename1 
    code_root2=$workdir/E3SM-$casename2

    do_fetch_code=true   
    do_run_case=true  

    # If you only need to tweak the plot, set the plot_str to the date of the run.
    do_plot=true 
    # plot_str=20250503_173057

    #---------------------------------------------------------------
    # User-defined configuration ENDs
    #---------------------------------------------------------------

    datestr=`date +'%Y%m%d_%H%M%S'`

    # Fetch code from GitHub 
    fetch_code $user1 $branch1 $code_root1 &
    pid1=$!
    fetch_code $user2 $branch2 $code_root2 &
    pid2=$!

    # Wait for each process and check its exit status
    wait $pid1
    status1=$?
    wait $pid2
    status2=$?

    if [ $status1 -ne 0 ] || [ $status2 -ne 0 ]; then
        echo "Error fetching code from GitHub"
        exit 1
    fi

    # Create SMS test case and run it 
    run_case $code_root1 $casename1 &
    # run_hr_case $code_root1 $casename1 $pe &  #ne256 and ne1024 on GPUs 
    pid1=$!
    run_case $code_root2 $casename2 &
    # run_hr_case $code_root2 $casename2 $pe & 
    pid2=$!

    # Wait for each and capture status
    wait $pid1
    status1=$?
    wait $pid2
    status2=$?

    if [ $status1 -ne 0 ] || [ $status2 -ne 0 ]; then
        echo "Error compiling or running case"
        exit 1
    fi

    # Grab timing data and do the plotting
    if [ "${do_plot,,}" != "true" ]; then
        echo $'\n----- Skipping plot -----\n'
        return
    fi
    if [ -z "${plot_str+x}" ]; then
        plot_str=$datestr 
    fi

    case_root1=$workdir/$case.${casename1}_${plot_str}  
    case_root2=$workdir/$case.${casename2}_${plot_str}
    
    python compare_model_performance_plot.py \
        --case1 $case_root1 --casename1 $casename1 \
        --case2 $case_root2 --casename2 $casename2 \
        --outdir $plotdir --html $html_address              

    if [ $? != 0 ]; then
        echo "Error plotting model performance"
        exit 1
    fi

    echo "Model performance comparison complete!"

}

#---------------------
# Function Definitions
#---------------------

fetch_code() {    
    if [ "${do_fetch_code,,}" != "true" ]; then
        echo $'\n----- Skipping fetch_code -----\n'
        return
    fi
    local username=$1 
    local branch=$2 
    local path=$3 

    echo "Fetching code from $branch branch to $path..."

    if [ -d $path ]; then
        pushd $path 
        echo "Code directory $path already exists. Updating $branch branch..."
        git fetch origin
        git checkout $branch 
        git reset --hard origin/$branch 
        git submodule deinit -f . && git submodule update --init --recursive
        popd 
    else
        mkdir -p $path 
        pushd $path 
        git clone git@github.com:$username/E3SM.git .     
        if [ "$branch" != "master" ]; then
            git checkout $branch
        fi
        git submodule update --init --recursive     
        popd
    fi 
} 

run_case() {
    if [ "${do_run_case,,}" != "true" ]; then
        echo $'\n----- Skipping create_newcase -----\n'
        return
    fi

    local code_root=$1 
    local casename=$2 

    local interval=60  # seconds to wait between each check 

    path=$workdir/$case.${casename}_${datestr} 

    # create new case and build it 
    echo "Creating case $case.${casename}_${datestr}..."
    $code_root/cime/scripts/create_test $case \
        -t ${casename}_${datestr} -p $project -q $queue --output-root $workdir --no-run #--no-build   

    if [ $? != 0 ]; then
        echo "Error creating case $case"
        exit 1
    fi   

    # submit job and monitor status
    pushd $path 
    ./xmlchange JOB_WALLCLOCK_TIME=$wallclock_time 
    ./case.submit >& log.submit 

    if [ $? != 0 ]; then
        echo "Error submitting job for $case"
        exit 1
    fi

    jobID=`awk -F " " '($1=="Submitted") {print $NF}' log.submit | tail -n 1 `
    
    while true; do 
        sacct --job $jobID > log.job_stat
        jobST=` awk "{if(NR==3) print}" log.job_stat | awk '{printf"%s\n",$6}'`

        if [ "${jobST}" == "COMPLETED" ]; then
            printf "%s Run complete: %-30s jobID: %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$casename" "$jobID"
            break 
        elif [[ ("${jobST}" == "FAILED") || ("${jobST}" == *"CANCELLED"*) || ("${jobST}" == "TIMEOUT") ]]; then
            printf "%s Run failed: %-30s jobID: %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$casename" "$jobID"
            exit 1 
        else
            printf "%s Waiting for case: %-30s to complete...  jobID: %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$casename" "$jobID"
            sleep $interval
        fi
    done 

    popd 
}

# for the high-resolution runs on GPUs: ne256 and ne1024 
gpu_layout() {
    mpi_per_node=4
    ntatm=16
    ntcpl=1
    nt=8
    np=$1 

    ./xmlchange --file env_mach_pes.xml MAX_MPITASKS_PER_NODE=$mpi_per_node

    ./xmlchange --file env_mach_pes.xml NTASKS_ATM="$np"
    ./xmlchange --file env_mach_pes.xml NTASKS_LND="$np"
    ./xmlchange --file env_mach_pes.xml NTASKS_ICE="$np"
    ./xmlchange --file env_mach_pes.xml NTASKS_OCN="$np"
    ./xmlchange --file env_mach_pes.xml NTASKS_CPL="$np"
    ./xmlchange --file env_mach_pes.xml NTASKS_ROF="$np"
    ./xmlchange --file env_mach_pes.xml NTASKS_GLC="1"
    ./xmlchange --file env_mach_pes.xml NTASKS_WAV="1"

    ./xmlchange --file env_mach_pes.xml NTHRDS_ATM="$ntatm"
    ./xmlchange --file env_mach_pes.xml NTHRDS_LND="$nt"
    ./xmlchange --file env_mach_pes.xml NTHRDS_ICE="$nt"
    ./xmlchange --file env_mach_pes.xml NTHRDS_OCN="$nt"
    ./xmlchange --file env_mach_pes.xml NTHRDS_CPL="$ntcpl"
    ./xmlchange --file env_mach_pes.xml NTHRDS_GLC="1"
    ./xmlchange --file env_mach_pes.xml NTHRDS_ROF="$nt"
    ./xmlchange --file env_mach_pes.xml NTHRDS_WAV="1"
}

run_hr_case() {
    if [ "${do_run_case,,}" != "true" ]; then
        echo $'\n----- Skipping create_newcase -----\n'
        return
    fi

    local code_root=$1 
    local casename=$2 

    local interval=60  # seconds to wait between each check 

    path=$workdir/$case.${casename}_${datestr} 

    # create new case and build it 
    echo "Creating case $case.${casename}_${datestr}..."
    $code_root/cime/scripts/create_test $case \
        -t ${casename}_${datestr} -p $project -q $queue --output-root $workdir --no-run --no-build   

    if [ $? != 0 ]; then
        echo "Error creating case $case"
        exit 1
    fi   

    # submit job and monitor status
    pushd $path 
    gpu_layout $3 
    ./case.setup -r && ./case.build 
    if [ $? != 0 ]; then
        echo "Error building case $case"
        exit 1
    fi 
    ./xmlchange JOB_WALLCLOCK_TIME=$wallclock_time,PIO_NETCDF_FORMAT=64bit_data 
    ./case.submit >& log.submit 

    if [ $? != 0 ]; then
        echo "Error submitting job for $case"
        exit 1
    fi

    jobID=`awk -F " " '($1=="Submitted") {print $NF}' log.submit | tail -n 1 `
    
    while true; do 
        sacct --job $jobID > log.job_stat
        jobST=` awk "{if(NR==3) print}" log.job_stat | awk '{printf"%s\n",$6}'`

        if [ "${jobST}" == "COMPLETED" ]; then
            printf "%s Run complete: %-30s jobID: %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$casename" "$jobID"
            break 
        elif [[ ("${jobST}" == "FAILED") || ("${jobST}" == *"CANCELLED"*) || ("${jobST}" == "TIMEOUT") ]]; then
            printf "%s Run failed: %-30s jobID: %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$casename" "$jobID"
            exit 1 
        else
            printf "%s Waiting for case: %-30s to complete...  jobID: %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$casename" "$jobID"
            sleep $interval
        fi
    done 

    popd 
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
