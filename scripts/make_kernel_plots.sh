#!/bin/bash

# NOTE: This script only works on Perlmutter at the moment, and relies on tools
# NOTE: built in Meng's directory.

case_dir=/pscratch/sd/m/meng/scream_scratch/SMS_P32x1_Ld10.ne30pg2_oECv3.F2010-EAMxx-MAM4xx.pm-gpu_gnugpu.ThirdTimingViews/run
plotdir=/pscratch/sd/m/meng/tmp/kokkos_tools
exe_kk="/global/u1/m/meng/utils/kokkos-tools/profiling/simple-kernel-timer/kp_json_writer"
# step 1. get a list of ${case_dir}/nid*.dat files.
nid_files=("${case_dir}"/nid*.dat)

if [[ ${#nid_files[@]} -eq 0 ]]; then
  echo "No nid*.dat files found in ${case_dir}"
  exit 1
fi

# step 2. run plot_kernel_times.py and save a plot per each nid* file.
# for nid_file in "${nid_files[@]}"; do
nid_file="${nid_files[0]}" 
  # echo $nid_file
  base=$(basename "${nid_file}" .dat)
  # echo $base
  if [[ ! -f "${base}.json" || "${nid_file}" -nt "${base}.json" ]]; then
   ${exe_kk} "${nid_file}" > "${plotdir}/${base}.json"  
  fi
  # echo "${exe_kk} "${nid_file}" > "${plotdir}/${base}.json""
  
  output="${plotdir}/${base}_kernel_times.png"
  echo $output
  python "plot_kernel_times.py" "${plotdir}/${base}.json" \
    -o "${output}" -t "Test Branch Kernel Times" \
    -k "MAMMicrophysics::run_impl::sethet" \
       "MAMAci::run_impl::call_function_dropmixnuc" \
       "MAMWetscav::run_impl::aero_model_wetdep" \
    # -n 10    
# done


