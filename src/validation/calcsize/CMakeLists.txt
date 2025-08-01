
set(CALCSIZE_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/calcsize)
set(CALCSIZE_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all calcsize-related parameterizations.
add_executable(calcsize_driver calcsize_driver.cpp
               compute_dry_volume.cpp
               adjust_num_sizes.cpp
               compute_tendencies.cpp
               modal_aero_calcsize_sub.cpp
               modal_aero_calcsize_sub_ptend.cpp
               aitken_accum_exchange.cpp)
target_link_libraries(calcsize_driver skywalker;validation;haero)


# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${CALCSIZE_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    adjust_num_sizes_case1
    adjust_num_sizes_case2
    calcsize_e3sm
    calcsize_sub
    aitken_accum_exchange_case1
    calcsize_compute_dry_volume
    stand_modal_aero_calcsize_sub
    stand_modal_aero_calcsize_sub_update_ptend
    stand_calcsize_aero_model_wetdep_ts_379
    )
# # matching the tests and errors, just for convenience

set(DEFAULT_TOL 2e-6)
set(ERROR_THRESHOLDS
    1e-12
    1e-12
    2e-6
    2e-6
    1e-12
    1e-12
    5e-11
    3e-5
    1.5e-3
   )


foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${CALCSIZE_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} calcsize_driver ${CALCSIZE_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()
