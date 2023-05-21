set(NDROP_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/ndrop)
set(NDROP_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all ndrop-related parameterizations.
add_executable(ndrop_driver  
               ndrop_driver.cpp
               ccncalc.cpp
               ccncalc_single_cell.cpp
               get_activate_frac.cpp
               loadaer.cpp
               activate_modal.cpp
               explmix.cpp
               maxsat.cpp
               )

target_link_libraries(ndrop_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${NDROP_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    loadaertype1_merged
    loadaertype2_merged
    loadaertype3_merged
    activate_modal_ts_1400_type1
    explmix_ts_1417_actmm1
    stand_get_activate_frac_ts_1417
    activate_modal_ts_1417_type2
    stand_activate_modal_ts_1417_type2
    activate_modal_merged
    ccncalc_merged
    ccncalc_loop_ts_1417
    ccncalc_ts_1400
    maxsat_ts_1400_type1
    )

set(DEFAULT_TOL 1e-11)

set(ERROR_THRESHOLDS
    6e-8
    6e-8
    6e-8
    8e-2
    9e-2
    5e-5
    1e-5
    5e-6
    2e-3
    2e-3
    1e-3
    5e-5 
    1e-11
   )

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.
  
  configure_file(
    ${NDROP_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} ndrop_driver ${NDROP_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})

endforeach()
