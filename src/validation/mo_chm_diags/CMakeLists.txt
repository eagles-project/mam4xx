set(MO_CHM_DIAGS_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/mo_chm_diags)
set(MO_CHM_DIAGS_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all mo_chm_diags-related parameterizations.
add_executable(mo_chm_diags_driver
               mo_chm_diags_driver.cpp
               het_diags.cpp
               chm_diags.cpp
               )

target_link_libraries(mo_chm_diags_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${MO_CHM_DIAGS_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()
# stand_calc_sum_wght_ts_355
# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    het_diags_ts_355
    chm_diags_ts_355
    )
# # matching the tests and errors, just for convenience

set(DEFAULT_TOL 1e-13)
set(ERROR_THRESHOLDS
   9e-2
   9e-2
   )
foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${MO_CHM_DIAGS_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} mo_chm_diags_driver ${MO_CHM_DIAGS_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()
