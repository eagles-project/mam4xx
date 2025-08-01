set(TROPOPAUSE_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/tropopause)
set(TROPOPAUSE_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all tropopause-related parameterizations.
add_executable(tropopause_driver
               tropopause_driver.cpp
               get_dtdz.cpp
               twmo.cpp
               tropopause_twmo.cpp
               )

target_link_libraries(tropopause_driver skywalker;validation;haero)

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${TROPOPAUSE_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()
# stand_calc_sum_wght_ts_355
# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    get_dtdz_merged
    twmo_merged
    tropopause_twmo_ts_1400
    )
# # matching the tests and errors, just for convenience

set(DEFAULT_TOL 1e-13)
set(ERROR_THRESHOLDS
   9e-3
   9e-4
   ${DEFAULT_TOL}  # Output is an integer.
   )
foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${TROPOPAUSE_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} tropopause_driver ${TROPOPAUSE_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()