set(MO_SETSOX_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/mo_setsox)
set(MO_SETSOX_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all mo_setsox-related parameterizations.
add_executable(mo_setsox_driver
               mo_setsox_driver.cpp
               setsox_test.cpp
               setsox_test_nlev.cpp
               calc_ph_values.cpp
               calc_sox_aqueous.cpp
               calc_ynetpos.cpp
               )

target_link_libraries(mo_setsox_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${MO_SETSOX_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    setsox_ts_355_merged
    setsox_ts_355_nlev
    calc_ph_values_ts_355
    calc_sox_aqueous_ts_355_merged
    calc_ynetpos_ts_355
    )

set(DEFAULT_TOL 1e-11)

set(ERROR_THRESHOLDS
    ${DEFAULT_TOL} # setsox_test
    ${DEFAULT_TOL} # setsox_test_nlev
    ${DEFAULT_TOL} # calc_ph_values
    ${DEFAULT_TOL} # calc_sox_aqueous
    ${DEFAULT_TOL} # calc_ynetpos
   )

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${MO_SETSOX_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} mo_setsox_driver ${MO_SETSOX_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()
