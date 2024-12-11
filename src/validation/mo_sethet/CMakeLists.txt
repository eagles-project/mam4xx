set(SETHET_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/mo_sethet)
set(SETHET_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all sethet-related parameterizations.
add_executable(sethet_driver
               sethet_driver.cpp
               calc_het_rates.cpp
               calc_precip_rescale.cpp
               find_ktop.cpp
               gas_washout.cpp
               sethet.cpp
               )

target_link_libraries(sethet_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${SETHET_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    calc_het_rates_merged
    calc_precip_rescale_merged
    find_ktop_merged
    gas_washout_merged
    sethet_merged
    )
# # matching the tests and errors, just for convenience

set(DEFAULT_TOL 1e-13)
set(ERROR_THRESHOLDS
    9e-8 # calc_het_rates, this is due to using a more specific value for mass_h2o
    ${DEFAULT_TOL} # calc_precip_rescale
    ${DEFAULT_TOL} # find_ktop, output is an int
    9e-8 # gas_washout
    9e-7 # sethet
   )
foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${SETHET_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} sethet_driver ${SETHET_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()
