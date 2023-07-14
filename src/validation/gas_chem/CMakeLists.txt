set(GAS_CHEM_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/gas_chem)
set(GAS_CHEM_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all gas_chem-related parameterizations.
add_executable(gas_chem_driver  gas_chem_driver.cpp linmat.cpp
               indprd.cpp)
target_link_libraries(gas_chem_driver skywalker;validation;${HAERO_LIBRARIES})


# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${GAS_CHEM_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    indprd_ts_355
    linmat_ts_355
    )

set(DEFAULT_TOL 1e-12)
# gas_chem

set(ERROR_THRESHOLDS  
   ${DEFAULT_TOL}
   ${DEFAULT_TOL}
   )

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.
  
  configure_file(
    ${GAS_CHEM_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} gas_chem_driver ${GAS_CHEM_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})

endforeach()