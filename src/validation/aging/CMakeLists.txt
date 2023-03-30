set(AGING_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/aging)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all aging-related parameterizations.
add_executable(aging_driver aging_driver.cpp
               mam_pcarbon_aging_frac.cpp
               mam_pcarbon_aging_1subarea.cpp)
target_link_libraries(aging_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py
        )
  configure_file(
    ${MAM_X_VALIDATION_DIR}/scripts/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.

set(TEST_LIST
    pcarbon_aging_frac
    pcarbon_aging_1subarea
    )

#set(DEFAULT_TOL 1e-9)

set(ERROR_THRESHOLDS 8e-5 4e-4)

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place; is the skywalker file produced by fortran code?
  
  configure_file(
    ${AGING_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} aging_driver ${AGING_VALIDATION_DIR}/${input}.yaml)

   # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})

endforeach()

