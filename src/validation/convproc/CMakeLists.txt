
set(CONVPROC_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/convproc)
set(CONVPROC_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all convproc-related parameterizations.
add_executable(convproc_driver convproc_driver.cpp
               update_tendency_final.cpp
               compute_column_tendency.cpp
               ma_precpevap.cpp
               ma_precpprod.cpp
               initialize_dcondt.cpp
               compute_downdraft_mixing_ratio.cpp
               ma_precpevap_convproc.cpp
               ma_resuspend_convproc.cpp)
target_link_libraries(convproc_driver skywalker;validation;${HAERO_LIBRARIES})


# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${CONVPROC_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.

# convproc
foreach (input
         update_tendency_final
         compute_column_tendency
         ma_precpevap
         ma_precpprod
         initialize_dcondt
         compute_downdraft_mixing_ratio
         ma_precpevap_convproc
         ma_resuspend_convproc
         )

  # copy the baseline file into place; is the skywalker file produced by fortran code?
  configure_file(
    ${CONVPROC_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} convproc_driver ${CONVPROC_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # Conservative approach: At the time of this commit, threshold error=2e-6.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error> 
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True 2e-6)
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()
