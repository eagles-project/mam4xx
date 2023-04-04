set(RENAME_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/rename)
set(RENAME_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all rename-related parameterizations.
add_executable(rename_driver  rename_driver.cpp
               mam_rename_1subarea.cpp
               find_renaming_pairs.cpp
               compute_dryvol_change_in_src_mode.cpp)
target_link_libraries(rename_driver skywalker;validation;${HAERO_LIBRARIES})


# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${RENAME_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    rename_1subarea_ts_1111_iscldy_false
    rename_1subarea_ts_1111_iscldy_true
    find_renaming_pairs_ts_1111
    compute_dryvol_change_in_src_mode_ts_1111
    rename_1subarea_epsilon
    )

set(DEFAULT_TOL 1e-11)
# rename

set(ERROR_THRESHOLDS 
   7e-11
   7e-10
   2e-8
   ${DEFAULT_TOL} 
   ${DEFAULT_TOL})

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.
  
  configure_file(
    ${RENAME_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} rename_driver ${RENAME_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})

endforeach()
