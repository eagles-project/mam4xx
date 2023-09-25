set(LIN_STRAT_CHEM_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/lin_strat_chem)
set(LIN_STRAT_CHEM_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all lin_strat_chem-related parameterizations.
add_executable(lin_strat_chem_driver
               lin_strat_chem_driver.cpp
               lin_strat_chem_solve.cpp
               lin_strat_chem_solve_multicol.cpp
               lin_strat_sfcsink.cpp
               )

target_link_libraries(lin_strat_chem_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${LIN_STRAT_CHEM_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()
# stand_calc_sum_wght_ts_355
# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    lin_strat_chem_solve_ts_1415
    lin_strat_sfcsink_ts_1415_multicol
    lin_strat_sfcsinkmulticol_merged
    lin_strat_chem_solve_merged
    lin_strat_chem_solve_ts_1415_multicol
    lin_strat_chem_solvemulticol_merged
    )
# # matching the tests and errors, just for convenience

set(DEFAULT_TOL 1e-13)
set(ERROR_THRESHOLDS
    6e-11
    ${DEFAULT_TOL}
    6e-13
    2e-9
    6e-9
    9e-7
   )
foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${LIN_STRAT_CHEM_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} lin_strat_chem_driver ${LIN_STRAT_CHEM_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()
