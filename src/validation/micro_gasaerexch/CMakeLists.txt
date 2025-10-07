set(MICRO_GASAEREXCH_DIR ${MAM_X_VALIDATION_DIR}/modal_aero_amicphys)
set(MICRO_GASAEREXCH_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)
# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all micro_gasaerexch-related parameterizations.
add_executable(micro_gasaerexch_driver
               micro_gasaerexch_driver.cpp
               mam_soaexch_1subarea.cpp
               gas_aer_uptkrates_1box1gas.cpp
               mam_gasaerexch_1subarea.cpp
               )

target_link_libraries(micro_gasaerexch_driver skywalker;validation;haero)

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${MICRO_GASAEREXCH_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()
# Run the driver in several configurations to produce datasets.

set(TEST_LIST
    mam_soaexch_1subarea_ts_379
    gas_aer_uptkrates_1box1gas_ts_379
    mam_gasaerexch_1subarea_ts_379
    )
# # matching the tests and errors, just for convenience

set(ERROR_THRESHOLDS
    2e-10
    1e-13
    4e-10
   )

set(TestLabel "micro_gasaerexch")

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${MICRO_GASAEREXCH_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} micro_gasaerexch_driver ${MICRO_GASAEREXCH_DIR}/${input}.yaml)
  set_tests_properties(run_${input} PROPERTIES LABELS ${TestLabel})

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_property(TEST validate_${input} PROPERTY SKIP_REGULAR_EXPRESSION "regex_fail_rel_tol")
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
  set_tests_properties(validate_${input} PROPERTIES LABELS ${TestLabel})

endforeach()
