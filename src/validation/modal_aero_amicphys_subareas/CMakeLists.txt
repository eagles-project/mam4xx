set(AMICPHYS_SUBAREAS_DRIVER_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/modal_aero_amicphys_subareas)
set(AMICPHYS_SUBAREAS_DRIVER_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all amicphys_subareas_driver-related parameterizations.
add_executable(amicphys_subareas_driver
               amicphys_subareas_driver.cpp
               compute_qsub_from_gcm_and_qsub_of_other_subarea.cpp
               form_gcm_of_gases_and_aerosols_from_subareas.cpp
               get_partition_factors.cpp
               set_subarea_gases_and_aerosols.cpp
               set_subarea_qmass_for_cldbrn_aerosols.cpp
               set_subarea_qmass_for_intrst_aerosols.cpp
               set_subarea_qnumb_for_cldbrn_aerosols.cpp
               set_subarea_qnumb_for_intrst_aerosols.cpp
               set_subarea_rh.cpp
               setup_subareas.cpp
)

target_link_libraries(amicphys_subareas_driver skywalker;validation;haero)

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${AMICPHYS_SUBAREAS_DRIVER_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    compute_qsub_from_gcm_and_qsub_of_other_subarea
    form_gcm_of_gases_and_aerosols_from_subareas
    get_partition_factors
    set_subarea_gases_and_aerosols
    set_subarea_qmass_for_cldbrn_aerosols
    set_subarea_qmass_for_intrst_aerosols
    set_subarea_qnumb_for_cldbrn_aerosols
    set_subarea_qnumb_for_intrst_aerosols
    set_subarea_rh
    setup_subareas
)

set(DEFAULT_TOL 2e-10)

set(ERROR_THRESHOLDS
    ${DEFAULT_TOL} # compute_qsub_from_gcm_and_qsub_of_other_subarea
    ${DEFAULT_TOL} # form_gcm_of_gases_and_aerosols_from_subareas
    1e-14 # get_partition_factors
    3e-10 # set_subarea_gases_and_aerosols
    ${DEFAULT_TOL} # set_subarea_qmass_for_cldbrn_aerosols
    5e-10 # set_subarea_qmass_for_intrst_aerosols
    1e-14 # set_subarea_qnumb_for_cldbrn_aerosols
    5e-10 # set_subarea_qnumb_for_intrst_aerosols
    ${DEFAULT_TOL} # set_subarea_rh
    ${DEFAULT_TOL} # setup_subareas
)

set(TestLabel "amicphys_1subarea")

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.
  configure_file(
    ${AMICPHYS_SUBAREAS_DRIVER_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} amicphys_subareas_driver ${AMICPHYS_SUBAREAS_DRIVER_VALIDATION_DIR}/${input}.yaml)
  set_tests_properties(run_${input} PROPERTIES LABELS ${TestLabel})

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_property(TEST validate_${input} PROPERTY SKIP_REGULAR_EXPRESSION "regex_fail_rel_tol")
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
  set_tests_properties(validate_${input} PROPERTIES LABELS ${TestLabel})
endforeach()
