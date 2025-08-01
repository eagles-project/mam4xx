set(AEROSOL_OPTICS_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/aerosol_optics)
set(AEROSOL_OPTICS_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all aerosol_optics-related parameterizations.

add_executable(aerosol_optics_driver
               aerosol_optics_driver.cpp
               binterp.cpp
               calc_diag_spec.cpp
               calc_refin_complex.cpp
               calc_volc_ext.cpp
               modal_size_parameters.cpp
               modal_aero_sw.cpp
               modal_aero_lw.cpp
               calc_parameterized.cpp
               update_aod_spec.cpp
               aer_rad_props_lw.cpp
               aer_rad_props_sw.cpp
               volcanic_cmip_sw.cpp
               data_transfer_state_q_qqwc_to_prog.cpp
               )

target_link_libraries(aerosol_optics_driver skywalker;validation;haero)

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${AEROSOL_OPTICS_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()
# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    binterp_ts_355
    calc_diag_spec_ts_355
    calc_refin_complex_ts_355_lw
    calc_refin_complex_ts_355_sw
    calc_volc_ext_ts_355
    modal_size_parameters_ts_355_ismethod2_false
    modal_size_parameters_ts_355_ismethod2_true
    modal_aero_sw_ts_355
    modal_aero_lw_ts_355
    calc_parameterized_ts_355
    update_aod_spec_ts_355
    aer_rad_props_lw_ts_355
    aer_rad_props_sw_ts_355
    volcanic_cmip_sw_ts_355
    data_transfer_state_q_qqwc_to_prog
    )
# # matching the tests and errors, just for convenience

set(DEFAULT_TOL 1e-13)
set(ERROR_THRESHOLDS
   2e-10
   ${DEFAULT_TOL}
   3e-10
   3e-10
   3e-10
   2e-8
   2e-8
   6e-9
   7e-11
   2e-11
   1e-12
   7e-11
   8e-11
   8e-11
   1e-14
   )
foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${AEROSOL_OPTICS_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} aerosol_optics_driver ${AEROSOL_OPTICS_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()
