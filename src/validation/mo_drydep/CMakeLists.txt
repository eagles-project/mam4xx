set(MO_DRYDEP_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/mo_drydep)
set(MO_DRYDEP_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all mo_drydep-related parameterizations.
add_executable(mo_drydep_driver
               mo_drydep_driver.cpp
               calculate_aerodynamic_and_quasilaminar_resistance.cpp
               calculate_gas_drydep_vlc_and_flux.cpp
               calculate_obukhov_length.cpp
               calculate_resistance_rclx.cpp
               calculate_resistance_rgsx_and_rsmx.cpp
               calculate_resistance_rlux.cpp
               calculate_ustar_over_water.cpp
               calculate_ustar.cpp
               calculate_uustar.cpp
               drydep_xactive.cpp
               )

target_link_libraries(mo_drydep_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${MO_DRYDEP_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    calculate_aerodynamic_and_quasilaminar_resistance
    calculate_gas_drydep_vlc_and_flux
    calculate_obukhov_length
    calculate_resistance_rclx
    calculate_resistance_rgsx_and_rsmx
    calculate_resistance_rlux
    calculate_ustar_over_water
    calculate_ustar
    calculate_uustar
    drydep_xactive
    )

set(DEFAULT_TOL 1e-11)

set(ERROR_THRESHOLDS
    ${DEFAULT_TOL} # calculate_aerodynamic_and_quasilaminar_resistance
    ${DEFAULT_TOL} # calculate_gas_drydep_vlc_and_flux
    ${DEFAULT_TOL} # calculate_resistance_rclx
    ${DEFAULT_TOL} # calculate_resistance_rgsx_and_rsmx
    ${DEFAULT_TOL} # calculate_resistance_rlux
    ${DEFAULT_TOL} # calculate_ustar_over_water
    ${DEFAULT_TOL} # calculate_ustar
    ${DEFAULT_TOL} # calculate_uustar
    ${DEFAULT_TOL} # drydep_xactive
   )

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.
  configure_file(
    ${MO_DRYDEP_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} mo_drydep_driver ${MO_DRYDEP_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()
