set(AERO_EMISSIONS_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/aero_emissions)
set(AERO_EMISSIONS_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all aero_emissions-related parameterizations.
add_executable(aero_emissions_driver
               aero_emissions_driver.cpp
               aero_emissions_smoketest.cpp
               calc_om_seasalt.cpp
               calculate_seasalt_numflux_in_bins.cpp
               marine_organic_emis.cpp
               marine_organic_massflx_calc.cpp
               marine_organic_numflx_calc.cpp
               seasalt_emisflx_calc_massflx.cpp
               seasalt_emisflx_calc_numflx.cpp
               seasalt_emis.cpp
               )

target_link_libraries(aero_emissions_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${AERO_EMISSIONS_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    aero_emissions_smoketest
    calc_om_seasalt
    calculate_seasalt_numflux_in_bins
    marine_organic_emis
    marine_organic_massflx_calc
    marine_organic_numflx_calc
    seasalt_emisflx_calc_massflx
    seasalt_emisflx_calc_numflx
    seasalt_emis
    )

set(DEFAULT_TOL 1e-12)

set(ERROR_THRESHOLDS
    ${DEFAULT_TOL}
    3e-9
    8e-10
    ${DEFAULT_TOL}
    ${DEFAULT_TOL}
    ${DEFAULT_TOL}
    ${DEFAULT_TOL}
    ${DEFAULT_TOL}
    ${DEFAULT_TOL}
   )

set(TestLabels "aero_emissions")

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${AERO_EMISSIONS_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} aero_emissions_driver ${AERO_EMISSIONS_VALIDATION_DIR}/${input}.yaml)
  set_tests_properties(run_${input} PROPERTIES LABELS ${TestLabels})

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
  set_tests_properties(validate_${input} PROPERTIES LABELS ${TestLabels})
endforeach()
