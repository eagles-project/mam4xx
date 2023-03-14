set(NUCLEATE_ICE_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/nucleate_ice)
set(NUCLEATE_ICE_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all nucleate_ice-related parameterizations.
add_executable(nucleate_ice_driver  nucleate_ice_driver.cpp hf.cpp
               compute_tendencies.cpp nucleate_ice.cpp hetero.cpp)
target_link_libraries(nucleate_ice_driver skywalker;validation;${HAERO_LIBRARIES})


# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${NUCLEATE_ICE_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

set(TEST_LIST
    nucleate_ice_cam_calc_merged # passing tolerance = 1.5e-5
    hetero_merged
    #hetero_t_merged
    hf_merged
    nucleati_merged # passing tolerance = 3e-9
    nucleate_ice # produced by mam4_standalone_validation
    )
set(DEFAULT_TOL 1e-9)
set(ERROR_THRESHOLDS 1.5e-5 ${DEFAULT_TOL} ${DEFAULT_TOL} 3e-9 ${DEFAULT_TOL})

# Run the driver in several configurations to produce datasets.

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place; is the skywalker file produced by fortran code?
  configure_file(
    ${NUCLEATE_ICE_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} nucleate_ice_driver ${NUCLEATE_ICE_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()
