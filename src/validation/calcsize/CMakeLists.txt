set(CALCSIZE_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/calcsize)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all calcsize-related parameterizations.
add_executable(calcsize_driver calcsize_driver.cpp
               get_relaxed_v2n_limits.cpp
               compute_dry_volume.cpp
               adjust_num_sizes.cpp
               update_diameter_and_vol2num.cpp
               compute_tendencies.cpp)
target_link_libraries(calcsize_driver skywalker;validation;${HAERO_LIBRARIES})


# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_calcsize.py)
  configure_file(
    ${CALCSIZE_VALIDATION_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.

# Nucleation rate
foreach (input
         get_relaxed_v2n_limits_case1
         get_relaxed_v2n_limits_case2
         adjust_num_sizes_case1
         adjust_num_sizes_case2
         update_diameter_and_vol2num_case1
         calcsize_e3sm
         compute_dry_volume_case1)

  # copy the baseline file into place; is the skywalker file produced by fortran code?
  configure_file(
    ${CALCSIZE_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} calcsize_driver ${CALCSIZE_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  add_test(validate_${input} python3 compare_calcsize.py mam4xx_${input}.py mam_${input}.py)
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()
