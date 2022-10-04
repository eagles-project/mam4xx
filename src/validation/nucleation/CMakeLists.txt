set(NUC_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/nucleation)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.
include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all nucleation-related parameterizations.
add_executable(nucleation_driver nucleation_driver.cpp
               mer07_veh02_wang08_nuc_1box.cpp
               newnuc_cluster_growth.cpp)
target_link_libraries(nucleation_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_nucleation_rate.py
        compare_particle_growth.py
        generate_plots.py)
  configure_file(
    ${NUC_VALIDATION_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.

# Nucleation rate
foreach (input
         vehkamaki2002_fig8
         vehkamaki2002_fig9
         vehkamaki2002_fig10
         vehkamaki2002_fig11
         vehkamaki2002_contour
         merikanto2007_fig2
         merikanto2007_fig3
         merikanto2007_fig4)

  # copy the baseline file into place
  configure_file(
    ${NUC_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} nucleation_driver ${NUC_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  add_test(validate_${input} python3 compare_nucleation_rate.py mam4xx_${input}.py mam_${input}.py)
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()

# Nucleation particle cluster growth
foreach (input
         nucgrowth_case1
         nucgrowth_case2
         nucgrowth_case3)

  # copy the baseline file into place
  configure_file(
    ${NUC_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} nucleation_driver ${NUC_VALIDATION_DIR}/${input}.yaml)

  # add a test to validate mam4xx's results against the baseline.
  add_test(validate_${input} python3 compare_particle_growth.py mam4xx_${input}.py mam_${input}.py)
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach()

