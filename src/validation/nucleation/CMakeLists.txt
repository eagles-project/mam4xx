set(NUC_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/nucleation)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.
include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all nucleation-related parameterizations.
add_executable(nucleation_driver nucleation_driver.cpp
               mer07_veh02_wang08_nuc_1box.cpp
               newnuc_cluster_growth.cpp
               binary_nuc_vehk2002.cpp
               pbl_nuc_wang2008.cpp
               mer07_veh02_nuc_mosaic_1box.cpp)
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

# additional tests added for integration
set(TestLabel "nuc_tests_new")

set(NUCLEATION_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${NUCLEATION_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    binary_nuc_vehk2002
    pbl_nuc_wang2008
    mer07_veh02_nuc_mosaic_1box
    )

set(DEFAULT_TOL 2e-10)

set(ERROR_THRESHOLDS
    ${DEFAULT_TOL} # binary_nuc_vehk2002
    ${DEFAULT_TOL} # pbl_nuc_wang2008
    ${DEFAULT_TOL} # mer07_veh02_nuc_mosaic_1box
   )

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.
  configure_file(
    ${NUC_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} nucleation_driver ${NUC_VALIDATION_DIR}/${input}.yaml)
  set_tests_properties(run_${input} PROPERTIES LABELS "${TestLabel}")

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
  set_tests_properties(validate_${input} PROPERTIES LABELS ${TestLabel})
endforeach()
