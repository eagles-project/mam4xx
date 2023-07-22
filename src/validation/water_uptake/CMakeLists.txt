set(DRYDEP_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/water_uptake)
set(DRYDEP_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/water_uptake)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all drydep-related parameterizations.
add_executable(water_uptake_driver
               water_uptake_driver.cpp
               )
target_link_libraries(water_uptake_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
 configure_file(
     ${DRYDEP_VALIDATION_SCRIPTS_DIR}/compare_water_uptake.py
     ${CMAKE_CURRENT_BINARY_DIR}/compare_water_upatke.py
     COPYONLY
   )

# Run the driver in several configurations to produce datasets.

foreach (input

          )

  # copy the baseline file into place; is the skywalker file produced by fortran code?
   configure_file(
     ${WATER_UPTAKE_VALIDATION_DIR}/${input}.py
     ${CMAKE_CURRENT_BINARY_DIR}/${input}.py
     COPYONLY
   )

  # add a test to run the skywalker driver
   add_test(run_${input} drydep_driver ${WATER_UPTAKE_VALIDATION_DIR}/${input}.yaml)
  
  # add a test to validate mam4xx's results against the baseline.
  add_test(validate_${input} python3 compare_water_uptake.py mam4xx_${input}.py ${input}.py)
endforeach() 