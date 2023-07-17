set(DRYDEP_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/drydep)
set(DRYDEP_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/drydep)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all drydep-related parameterizations.
add_executable(drydep_driver
               drydep_driver.cpp
               air_dynamic_viscosity.cpp
               air_kinematic_viscosity.cpp
               gravit_settling_velocity.cpp
               radius_for_moment.cpp
               schmidt_number.cpp
               slip_correction_factor.cpp
               )
target_link_libraries(drydep_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
 configure_file(
     ${DRYDEP_VALIDATION_SCRIPTS_DIR}/compare_drydep.py
     ${CMAKE_CURRENT_BINARY_DIR}/compare_drydep.py
     COPYONLY
   )

# Run the driver in several configurations to produce datasets.

foreach (input
          gravit_settling_velocity
          air_dynamic_viscosity
          air_kinematic_viscosity
          radius_for_moment
          schmidt_number
          slip_correction_factor
          )

  # copy the baseline file into place; is the skywalker file produced by fortran code?
   configure_file(
     ${DRYDEP_VALIDATION_DIR}/${input}.py
     ${CMAKE_CURRENT_BINARY_DIR}/${input}.py
     COPYONLY
   )

  # add a test to run the skywalker driver
   add_test(run_${input} drydep_driver ${DRYDEP_VALIDATION_DIR}/${input}.yaml)
  
  # add a test to validate mam4xx's results against the baseline.
  add_test(validate_${input} python3 compare_drydep.py mam4xx_${input}.py ${input}.py)
endforeach() 


