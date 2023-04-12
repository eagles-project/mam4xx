set(HETFRZ_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/hetfrz)
set(HETFRZ_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/hetfrz)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all aging-related parameterizations.
add_executable(hetfrz_driver
               hetfrz_driver.cpp
               hetfrz_get_reynolds_number.cpp)
target_link_libraries(hetfrz_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
 foreach(script
         compare_get_reynolds_num.py)
   configure_file(
     ${HETFRZ_VALIDATION_SCRIPTS_DIR}/${script}
     ${CMAKE_CURRENT_BINARY_DIR}/${script}
     COPYONLY
   )
 endforeach()

# Run the driver in several configurations to produce datasets.

# Aging
foreach (input
          get_reynolds_num
          )

  # copy the baseline file into place; is the skywalker file produced by fortran code?
   configure_file(
     ${HETFRZ_VALIDATION_DIR}/${input}.py
     ${CMAKE_CURRENT_BINARY_DIR}/${input}.py
     COPYONLY
   )

  # add a test to run the skywalker driver
   add_test(run_${input} hetfzr_driver ${HETFRZ_VALIDATION_DIR}/${input}.yaml)
  
  # add a test to validate mam4xx's results against the baseline.
  add_test(validate_${input} python3 compare_get_reynolds_num.py mam4xx_${input}.py ${input}.py)
endforeach() 


