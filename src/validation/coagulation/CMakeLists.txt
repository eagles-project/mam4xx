set(COAGULATION_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/coagulation)
set(COAGULATION_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/coagulation)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all aging-related parameterizations.
add_executable(coagulation_driver
               coagulation_driver.cpp
               coag_1subarea.cpp
               getcoags.cpp
               getcoags_wrapper_f.cpp
               coag_aer_update.cpp 
               coag_num_update.cpp )
target_link_libraries(coagulation_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_coag_validation.py)
  configure_file(
    ${COAGULATION_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.

# Aging
foreach (input
         coag_1subarea
         getcoags
         getcoags_wrapper_f
         coag_aer_update
         coag_num_update
         )

  # copy the baseline file into place; is the skywalker file produced by fortran code?
  configure_file(
    ${COAGULATION_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} coagulation_driver ${COAGULATION_VALIDATION_DIR}/${input}.yaml)
  
  # add a test to validate mam4xx's results against the baseline.
  add_test(validate_${input} python3 compare_coag_validation.py mam4xx_${input}.py mam_${input}.py)
  endforeach() 


