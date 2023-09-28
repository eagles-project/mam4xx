set(WATER_UPTAKE_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/water_uptake)
set(WATER_UPTAKE_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/water_uptake)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all water uptake-related parameterizations.
add_executable(water_uptake_driver
               water_uptake_driver.cpp
               water_uptake_wetdens.cpp
               find_real_solution.cpp
               makoh_quartic.cpp
               modal_aero_kohler.cpp
               modal_aero_water_uptake_rh_clearair.cpp
               modal_aero_water_uptake_wetaer.cpp
               modal_aero_water_uptake_dryaer.cpp
               modal_aero_water_uptake_dr.cpp
               modal_aero_water_uptake_dr_wetdens.cpp
               water_uptake_wetdens.cpp
               )
target_link_libraries(water_uptake_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
 configure_file(
     ${WATER_UPTAKE_VALIDATION_SCRIPTS_DIR}/compare_water_uptake.py
     ${CMAKE_CURRENT_BINARY_DIR}/compare_water_uptake.py
     COPYONLY
   )

# Run the driver in several configurations to produce datasets.

foreach (input
          find_real_solution
          makoh_quartic
          modal_aero_kohler
          modal_aero_water_uptake_rh_clearair
          modal_aero_water_uptake_wetaer
          modal_aero_water_uptake_dryaer
          modal_aero_water_uptake_dr
          modal_aero_water_uptake_dr_wetdens
          water_uptake_wetdens
          )

  # copy the baseline file into place; is the skywalker file produced by fortran code?
   configure_file(
     ${WATER_UPTAKE_VALIDATION_DIR}/${input}.py
     ${CMAKE_CURRENT_BINARY_DIR}/${input}.py
     COPYONLY
   )

  # add a test to run the skywalker driver
   add_test(run_${input} water_uptake_driver ${WATER_UPTAKE_VALIDATION_DIR}/${input}.yaml)
  
  # add a test to validate mam4xx's results against the baseline.
  add_test(validate_${input} python3 compare_water_uptake.py mam4xx_${input}.py ${input}.py)
endforeach() 