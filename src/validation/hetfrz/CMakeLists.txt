set(HETFRZ_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/hetfrz)
set(HETFRZ_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/hetfrz)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all aging-related parameterizations.
add_executable(hetfrz_driver
               hetfrz_driver.cpp
               hetfrz_get_reynolds_number.cpp
               hetfrz_get_temperature_diff.cpp
               hetfrz_get_latent_heat_vapor.cpp
               hetfrz_get_air_viscosity.cpp
               hetfrz_calculate_collkernel_sub.cpp
               hetfrz_collkernel.cpp
               hetfrz_get_Aimm.cpp
               hetfrz_get_dg0imm.cpp 
               hetfrz_get_form_factor.cpp
               hetfrz_calculate_hetfrz_contact_nucleation.cpp
               hetfrz_calculate_hetfrz_deposition_nucleation.cpp
               hetfrz_calculate_vars_for_pdf_imm.cpp
               hetfrz_calculate_hetfrz_immersion_nucleation.cpp
               hetfrz_calculate_rgimm_and_determine_spec_flag.cpp
               hetfrz_calculate_water_activity.cpp
               hetfrz_calculate_interstitial_aer_num.cpp
               hetfrz_calculate_cloudborne_aer_num.cpp
               hetfrz_get_aer_radius.cpp
               hetfrz_calculate_mass_mean_radius.cpp
               hetfrz_calculate_coated_fraction.cpp
               hetfrz_calculate_vars_for_water_activity.cpp
               hetfrz_rates_1box.cpp
               hetfrz_classnuc_calc.cpp)
target_link_libraries(hetfrz_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
 configure_file(
     ${HETFRZ_VALIDATION_SCRIPTS_DIR}/compare_hetfzr.py
     ${CMAKE_CURRENT_BINARY_DIR}/compare_hetfzr.py
     COPYONLY
   )

# Run the driver in several configurations to produce datasets.

foreach (input
          get_reynolds_num
          get_temperature_diff
          get_air_viscosity
          get_latent_heat_vapor
          calculate_collkernel_sub
          collkernel
          get_Aimm
          get_dg0imm
          get_form_factor
          calculate_hetfrz_contact_nucleation
          calculate_hetfrz_deposition_nucleation
          calculate_vars_for_pdf_imm
          calculate_hetfrz_immersion_nucleation
          calculate_water_activity
          calculate_rgimm_and_determine_spec_flag
          calculate_interstitial_aer_num
          calculate_cloudborne_aer_num
          get_aer_radius
          calculate_mass_mean_radius
          calculate_coated_fraction
          calculate_vars_for_water_activity
          hetfrz_rates_1box
          hetfrz_classnuc_calc
          )

  # copy the baseline file into place; is the skywalker file produced by fortran code?
  configure_file(
     ${HETFRZ_VALIDATION_DIR}/${input}.py
     ${CMAKE_CURRENT_BINARY_DIR}/${input}.py
     COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} hetfrz_driver ${HETFRZ_VALIDATION_DIR}/${input}.yaml)
  
  # add a test to validate mam4xx's results against the baseline.
  add_test(validate_${input} python3 compare_hetfzr.py mam4xx_${input}.py ${input}.py)
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
endforeach() 


