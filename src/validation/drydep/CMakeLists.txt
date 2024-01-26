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
               modal_aero_turb_drydep_velocity.cpp
               modal_aero_depvel_part.cpp
               aero_model_drydep.cpp
               compute_tendencies.cpp
               modal_aero_gravit_settling_velocity.cpp
               calcram.cpp
               )
target_link_libraries(drydep_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy some Python scripts from mam_x_validation to our binary directory.
 configure_file(
     ${DRYDEP_VALIDATION_SCRIPTS_DIR}/compare_drydep.py
     ${CMAKE_CURRENT_BINARY_DIR}/compare_drydep.py
     COPYONLY
   )

# Run the driver in several configurations to produce datasets.
function(add_drydep_test input_name)
  string(REPLACE "_input_" "_output_"  output_name ${input_name})
  string(REPLACE "_input_" "_"  test_name ${input_name})
  # copy the baseline file into place; is the skywalker file produced by fortran code?
  configure_file(${DRYDEP_VALIDATION_DIR}/${output_name}.py ${CMAKE_CURRENT_BINARY_DIR}/${output_name}.py COPYONLY)
  # add a test to run the skywalker driver
  add_test(run_${test_name} drydep_driver ${DRYDEP_VALIDATION_DIR}/${input_name}.yaml)
  # add a test to validate mam4xx's results against the baseline.
  add_test(validate_${test_name} python3 compare_drydep.py mam4xx_${input_name}.py ${output_name}.py)
  set_tests_properties(validate_${test_name} PROPERTIES DEPENDS run_${test_name})
endfunction()

foreach (input
          gravit_settling_velocity
          air_dynamic_viscosity
          air_kinematic_viscosity
          radius_for_moment
          schmidt_number
          slip_correction_factor
          )
  add_drydep_test(${input})
endforeach() 


foreach (test_name
    turb_drydep_velocity
    gravit_settling_velocity
    depvel_part
    )
    foreach (ts
	#1381 1382 1383 1384 1385 1386 1387 1388 1389 1390 1391 1392 
       1393 #1394 1395 1396 1397 1398 1399 1400 1401 1402 1403 1404 
       #1405 1406 1407 1408 1409 1410 1411 1412 1413 1414 1415 1416
        )
        foreach (jvlc 3 4)
	    math(EXPR imnt "3 * ( ${jvlc} - 3)")
            # Special case of imode is No_mode
            set(imode No_mode)
            set(input modal_aero_${test_name}_input_ts_${ts}_jvlc_${jvlc}_imnt_${imnt}_imode_${imode})
	    add_drydep_test(${input})
        endforeach() 

        foreach (jvlc 1 2)
	    math(EXPR imnt "3 * ( ${jvlc} - 1)")
            foreach (imode 1 2 3 4)
                set(input modal_aero_${test_name}_input_ts_${ts}_jvlc_${jvlc}_imnt_${imnt}_imode_${imode})
	        add_drydep_test(${input})
            endforeach() 
        endforeach() 
    endforeach() 
endforeach() 

foreach (ts 1400 1401 1402 1403)# 1404 1405 1406 1407 1408 1409 1410 1411 1412 1413 1414 1415 1416)
    set(input calcram_input_ts_${ts})
    add_drydep_test(${input})
endforeach() 

foreach (ts
    1381 1382 1383 1384 1385 1386 1387 1388 1389 1390 1391 1392 
#   1393 1394 1395 1396 1397 1398 1399 1400 1401 1402 1403 1404 
#   1405 1406 1407 1408 1409 1410 1411 1412 1413 1414 1415 1416
          )
    set(test_name drydep)
    set(input aero_model_${test_name}_input_ts_${ts})
    add_drydep_test(${input})
endforeach() 

set(test_name drydep)
set(ts compute_tendencies)
set(input aero_model_${test_name}_input_${ts})
add_drydep_test(${input})


