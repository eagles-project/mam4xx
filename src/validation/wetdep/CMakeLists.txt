set(WETDEP_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/wetdep)
set(WETDEP_VALIDATION_TOOLS_DIR ${MAM_X_VALIDATION_DIR}/tools)

include_directories(${PROJECT_BINARY_DIR}/haero)
include_directories(${PROJECT_BINARY_DIR}/validation)

add_executable(wetdep_driver 
               wetdep_driver.cpp 
               wetdep_clddiag.cpp 
               update_scavenging.cpp 
               wetdep_prevap.cpp 
               wetdep_resusp_nonlinear.cpp
               wetdep_resusp_noprecip.cpp
               wetdep_scavenging.cpp
               compute_evap_frac.cpp
               rain_mix_ratio.cpp
               calculate_cloudy_volume.cpp
               local_precip_production.cpp
               wetdep_resusp.cpp
               wetdepa_v2.cpp
               compute_tendencies.cpp
              )
target_link_libraries(wetdep_driver skywalker;validation;${HAERO_LIBRARIES})

foreach (TEST_NAME
         clddiag
	 update_scavenging_0
	 update_scavenging_130
	 update_scavenging_230
	 wetdep_prevap_130
	 wetdep_prevap_230
	 wetdep_resusp_nonlinear_130
	 wetdep_resusp_nonlinear_230
	 wetdep_resusp_noprecip_130 
	 wetdep_resusp_noprecip_230 
	 wetdep_scavenging_true
	 wetdep_scavenging_false
	 compute_evap_frac
	 rain_mix_ratio
	 calculate_cloudy_volume
	 local_precip_production
	 wetdep_resusp_130
	 wetdep_resusp_230
	 wetdepa_v2_0
	 wetdepa_v2_130
	 wetdepa_v2_230
	 compute_tendencies
	)
  set(SKYWALKER_RESULT ${TEST_NAME}_output_ts_355)
  configure_file(
     ${WETDEP_VALIDATION_DIR}/${SKYWALKER_RESULT}.py
     ${CMAKE_CURRENT_BINARY_DIR}/${SKYWALKER_RESULT}_ref.py
     COPYONLY)

  add_test(wetdep_validation_${TEST_NAME} wetdep_driver
    ${WETDEP_VALIDATION_DIR}/${TEST_NAME}_input_ts_355.yaml)

  set(MAM4XX_WETDEP_RESULT mam4xx_${TEST_NAME}_input_ts_355)
  # Just use skywalker diff since cmake compare is too specific
  add_test(NAME wetdep_compare_${TEST_NAME}_output
    COMMAND python3 ${MAM_X_VALIDATION_DIR}/scripts/compare_mam4xx_mam4.py ${MAM4XX_WETDEP_RESULT} ${SKYWALKER_RESULT}_ref
  )
  set_tests_properties(wetdep_compare_${TEST_NAME}_output
    PROPERTIES DEPENDS wetdep_validation_{TEST_NAME})
endforeach()
