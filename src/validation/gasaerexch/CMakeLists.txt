set(EXCH_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/gasaerexch)

# Include directory for .mod files.
include_directories(${PROJECT_BINARY_DIR}/haero)
include_directories(${PROJECT_BINARY_DIR}/validation)

# These skywalker drivers compute gasaerexch update rates.
add_executable(gasaerexch_driver gasaerexch_driver.cpp
               gasaerexch_uptkrates_1box1gas.cpp)
target_link_libraries(gasaerexch_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy the gas-aerosol exchange plotting script to our binary directory for
# ease of use.
set(SKYWALKER_RESULT mam4xx_skywalker_gasaerexch_uptkrates_1box1gas)
configure_file(
   ${EXCH_VALIDATION_DIR}/${SKYWALKER_RESULT}.py
   ${CMAKE_CURRENT_BINARY_DIR}/${SKYWALKER_RESULT}_ref.py
   COPYONLY)

add_test(gasaerexch_uptkrates_1box1gas gasaerexch_driver
         ${EXCH_VALIDATION_DIR}/skywalker_gasaerexch_uptkrates_1box1gas.yaml)

if (${HAERO_PRECISION} MATCHES double)
  add_test(NAME gasaerexch_compare_output
    COMMAND ${CMAKE_COMMAND} -E compare_files ${SKYWALKER_RESULT}.py ${SKYWALKER_RESULT}_ref.py
  )
else ()
  add_test(NAME gasaerexch_compare_output
    COMMAND ${MAM_X_VALIDATION_DIR}/tools/skywalker_diff.py ${SKYWALKER_RESULT} ${SKYWALKER_RESULT}_ref
  )
endif ()

set_tests_properties(gasaerexch_compare_output
  PROPERTIES DEPENDS gasaerexch_uptkrates_1box1gas)

# Copy a Python tool from mam_x_validation to the test directory.
set(GASAER_SIZE_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/gasaerexch)
set(GASAER_SIZE_VALIDATION_TOOLS_DIR ${MAM_X_VALIDATION_DIR}/tools)
configure_file(
  ${GASAER_SIZE_VALIDATION_TOOLS_DIR}/skywalker_diff.py
  ${CMAKE_CURRENT_BINARY_DIR}/skywalker_diff.py
  COPYONLY
)

# These skywalker drivers test gasaerexch convergence rates
add_executable(exe_skywkr_gasaerexch_timestepping skywkr_gasaerexch_timestepping.cpp)
target_link_libraries(exe_skywkr_gasaerexch_timestepping skywalker;validation;${HAERO_LIBRARIES})

add_test(skywkr_soa_cond_timestepping_with_adptv_dt exe_skywkr_gasaerexch_timestepping
         ${EXCH_VALIDATION_DIR}/skywkr_soa_cond_timestepping_with_adptv_dt.yaml)

# copy the baseline file into place; is the skywalker file produced by fortran code?
configure_file(
  ${MAM_X_VALIDATION_DIR}/gasaerexch/mam_skywkr_soa_cond_timestepping_with_adptv_dt.py
  ${CMAKE_CURRENT_BINARY_DIR}/mam_skywkr_soa_cond_timestepping_with_adptv_dt.py
  COPYONLY
)
# add a test to validate mam4xx's results against the baseline.
add_test(validate_skywkr_soa_cond_timestepping_with_adptv_dt skywalker_diff.py -r 0.0001
  skywkr_soa_cond_timestepping_with_adptv_dt
  mam_skywkr_soa_cond_timestepping_with_adptv_dt
)
set_tests_properties(validate_skywkr_soa_cond_timestepping_with_adptv_dt PROPERTIES DEPENDS skywkr_soa_cond_timestepping_with_adptv_dt)

