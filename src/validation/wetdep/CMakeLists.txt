set(WETDEP_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/wetdep)
set(WETDEP_VALIDATION_TOOLS_DIR ${MAM_X_VALIDATION_DIR}/tools)

include_directories(${PROJECT_BINARY_DIR}/haero)
include_directories(${PROJECT_BINARY_DIR}/validation)

add_executable(wetdep_driver wetdep_driver.cpp wetdep_clddiag.cpp)
target_link_libraries(wetdep_driver skywalker;validation;${HAERO_LIBRARIES})

set(SKYWALKER_RESULT clddiag_output_ts_355)
configure_file(
   ${WETDEP_VALIDATION_DIR}/${SKYWALKER_RESULT}.py
   ${CMAKE_CURRENT_BINARY_DIR}/${SKYWALKER_RESULT}_ref.py
   COPYONLY)

add_test(wetdep_validation wetdep_driver
         ${WETDEP_VALIDATION_DIR}/clddiag_input_ts_355.yaml)

# Just use skywalker diff since cmake compare is too specific
add_test(NAME wetdep_compare_output
  COMMAND ${MAM_X_VALIDATION_DIR}/tools/skywalker_diff.py ${SKYWALKER_RESULT} ${SKYWALKER_RESULT}_ref
)

set_tests_properties(wetdep_compare_output
  PROPERTIES DEPENDS wetdep_validation)

# Copy a Python tool from mam_x_validation to the test directory.
configure_file(
  ${WETDEP_VALIDATION_TOOLS_DIR}/skywalker_diff.py
  ${CMAKE_CURRENT_BINARY_DIR}/skywalker_diff.py
  COPYONLY
)
