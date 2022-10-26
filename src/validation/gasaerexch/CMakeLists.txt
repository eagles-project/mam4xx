set(EXCH_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/gasaerexch)

# Include directory for .mod files.
include_directories(${PROJECT_BINARY_DIR}/haero)
include_directories(${PROJECT_BINARY_DIR}/validation)

# These skywalker drivers compute gasaerexch update rates.
add_executable(gasaerexch_driver gasaerexch_driver.cpp
               gasaerexch_uptkrates_1box1gas.cpp)
target_link_libraries(gasaerexch_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy the gas-aerosol exchange plotting ѕcript to our binary directory for
# ease of use.
set(SKYWALKER_RESULT mam4xx_skywalker_gasaerexch_uptkrates_1box1gas.py)
configure_file(
   ${EXCH_VALIDATION_DIR}/${SKYWALKER_RESULT}
   ${CMAKE_CURRENT_BINARY_DIR}/${SKYWALKER_RESULT}.ref
   COPYONLY)

add_test(gasaerexch_uptkrates_1box1gas gasaerexch_driver
         ${EXCH_VALIDATION_DIR}/skywalker_gasaerexch_uptkrates_1box1gas.yaml)

add_test(NAME gasaerexch_compare_output
  COMMAND ${CMAKE_COMMAND} -E compare_files ${SKYWALKER_RESULT} ${SKYWALKER_RESULT}.ref
)

set_tests_properties(gasaerexch_compare_output
  PROPERTIES DEPENDS gasaerexch_uptkrates_1box1gas)

