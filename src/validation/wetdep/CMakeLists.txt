set(WETDEP_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/wetdep)
set(WETDEP_VALIDATION_TOOLS_DIR ${MAM_X_VALIDATION_DIR}/tools)
# TODO - use EkatCreateUnitTest
# TODO - update mam_x_validation with wetdep data in git
# Include directory for .mod files.
include_directories(${PROJECT_BINARY_DIR}/haero)
include_directories(${PROJECT_BINARY_DIR}/validation)

# These skywalker drivers compute wetdep update rates.
add_executable(wetdep_driver wetdep_driver.cpp wetdep_clddiag.cpp)
target_link_libraries(wetdep_driver skywalker;validation;${HAERO_LIBRARIES})

# Copy the gas-aerosol wetdepange plotting script to our binary directory for
# ease of use.
set(SKYWALKER_RESULT clddiag_output_ts_355)
configure_file(
   ${WETDEP_VALIDATION_DIR}/${SKYWALKER_RESULT}.py
   ${CMAKE_CURRENT_BINARY_DIR}/${SKYWALKER_RESULT}_ref.py
   COPYONLY)

add_test(wetdep_validation wetdep_driver
         ${WETDEP_VALIDATION_DIR}/clddiag_input_ts_355.yaml)

if (${HAERO_PRECISION} MATCHES double)
  add_test(NAME wetdep_compare_output
    COMMAND ${CMAKE_COMMAND} -E compare_files ${SKYWALKER_RESULT}.py ${SKYWALKER_RESULT}_ref.py
  )
else ()
  add_test(NAME wetdep_compare_output
    COMMAND ${MAM_X_VALIDATION_DIR}/tools/skywalker_diff.py ${SKYWALKER_RESULT} ${SKYWALKER_RESULT}_ref
  )
endif ()

set_tests_properties(wetdep_compare_output
  PROPERTIES DEPENDS wetdep_validation)

# Copy a Python tool from mam_x_validation to the test directory.
configure_file(
  ${WETDEP_VALIDATION_TOOLS_DIR}/skywalker_diff.py
  ${CMAKE_CURRENT_BINARY_DIR}/skywalker_diff.py
  COPYONLY
)
