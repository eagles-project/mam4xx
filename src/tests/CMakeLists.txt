include(create_unit_test)

# For some reason, ctest is looking for test-launcher in
# ${PROJECT_BINARY_DIR}/bin, so let's make sure it's there.
file(COPY ${MAM4XX_HAERO_DIR}/bin/test-launcher DESTINATION ${PROJECT_BINARY_DIR}/bin)

CreateUnitTest(mam4_nucleation_unit_tests mam4_nucleation_unit_tests.cpp)
CreateUnitTest(mam4_gasaerexch_unit_tests mam4_gasaerexch_unit_tests.cpp)
target_compile_options(mam4_nucleation_unit_tests PRIVATE -Werror)
target_compile_options(mam4_gasaerexch_unit_tests PRIVATE -Werror)
