include(create_unit_test)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mam4_test_config.hpp.in
  ${CMAKE_CURRENT_SOURCE_DIR}/mam4_test_config.hpp
  @ONLY
)

# For some reason, ctest is looking for test-launcher in
# ${PROJECT_BINARY_DIR}/bin, so let's make sure it's there.
file(COPY ${MAM4XX_HAERO_DIR}/bin/test-launcher DESTINATION ${PROJECT_BINARY_DIR}/bin)

add_library(mam4xx_tests kohler_verification.cpp)
target_include_directories(mam4xx_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mam4xx_tests ${HAERO_LIBRARIES})

CreateUnitTest(mam4_nucleation_unit_tests mam4_nucleation_unit_tests.cpp)
CreateUnitTest(mam4_gasaerexch_unit_tests mam4_gasaerexch_unit_tests.cpp)
target_compile_options(mam4_nucleation_unit_tests PRIVATE -Werror)
target_compile_options(mam4_gasaerexch_unit_tests PRIVATE -Werror)

if (${HAERO_PRECISION} MATCHES double)
EkatCreateUnitTest(mam4_kohler_unit_tests mam4_kohler_unit_tests.cpp
  LIBS mam4xx_tests ${HAERO_LIBRARIES})
else ()
  message(STATUS "disabling Kohler verification tests (they require double precision)")
endif ()
CreateUnitTest(aero_modes_unit_tests aero_modes_tests.cpp)

CreateUnitTest(aero_config_unit_tests aero_config_unit_tests.cpp)
