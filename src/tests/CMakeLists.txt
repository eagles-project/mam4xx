include(create_unit_test)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mam4_test_config.hpp.in
  ${CMAKE_CURRENT_SOURCE_DIR}/mam4_test_config.hpp
  @ONLY
)

# For some reason, ctest is looking for test-launcher in
# ${PROJECT_BINARY_DIR}/bin, so let's make sure it's there.
file(COPY ${MAM4XX_HAERO_DIR}/bin/test-launcher DESTINATION ${PROJECT_BINARY_DIR}/bin)

add_library(mam4xx_tests kohler_verification.cpp
                         atmosphere_utils.cpp)
target_include_directories(mam4xx_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mam4xx_tests PUBLIC ${HAERO_INCLUDE_DIRS})
target_link_libraries(mam4xx_tests ${HAERO_LIBRARIES})
target_link_libraries(mam4xx_tests mam4xx)

CreateUnitTest(utils_unit_tests utils_unit_tests.cpp)
CreateUnitTest(mam4_nucleation_unit_tests mam4_nucleation_unit_tests.cpp)
CreateUnitTest(mam4_gasaerexch_unit_tests mam4_gasaerexch_unit_tests.cpp)
CreateUnitTest(mam4_coagulation_unit_tests mam4_coagulation_unit_tests.cpp)
CreateUnitTest(mam4_calcsize_unit_tests mam4_calcsize_unit_tests.cpp)
CreateUnitTest(mam4_rename_unit_tests mam4_rename_unit_tests.cpp)
CreateUnitTest(mam4_aging_unit_tests mam4_aging_unit_tests.cpp)

target_compile_options(utils_unit_tests PRIVATE -Werror)
target_compile_options(mam4_nucleation_unit_tests PRIVATE -Werror)
target_compile_options(mam4_coagulation_unit_tests PRIVATE -Werror)
target_compile_options(mam4_gasaerexch_unit_tests PRIVATE -Werror)
target_compile_options(mam4_calcsize_unit_tests PRIVATE -Werror)
target_compile_options(mam4_nucleation_unit_tests PRIVATE -Werror)
target_compile_options(mam4_gasaerexch_unit_tests PRIVATE -Werror)
target_compile_options(mam4_calcsize_unit_tests PRIVATE -Werror)
target_compile_options(mam4_aging_unit_tests PRIVATE -Werror)

if (${HAERO_PRECISION} MATCHES double)
EkatCreateUnitTest(kohler_unit_tests kohler_unit_tests.cpp
  LIBS mam4xx mam4xx_tests ${HAERO_LIBRARIES})
else ()
  message(STATUS "disabling Kohler verification tests (they require double precision)")
endif ()
EkatCreateUnitTest(aero_modes_unit_tests aero_modes_tests.cpp
  LIBS mam4xx ${HAERO_LIBRARIES})

CreateUnitTest(aero_config_unit_tests aero_config_unit_tests.cpp)

# FIXME: Temporarily disabling the mode_averages tests for single-precision
# FIXME: builds so we can proceed with our un-Packed version of mam4xx.
if (${HAERO_PRECISION} MATCHES double)
  EkatCreateUnitTest(mode_averages mode_averages_unit_tests.cpp
    LIBS mam4xx mam4xx_tests ${HAERO_LIBRARIES})
endif()

EkatCreateUnitTest(conversions_unit_tests conversions_unit_tests.cpp
  LIBS mam4xx_tests ${HAERO_LIBRARIES})
