include(EkatCreateUnitTest)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mam4_test_config.hpp.in
  ${PROJECT_BINARY_DIR}/include/mam4_test_config.hpp
  @ONLY
)

# For some reason, ctest is looking for test-launcher in
# ${PROJECT_BINARY_DIR}/bin, so let's make sure it's there.
file(COPY ${MAM4XX_HAERO_DIR}/bin/test-launcher DESTINATION ${PROJECT_BINARY_DIR}/bin
    FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_EXECUTE WORLD_READ)

add_library(mam4xx_tests kohler_verification.cpp
                         atmosphere_utils.cpp
                         testing.cpp)
target_include_directories(mam4xx_tests PUBLIC ${PROJECT_BINARY_DIR}/include)
target_include_directories(mam4xx_tests PUBLIC ${HAERO_INCLUDE_DIRS})
target_link_libraries(mam4xx_tests haero)
target_link_libraries(mam4xx_tests mam4xx)

#------------
# Unit Tests
#------------

# We use the EkatCreateUnitTest CMake function to create unit tests that are
# configured properly to work in the Kokkos environment provided by EKAT.
# EkatCreateUnitTest must be called with the following options:
# 1. LIBS mam4xx_tests haero <-- links against mam4xx and Haero
# 2. USER_DEFINED_TEST_SESSION                 <-- uses Haero's setup/breakdown functions

EkatCreateUnitTest(utils_unit_tests utils_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_nucleation_unit_tests mam4_nucleation_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_gasaerexch_unit_tests mam4_gasaerexch_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_coagulation_unit_tests mam4_coagulation_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_calcsize_unit_tests mam4_calcsize_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_convproc_unit_tests mam4_convproc_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_rename_unit_tests mam4_rename_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_aging_unit_tests mam4_aging_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_hetfrz_unit_tests mam4_hetfrz_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
# EkatCreateUnitTest(mam4_amicphys_1gridcell_tests mam4_amicphys_1gridcell.cpp
#   LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_nucleate_ice_unit_tests mam4_nucleate_ice_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_wet_deposition_unit_tests mam4_wet_deposition_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_dry_deposition_unit_tests mam4_dry_deposition_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_spitfire_transport_unit_tests mam4_spitfire_transport_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_mo_setsox_unit_tests mam4_mo_setsox_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_aero_emissions_unit_tests mam4_aero_emissions_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)

if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel") #The Intel compiler always emit a warning for inline functions.
  target_compile_options(utils_unit_tests PRIVATE)
  target_compile_options(mam4_nucleation_unit_tests PRIVATE)
  target_compile_options(mam4_gasaerexch_unit_tests PRIVATE)
  target_compile_options(mam4_coagulation_unit_tests PRIVATE)
  target_compile_options(mam4_calcsize_unit_tests PRIVATE)
  target_compile_options(mam4_convproc_unit_tests PRIVATE)
  # target_compile_options(mam4_amicphys_1gridcell_tests PRIVATE)
  target_compile_options(mam4_rename_unit_tests PRIVATE)
  target_compile_options(mam4_aging_unit_tests PRIVATE)
  target_compile_options(mam4_hetfrz_unit_tests PRIVATE)
  target_compile_options(mam4_nucleate_ice_unit_tests PRIVATE)
  target_compile_options(mam4_spitfire_transport_unit_tests PRIVATE)
  target_compile_options(mam4_mo_setsox_unit_tests PRIVATE)
  target_compile_options(mam4_aero_emissions_unit_tests PRIVATE)
endif ()

# if (${HAERO_PRECISION} MATCHES double)
# EkatCreateUnitTest(kohler_unit_tests kohler_unit_tests.cpp
#   LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
# else ()
#   message(STATUS "disabling Kohler verification tests (they require double precision)")
# endif ()

EkatCreateUnitTest(mam4_ndrop_unit_tests mam4_ndrop_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)

EkatCreateUnitTest(aero_modes_unit_tests aero_modes_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)

EkatCreateUnitTest(aero_config_unit_tests aero_config_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)

# FIXME: This test fails on single-precision builds.
if (${HAERO_PRECISION} MATCHES double)
  EkatCreateUnitTest(mode_averages mode_averages_unit_tests.cpp
    LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
endif()

EkatCreateUnitTest(conversions_unit_tests conversions_unit_tests.cpp
  LIBS mam4xx_tests haero USER_DEFINED_TEST_SESSION)
