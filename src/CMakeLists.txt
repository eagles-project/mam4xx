# MAM4xx library and friends
add_subdirectory(mam4xx)

# MAM4 process unit tests.
add_subdirectory(tests)

# MAM4 cross validation with Fortran implementations (double precision only)
if (ENABLE_SKYWALKER AND HAERO_PRECISION STREQUAL "double")
  add_subdirectory(validation)
endif()

# Generate mam4xx.cmake for easy incorporation into larger projects
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/mam4xx.cmake.in
  ${PROJECT_BINARY_DIR}/share/mam4xx.cmake
  @ONLY
)
install(DIRECTORY ${PROJECT_BINARY_DIR}/share DESTINATION .)
