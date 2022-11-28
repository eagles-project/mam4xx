# MAM4xx library and friends
add_subdirectory(mam4xx)

# MAM4 process unit tests.
add_subdirectory(tests)

# MAM4 cross validation with Fortran implementations (double precision only)
if (HAERO_PRECISION STREQUAL "double")
  add_subdirectory(validation)
endif()
