# MAM4 process unit tests.
add_subdirectory(tests)

# MAM4 cross validation with MAM4's box model (via Skywalker drivers).
if (HAERO_ENABLE_MAM4BOX)
  add_subdirectory(validation)
endif()
