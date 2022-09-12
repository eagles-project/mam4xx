# MAM4 process unit tests.
add_subdirectory(tests)

# MAM4 cross validation with MAM4's box model (via Skywalker drivers).
if (MAM4XX_ENABLE_BOX)
  add_subdirectory(validation)
endif()
