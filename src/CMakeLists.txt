
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mam4.hpp.in
  ${CMAKE_CURRENT_SOURCE_DIR}/mam4.hpp
  @ONLY
)

# MAM4 support library (host/cpu functions only)
add_library(mam4xx
  kohler.cpp)

# MAM4 process unit tests.
add_subdirectory(tests)

# MAM4 cross validation with MAM4's box model (via Skywalker drivers).
if (MAM4XX_ENABLE_BOX)
  add_subdirectory(validation)
endif()
