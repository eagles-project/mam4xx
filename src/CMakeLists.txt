# MAM4xx library and friends
add_subdirectory(mam4xx)

add_library(mam4xx aero_modes.cpp)

# MAM4 process unit tests.
add_subdirectory(tests)

# MAM4 cross validation with Fortran implementations
add_subdirectory(validation)
