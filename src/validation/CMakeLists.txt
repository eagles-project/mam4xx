# Here's a submodule that contains Skywalker inputs and Python scripts for
# processing outputs.
set(MAM_X_VALIDATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mam_x_validation)

# This library has some common utilities used by Skywalker drivers in this
# directory tree.
add_library(validation validation.cpp)
add_dependencies(validation ext_libraries)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Here's the root of the directory tree where the MAM4 box model writes its
# Skywalker output. Append a process's subdirectory to get the path for
# baselines for that process (e.g. ${MAM4BOX_BASELINE_DIR}/nucleation).
set(MAM4BOX_BASELINE_DIR ${PROJECT_BINARY_DIR}/mam4box/standalone/tests)

# Haero Skywalker drivers

add_subdirectory(nucleation)
add_subdirectory(gasaerexch)
