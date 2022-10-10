set(CALCSIZE_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/calcsize)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.
include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all nucleation-related parameterizations.
add_executable(calcsize_driver calcsize_driver.cpp
               get_relaxed_v2n_limits.cpp)
target_link_libraries(calcsize_driver skywalker;validation;${HAERO_LIBRARIES})


