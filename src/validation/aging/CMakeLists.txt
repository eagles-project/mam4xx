set(CALCSIZE_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/calcsize)
set(CALCSIZE_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all calcsize-related parameterizations.
add_executable(aging_driver aging_driver.cpp
               mam_pcarbon_aging_frac.cpp
               mam_pcarbon_aging_1subarea.cpp
               transfer_aged_pcarbon_to_accum.cpp
               transfer_cond_coag_mass_to_accum.cpp)
target_link_libraries(aging_driver skywalker;validation;${HAERO_LIBRARIES})