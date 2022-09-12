cmake_minimum_required (VERSION 3.12.0)

# Do we have make?
find_program(MAKE make)
if (MAKE STREQUAL "MAKE_NOTFOUND")
  message(FATAL_ERROR "Make is required, but is not available on this system.")
endif()

#-----------------------------------------
# Options (most are inherited from Haero)
#-----------------------------------------
option(MAM4XX_ENABLE_BOX "Enable MAM4 box model cross-validation"     OFF)

message(STATUS "Configuring with build type: ${CMAKE_BUILD_TYPE}")

# We rely on Haero for basic data structures and third-party libraries.
if (NOT MAM4XX_HAERO_DIR)
  message(FATAL_ERROR "MAM4XX_HAERO_DIR is not specified! Please provide a path to Haero.")
elseif (NOT EXISTS ${MAM4XX_HAERO_DIR}/share)
  message(FATAL_ERROR "Invalid MAM4XX_HAERO_DIR. Please provide a valid path to Haero.")
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${MAM4XX_HAERO_DIR}/share/")
include(haero)

if(HAERO_ENABLE_GPU)
  message(STATUS "Building for GPU (${HAERO_DEVICE_ARCH})")
else() #CPU
  message(STATUS "Building for CPU (${HAERO_DEVICE_ARCH})")
endif()

# Set compilers, linkers, and flags from Haero.
set(CMAKE_CXX_STANDARD ${HAERO_CXX_STANDARD})
set(CMAKE_CXX_COMPILER ${HAERO_CXX_COMPILER})
set(CMAKE_CXX_FLAGS    ${HAERO_CXX_FLAGS})

set(CMAKE_C_STANDARD ${HAERO_C_STANDARD})
set(CMAKE_C_COMPILER ${HAERO_C_COMPILER})
set(CMAKE_C_FLAGS    ${HAERO_C_FLAGS})

enable_language(CXX)
enable_language(Fortran)

project (mam4xx)

set(CMAKE_EXE_LINKER_FLAGS ${HAERO_EXE_LINKER_FLAGS})

message(STATUS "Generating project files in build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "C++ compiler is ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
message(STATUS "C compiler is ${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION})")
message(STATUS "Fortran compiler is ${CMAKE_Fortran_COMPILER} (${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION})")

# Version numbers.
set(MAM4XX_MAJOR_VERSION 0)
set(MAM4XX_MINOR_VERSION 1)
set(MAM4XX_PATCH_VERSION 0)

set(MAM4XX_VERSION "${MAM4XX_MAJOR_VERSION}.${MAM4XX_MINOR_VERSION}.${MAM4XX_PATCH_VERSION}")
message(STATUS "Configuring MAM4xx v${MAM4XX_VERSION}")

# Precision of floating point numbers.
message(STATUS "Using ${HAERO_PRECISION} precision floating point numbers")

# We build static libraries only.
set(BUILD_SHARED_LIBS OFF)

# Fortran setting and compiler flags
set(CMAKE_Fortran_FLAGS "-cpp")
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fbacktrace")
  endif()
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall -ffree-form -ffree-line-length-none")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wno-unused-dummy-argument")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Werror=tabs")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wno-maybe-uninitialized")
elseif (CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -traceback")
  endif()
endif()
if (APPLE)
  set(CMAKE_Fortran_STANDARD_LIBRARIES "-lc++")
else()
  set(CMAKE_Fortran_STANDARD_LIBRARIES "-lstdc++ -no-pie")
endif()

# Report the installation prefix.
message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

set(MAM4XX_LIBRARIES ${HAERO_LIBRARIES})

# We look for headers in these directories.
include_directories(${HAERO_INCLUDE_DIRS})
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${MAM4XX_HAERO_DIR}/include")
include_directories(${PROJECT_SOURCE_DIR}/src)

link_directories("${MAM4XX_HAERO_DIR}/lib")

# Support for valgrind -- Linux only.
if (UNIX AND NOT APPLE)
  find_package(Valgrind QUIET)
  if (VALGRIND_FOUND)
    set(VALGRIND_FOUND 1) # regularize this value
    include_directories(${VALGRIND_INCLUDE_DIR})
    set(MEMORYCHECK_COMMAND ${VALGRIND_PROGRAM})
    # Add "--gen-suppressions=all" to MEMORYCHECK_COMMAND_OPTIONS to generate
    # suppressions for Valgrind's false positives. The suppressions show up
    # right in the MemoryChecker.*.log files.
    set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --show-leak-kinds=all --errors-for-leak-kinds=definite,possible --track-origins=yes --error-exitcode=1 --trace-children=yes --suppressions=${PROJECT_SOURCE_DIR}/tools/valgrind/scasm.supp" CACHE STRING "Options passed to Valgrind." FORCE)

    # make memcheck target
    add_custom_target(memcheck ctest -T memcheck -j ${NUMBER_OF_CORES} USES_TERMINAL)
  else()
    set(VALGRIND_FOUND 0)
  endif()
else()
  # Valgrind doesn't work on Macs.
  set(VALGRIND_FOUND 0)
endif()

# Testing
include(CTest)
enable_testing()

# Source directories.
add_subdirectory(src)

# MAM4 box model (for testing ported parameterizations)
if (MAM4XX_ENABLE_BOX)
  message(STATUS "Enabling cross validation with MAM4 box model.")
  if (NOT MAM4BOX_DIR)
    message(FATAL_ERROR "MAM4 box model cross validation requested, but MAM4BOX_DIR is not set!")
  elseif(NOT EXISTS ${MAM4BOX_DIR})
    message(FATAL_ERROR "Invalid MAM4BOX_DIR: ${MAM4BOX_DIR}")
  endif()
  # Build the MAM4 box model's core library.
  set(MAM4BOX_CMAKE_OPTS -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
                         -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                         -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                         -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                         -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                         -DMAMBOX_PRECISION=${HAERO_PRECISION}
                         -DMAMBOX_ENABLE_DRIVER=OFF
                         -DMAMBOX_ENABLE_VALIDATION=OFF
                         -DMAMBOX_NUM_COLUMNS=1
                         -DMAMBOX_NUM_LEVELS=1
                         -DMAMBOX_NUM_SPECIES=35
                         -DMAMBOX_NUM_BLACK_CARBON_TAGS=1
                         -DMAMBOX_NUM_PRIMARY_ORGANIC_TAGS=1
                         -DMAMBOX_NUM_SECONDARY_ORGANIC_TAGS=1
                         -DMAMBOX_NUM_SECONDARY_ORGANIC_GAS_TAGS=1
                         -DMAMBOX_RAIN_EVAP_TO_COARSE_AERO=1)
  include(ExternalProject)
  ExternalProject_Add(mam4box_proj
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}/mam4box
                      SOURCE_DIR ${MAM4BOX_DIR}
                      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/mam4box
                      INSTALL_DIR ${PROJECT_BINARY_DIR}
                      CMAKE_ARGS ${MAM4BOX_CMAKE_OPTS}
                      LOG_CONFIGURE TRUE
                      BUILD_COMMAND ${MAKE} -j
                      LOG_BUILD TRUE
                      INSTALL_COMMAND ${MAKE} install
                      LOG_INSTALL TRUE)
  # The MAM4 box model calls its library "mam_core", so we must do the same.
  add_library(mam_core STATIC IMPORTED GLOBAL)
  set_target_properties(mam_core PROPERTIES IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/lib/libmam_core.a)
  add_dependencies(mam_core mam4box_proj skywalker_f90)

  #----------------------------------------------
  # Build the MAM4 box model's Skywalker drivers
  #----------------------------------------------

  # Make sure the drivers can access the relevant Fortran modules and libraries.
  include_directories(${PROJECT_BINARY_DIR}/mam4box/core)
  #include_directories(${CMAKE_CURRENT_BINARY_DIR})
  set(MAMBOX_LIBRARIES skywalker_f90;skywalker;mam_core)

  # Copy the box model's standalone tests into its own source directory.
  file(COPY ${MAM4BOX_DIR}/standalone/tests/
       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/mam4box_drivers)

  # Build the drivers.
  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/mam4box_drivers)
endif()

# Formatting and format checking using clang-format.
find_program(CLANG_FORMAT clang-format)
if (NOT CLANG_FORMAT STREQUAL "CLANG_FORMAT-NOTFOUND")
  add_custom_target(format-cxx
    find ${PROJECT_SOURCE_DIR}/src -name "*.[hc]pp" -exec ${CLANG_FORMAT} -i --style=google {} \+;
    VERBATIM
    COMMENT "Auto-formatting C++ code...")
  add_custom_target(format-cxx-check
    find ${PROJECT_SOURCE_DIR}/src -name "*.[hc]pp" -exec ${CLANG_FORMAT} -n --Werror -ferror-limit=1 --style=google {} \+;
    VERBATIM
    COMMENT "Checking C++ formatting...")
endif()
