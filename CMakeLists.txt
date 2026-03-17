cmake_minimum_required(VERSION 3.17.0)

#----------------------
# Configurable options
#----------------------

option(MAM4XX_ENABLE_GPU       "Enable GPU support" OFF)
option(MAM4XX_ENABLE_MPI       "Enable MPI parallelism" ON)
option(MAM4XX_ENABLE_COVERAGE  "Enable code coverage instrumentation" OFF)
option(MAM4XX_ENABLE_SKYWALKER "Enable Skywalker cross validation" ON)
option(MAM4XX_ENABLE_TESTS     "Enable unit tests" ON)

# This option is only used in CI, where we have to use special sauce to get the
# submodules working with SSH. No mortal user should be concerned with this.
option(DISABLE_SUBMODULE_CHECKS "Skip checks for git submodules" OFF)
mark_as_advanced(FORCE DISABLE_SUBMODULE_CHECKS)

# Single or double precision?
if (CMAKE_VERSION GREATER_EQUAL 4.2)
  set(CACHE{MAM4XX_PRECISION} TYPE STRING HELP "Precision of Real numbers (single, [double])" VALUE double)
else()
  set(MAM4XX_PRECISION double CACHE STRING "Precision of Real numbers (single, [double])")
endif()

# Are we in standalone mode?
if (NOT DEFINED EAMXX_VERSION_MAJOR)
  set(MAM4XX_STANDALONE ON)
  set(CXX_STANDARD 17)
else()
  set(MAM4XX_STANDALONE OFF)
endif()

if( NOT DEFINED SCREAM_NUM_VERTICAL_LEV)
  # Set the default number of level
  set(NUM_VERTICAL_LEVELS 72 CACHE STRING "the number of vertical levels per column")
else()
  # Use the same number of levels as EAMxx if MAM4xx is run under EAMxx (or SCREAM)
  set(NUM_VERTICAL_LEVELS ${SCREAM_NUM_VERTICAL_LEV} CACHE STRING "the number of vertical levels per column equals scream num levels")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Blessed version of clang-format.
set(CLANG_FORMAT_VERSION 14)

#--------------------------
# End configurable options
#--------------------------

project(mam4xx)
enable_language(CXX)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

if (MAM4XX_ENABLE_MPI)
  find_package(MPI COMPONENTS C CXX REQUIRED)
  include_directories(${MPI_INCLUDE_DIRS})
  link_libraries(${MPI_LIBRARIES})
endif()

message(STATUS "Generating project files in build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "C++ compiler is ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
message(STATUS "C compiler is ${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION})")

# Version numbers.
set(MAM4XX_MAJOR_VERSION 0)
set(MAM4XX_MINOR_VERSION 1)
set(MAM4XX_PATCH_VERSION 0)

set(MAM4XX_VERSION "${MAM4XX_MAJOR_VERSION}.${MAM4XX_MINOR_VERSION}.${MAM4XX_PATCH_VERSION}")
message(STATUS "Configuring MAM4xx v${MAM4XX_VERSION} (build type: ${CMAKE_BUILD_TYPE})")

if(MAM4XX_ENABLE_GPU)
  message(STATUS "Building for GPU")
else() #CPU
  message(STATUS "Building for CPU")
endif()

# Precision of floating point numbers.
if (MAM4XX_PRECISION STREQUAL "double")
  set(MAM4XX_REAL_TYPE double)
  set(MAM4XX_DOUBLE_PRECISION 1)
elseif (MAM4XX_PRECISION STREQUAL "single")
  set(MAM4XX_REAL_TYPE float)
  set(MAM4XX_DOUBLE_PRECISION 0)
else()
  message(FATAL_ERROR "Invalid MAM4XX_PRECISION: ${MAM4XX_PRECISION} (must be single or double)")
endif()
message(STATUS "Using ${MAM4XX_PRECISION} precision floating point numbers")

# Report the installation prefix.
message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

include_directories(${PROJECT_BINARY_DIR})
include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)

# Emit compile_commands.json for use with clangd.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Third-party libraries
include(FetchContent)
if (MAM4XX_STANDALONE)
  set(EKAT_ENABLE_ALL_PACKAGES ON)
  set(EKAT_ENABLE_MPI ${MAM4XX_ENABLE_MPI})
  set(EKAT_ENABLE_TESTS OFF)
  FetchContent_Declare(
    EKAT
    GIT_REPOSITORY https://github.com/E3SM-Project/EKAT.git
    GIT_TAG        master
  )
  list(APPEND mam4xx_tpls EKAT)
endif()
FetchContent_Declare(
  skywalker
  GIT_REPOSITORY https://github.com/eagles-project/skywalker.git
  GIT_TAG        main
)
list(APPEND mam4xx_tpls skywalker)
FetchContent_MakeAvailable(${mam4xx_tpls})
if (MAM4XX_STANDALONE)
  set(EKAT_INCLUDE_DIRS
    ${ekat_SOURCE_DIR}/src/kokkos
    ${ekat_SOURCE_DIR}/extern/kokkos/core/src
  )
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ekat_SOURCE_DIR}/cmake/")
endif()

# Code coverage
if (MAM4XX_ENABLE_COVERAGE)
  message(STATUS "Enabling code coverage instrumentation")
  find_program(LCOV_EXE lcov DOC "Lcov code coverage tool")
  if (LCOV_EXE MATCHES "NOTFOUND")
    message(FATAL_ERROR "Could not find lcov for code coverage reporting!")
  endif()

  # Add code coverage compiler/linker flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")

  # Add a "make coverage" target.
  add_custom_target(coverage ${LCOV_EXE} --capture --directory . -o unfiltered_coverage.info
    COMMAND ${LCOV_EXE} --remove unfiltered_coverage.info -o coverage.info '*/external/*' '*/tests/*' '*/validation/*'
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT "Generating coverage report (coverage.info)")
endif()

# Support for valgrind (Linux only)
if (NOT TARGET memcheck)
  include(add_memcheck_target)
  add_memcheck_target()
endif()

# Testing
if (MAM4XX_ENABLE_TESTS OR MAM4XX_ENABLE_SKYWALKER)
  include(CTest)
  enable_testing()
endif()

# Source directories.
add_subdirectory(src)

# Formatting and format checking using clang-format.
if (NOT TARGET format-cxx)
  find_program(CLANG_FORMAT clang-format)
  if (NOT CLANG_FORMAT STREQUAL "CLANG_FORMAT-NOTFOUND")
    # Is this the blessed version? If not, we create targets that warn the user
    # to obtain the right version.
    execute_process(COMMAND clang-format --version
      OUTPUT_VARIABLE CF_VERSION)
    string(STRIP ${CF_VERSION} CF_VERSION)
    if (NOT ${CF_VERSION} MATCHES ${CLANG_FORMAT_VERSION})
      add_custom_target(format-cxx
        echo "You have clang-format version ${CF_VERSION}, but ${CLANG_FORMAT_VERSION} is required."
        "Please make sure this version appears in your path and rerun config.sh.")
      add_custom_target(format-cxx-check
        echo "You have clang-format version ${CF_VERSION}, but ${CLANG_FORMAT_VERSION} is required."
        "Please make sure this version appears in your path and rerun config.sh.")
    else()
      add_custom_target(format-cxx
        find ${PROJECT_SOURCE_DIR}/src -name "*.[hc]pp" -exec ${CLANG_FORMAT} -i {} \+;
        VERBATIM
        COMMENT "Auto-formatting C++ code...")
      add_custom_target(format-cxx-check
        find ${PROJECT_SOURCE_DIR}/src -name "*.[hc]pp" -exec ${CLANG_FORMAT} -n --Werror -ferror-limit=1 {} \+;
        VERBATIM
        COMMENT "Checking C++ formatting...")
    endif()
  endif()
endif()
