cmake_minimum_required(VERSION 3.17.0)

message(STATUS "Configuring with build type: ${CMAKE_BUILD_TYPE}")

#----------------------
# Configurable options
#----------------------

option(ENABLE_COVERAGE  "Enable code coverage instrumentation" OFF)
option(ENABLE_SKYWALKER "Enable Skywalker cross validation" ON)
option(ENABLE_TESTS     "Enable unit tests" ON)

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

project(mam4xx)
enable_language(CXX)

# We rely on Haero for basic data structures and third-party libraries.
if (NOT TARGET haero)
  if (NOT MAM4XX_HAERO_DIR)
    message(FATAL_ERROR "MAM4XX_HAERO_DIR is not specified! Please provide a path to Haero.")
  endif()
  find_package(haero REQUIRED HINTS ${MAM4XX_HAERO_DIR})
  set (required_ekat_pkgs Logging Pack KokkosUtils)
  set (missing_ekat_pkgs)
  foreach (pkg IN ITEMS ${required_ekat_pkgs})
    if (NOT TARGET ekat::${pkg})
      list(APPEND missing_ekat_pkgs ${pkg})
    endif()
  endforeach()
  if (missing_ekat_pkgs)
    string(CONCAT msg
      "Error! Haero is linking against a non-sufficient version of ekat\n"
      " Required Ekat pacakges: KokkosUtils, Pack, Logging\n"
      " Missing packages: ${missing_ekat_pkgs}")
    message ("${msg}")
    message (FATAL_ERROR "Aborting...")
  endif()

  if(HAERO_ENABLE_GPU)
    message(STATUS "Building for GPU")
  else() #CPU
    message(STATUS "Building for CPU")
  endif()
endif()

message(STATUS "Generating project files in build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "C++ compiler is ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
message(STATUS "C compiler is ${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION})")

# Version numbers.
set(MAM4XX_MAJOR_VERSION 0)
set(MAM4XX_MINOR_VERSION 1)
set(MAM4XX_PATCH_VERSION 0)

set(MAM4XX_VERSION "${MAM4XX_MAJOR_VERSION}.${MAM4XX_MINOR_VERSION}.${MAM4XX_PATCH_VERSION}")
message(STATUS "Configuring MAM4xx v${MAM4XX_VERSION}")

# Precision of floating point numbers.
message(STATUS "Using ${HAERO_PRECISION} precision floating point numbers")

# Report the installation prefix.
message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

# We look for headers in these directories.
include_directories("${PROJECT_BINARY_DIR}")
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)

link_directories("${MAM4XX_HAERO_DIR}/${CMAKE_INSTALL_LIBDIR}")

# emit compile_commands.json for use with clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Code coverage
if (ENABLE_COVERAGE)
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
if (ENABLE_TESTS OR ENABLE_SKYWALKER)
  include(CTest)
  enable_testing()
endif()

# Third-party libraries.
add_subdirectory(ext)

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
