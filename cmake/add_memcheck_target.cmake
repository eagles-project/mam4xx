# This macro adds a "make memcheck" target that runs Valgrind on all tests.
# It only works on Linux.
macro(add_memcheck_target)
  if (UNIX AND NOT APPLE)
    find_program(VALGRIND valgrind)
    if (NOT VALGRIND MATCHES "-NOTFOUND")
      message(STATUS "Valgrind found. Enabling `make memcheck`")
      set(VALGRIND_FOUND 1)
      set(CTEST_MEMORYCHECK_COMMAND ${VALGRIND})
      # Add "--gen-suppressions=all" to MEMORYCHECK_COMMAND_OPTIONS to generate
      # suppressions for Valgrind's false positives. The suppressions show up
      # right in the MemoryChecker.*.log files.
      set(CTEST_MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --show-leak-kinds=all --errors-for-leak-kinds=definite,possible --track-origins=yes --error-exitcode=1 --trace-children=yes --suppressions=${PROJECT_SOURCE_DIR}/tools/valgrind/scasm.supp" CACHE STRING "Options passed to Valgrind." FORCE)

      # make memcheck target
      add_custom_target(memcheck ctest -T memcheck -j USES_TERMINAL)
    else()
      set(VALGRIND_FOUND 0)
    endif()
  else()
    # Valgrind doesn't work on Macs.
    set(VALGRIND_FOUND 0)
  endif()

endmacro()
