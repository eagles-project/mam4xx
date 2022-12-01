message(STATUS "Using Deception CMake configuration")

message(STATUS "Configuring Kokkos for CUDA Architecture 60")
set(Kokkos_ARCH_PASCAL60 ON CACHE BOOL "")
# set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "")
# set(Kokkos_ARCH_TURING75 ON CACHE BOOL "")
# set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "")
