#ifndef MAM4XX_SEQ_DRYDEP_HPP
#define MAM4XX_SEQ_DRYDEP_HPP

#include <mam4xx/mam4_types.hpp>

namespace mam4::seq_drydep { // C++ version of E3SM's seq_drydep_mod.F90

// maximum number of species involved in dry deposition
constexpr int maxspc = 210;

// number of seasons
constexpr int NSeas = 5;

//=========================================
// data for E3SM dry deposition of tracers
//=========================================

// This struct holds device views related to dry gas deposition for tracers.
// It replaces the global arrays used by E3SM in the seq_drydep Fortran module.
// These views must be allocated and populated with data by a host model,
// testing environment, etc.
struct Data {
  // molecular diffusivity ratio (D_H2O/D_X) [-], shape=(n_drydep)
  DeviceType::view_1d<Real> drat;
  // reactive factor for oxidation [-], shape=(n_drydep)
  DeviceType::view_1d<Real> foxd;
  // aerodynamic resistance to lower canopy [s/m], shape=(NSeas, NLUse)
  DeviceType::view_2d<Real> rac;
  // lower canopy resistance for O3 [s/m], shape=(NSeas, NLUse)
  DeviceType::view_2d<Real> rclo;
  // lower canopy resistance for SO2 [s/m], shape=(NSeas, NLUse)
  DeviceType::view_2d<Real> rcls;
  // ground ѕurface resistance for O3 [s/m], shape=(NSeas, NLUse)
  DeviceType::view_2d<Real> rgso;
  // ground surface resistance for SO2 [s/m], shape=(NSeas, NLUse)
  DeviceType::view_2d<Real> rgss;
  // richardson number [-], shape=(NSeas, NLUse)
  DeviceType::view_2d<Real> ri;
  // resistance of leaves in upper canopy [s/m], shape=(NSeas, NLUse)
  DeviceType::view_2d<Real> rlu;
  // roughness length [m], shape=(NSeas, NLUse)
  DeviceType::view_2d<Real> z0;

  // This array maps species indices to true or false, depending on whether
  // the species participates in dry deposition.
  DeviceType::view_1d<bool> has_dvel; // shape=(gas_pcnst)
  // This array maps species indices to dry deposition indices. These
  // dry deposition indices count range from 0 to n_drydep-1.
  DeviceType::view_1d<Real> map_dvel; // shape=(gas_pcnst)
};

// This function computes the H coefficients corresponding to the given surface
// temperature. It must be implemented by the client application as an on-device
// Kokkos function, and must be defined IN THE TRANSLATION UNIT IN WHICH IT'S
// CALLED because we don't use relocatable device code. The function can be
// implemented in the atmospheric host model or in a testing environment, for
// example.
KOKKOS_FUNCTION void setHCoeff(Real sfc_temp, Real heff[maxspc]);

} // namespace mam4::seq_drydep

#endif
