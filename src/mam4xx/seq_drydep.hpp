#ifndef MAM4XX_SEQ_DRYDEP_HPP
#define MAM4XX_SEQ_DRYDEP_HPP

#include <haero/math.hpp>
#include <mam4xx/mam4_types.hpp>

namespace mam4::seq_drydep { // C++ version of E3SM's seq_drydep_mod.F90

// maximum number of species involved in dry deposition
constexpr int maxspc = 210;

// number of seasons
constexpr int NSeas = 5;

// number of land use types
constexpr int NLUse = 11;

// number of gas species in dry dep list.
constexpr int n_drydep = 3;

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
  DeviceType::view_1d<int> map_dvel; // shape=(gas_pcnst)

  // the constituent index corresponding to SO2 gas (or -1 if not present)
  int so2_ndx;
};

// Define the Species enum for dry deposition species identification
// BAD CONSTANT
enum class GasDrydepSpecies { H2O2, H2SO4, SO2, CO2, NH3 };
/**
 * Calculates Henry's law coefficients based on surface temperature and other
 * parameters.
 *
 * @param sfc_temp The surface temperature in Kelvin. [input]
 * @param heff Array where the calculated Henry's law coefficients will be
 * stored. [output]
 */
KOKKOS_INLINE_FUNCTION
void set_hcoeff_scalar(const Real sfc_temp, Real heff[]) {

  // Define dheff array with size n_species_table*6
  // NOTE: We are hard-coding the table dheff with only 3 species:
  // H2O2, H2SO4, SO2.
  // The original table can be found in the seq_drydep_mod.F90 module.
  // BAD CONSTANT

  // NOTE:
  const GasDrydepSpecies drydep_list[n_drydep] = {
      GasDrydepSpecies::H2O2, GasDrydepSpecies::H2SO4, GasDrydepSpecies::SO2};
  constexpr Real dheff[n_drydep * 6] = {
      8.70e+04, 7320., 2.2e-12,  -3730., 0.,      0.,   // H2O2
      1.e+11,   6014., 0.,       0.,     0.,      0.,   // H2SO4
      1.36e+00, 3100., 1.30e-02, 1960.,  6.6e-08, 1500. // SO2
  };
  constexpr int mapping[n_drydep] = {1, 2, 3};
  // BAD CONSTANT
  constexpr Real ph = 1.e-5; // measure of the acidity (dimensionless)

  const Real t0 = 298.0;        // Standard Temperature
  const Real ph_inv = 1.0 / ph; // Inverse of PH

  const Real wrk = (t0 - sfc_temp) / (t0 * sfc_temp);

  for (int m = 0; m < n_drydep; ++m) {
    const int l = mapping[m] - 1; // Adjust for 0-based indexing
    const int id = 6 * l;
    Real e298 = dheff[id];    // Adjusted for 0-based indexing
    Real dhr = dheff[id + 1]; // Adjusted for 0-based indexing
    heff[m] = haero::exp(dhr * wrk) * e298;

    // Calculate coefficients based on the drydep tables
    if (dheff[id + 2] != 0.0 && dheff[id + 4] == 0.0) {
      e298 = dheff[id + 2];
      dhr = dheff[id + 3];
      Real dk1 = haero::exp(dhr * wrk) * e298;
      heff[m] =
          (heff[m] != 0.0) ? heff[m] * (1.0 + dk1 * ph_inv) : dk1 * ph_inv;
    }

    // For coefficients that are non-zero AND CO2 or NH3 handle things this way
    if (dheff[id + 4] != 0.0) {
      GasDrydepSpecies species = drydep_list[m];
      if (species == GasDrydepSpecies::CO2 ||
          species == GasDrydepSpecies::NH3 ||
          species == GasDrydepSpecies::SO2) {
        e298 = dheff[id + 2];
        dhr = dheff[id + 3];
        Real dk1 = haero::exp(dhr * wrk) * e298;
        e298 = dheff[id + 4];
        dhr = dheff[id + 5];
        Real dk2 = haero::exp(dhr * wrk) * e298;
        if (species == GasDrydepSpecies::CO2 ||
            species == GasDrydepSpecies::SO2) {
          heff[m] *= (1.0 + dk1 * ph_inv * (1.0 + dk2 * ph_inv));
        } else if (species == GasDrydepSpecies::NH3) {
          heff[m] *= (1.0 + dk1 * ph / dk2);
        } else {
          EKAT_KERNEL_ERROR_MSG("ERROR: Bad species encountered.\n");
        }
      }
    }
  }
}

} // namespace mam4::seq_drydep

#endif
