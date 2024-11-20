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
  const GasDrydepSpecies drydep_list[n_drydep] = {
      GasDrydepSpecies::H2O2, GasDrydepSpecies::H2SO4, GasDrydepSpecies::SO2};
  // --- data for effective Henry's Law coefficient ---
  constexpr Real dheff[n_drydep * 6] = {
      8.70e+04, 7320., 2.2e-12,  -3730., 0.,      0.,   // H2O2
      1.e+11,   6014., 0.,       0.,     0.,      0.,   // H2SO4
      1.36e+00, 3100., 1.30e-02, 1960.,  6.6e-08, 1500. // SO2
  };
  // NOTE: we are using fortran indexing.
  constexpr int mapping[n_drydep] = {1, 2, 3};
  // BAD CONSTANT
  constexpr Real ph = 1.e-5; // measure of the acidity (dimensionless)

  constexpr Real t0 = 298.0;    // Standard Temperature
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

inline Data set_gas_drydep_data() {
  using View1D = typename DeviceType::view_1d<Real>;
  using View2D = typename DeviceType::view_2d<Real>;
  using View1DHost = typename HostType::view_1d<Real>;

  using ViewBool1D = typename DeviceType::view_1d<bool>;
  using ViewInt1D = typename DeviceType::view_1d<int>;
  using ViewBool1DHost = typename HostType::view_1d<bool>;
  using ViewInt1DHost = typename HostType::view_1d<int>;

  // allocate the views
  constexpr int n_drydep = 3;
  constexpr int gas_pcnst = mam4::gas_chemistry::gas_pcnst;
  Data data{
      View1D("drat", n_drydep),
      View1D("foxd", n_drydep),
      View2D("rac", NSeas, NLUse),
      View2D("rclo", NSeas, NLUse),
      View2D("rcls", NSeas, NLUse),
      View2D("rgso", NSeas, NLUse),
      View2D("rgss", NSeas, NLUse),
      View2D("ri", NSeas, NLUse),
      View2D("rlu", NSeas, NLUse),
      View2D("z0", NSeas, NLUse),
      ViewBool1D("has_dvel", gas_pcnst),
      ViewInt1D("map_dvel", gas_pcnst),
      -1, // FIXME: so2_ndx
  };

  // populate the views with data taken from mam4 validation test data
  Real drat_a[n_drydep] = {1.3740328303634228, 2.3332297194282963,
                           1.8857345177418792};
  View1DHost drat_h(drat_a, n_drydep);
  Kokkos::deep_copy(data.drat, drat_h);

  Real foxd_a[n_drydep] = {1.0, 1e-36, 1e-36};
  View1DHost foxd_h(foxd_a, n_drydep);
  Kokkos::deep_copy(data.foxd, foxd_h);

  // clang-format off
  Real rac_a[NSeas][NLUse] = { 
     { 100.0, 200.0, 100.0,2000.0,2000.0,2000.0,   0.0,   0.0, 300.0, 150.0, 200.0}, 
     { 100.0, 150.0, 100.0,1500.0,2000.0,1700.0,   0.0,   0.0, 200.0, 120.0, 140.0},
     { 100.0,  10.0, 100.0,1000.0,2000.0,1500.0,   0.0,   0.0, 100.0,  50.0, 120.0}, 
     { 100.0,  10.0,  10.0,1000.0,2000.0,1500.0,   0.0,   0.0,  50.0,  10.0,  50.0}, 
     { 100.0,  50.0,  80.0,1200.0,2000.0,1500.0,   0.0,   0.0, 200.0,  60.0, 120.0} };
  auto rac_h = Kokkos::create_mirror_view(data.rac);
  for (int i = 0; i < NSeas; ++i) {
    for (int j = 0; j < NLUse; ++j) {
        rac_h(i, j) = rac_a[i][j];
    }
  }  
  Kokkos::deep_copy(data.rac, rac_h);

  Real rclo_a[NSeas][NLUse] = {
     { 1e+36,1000.0,1000.0,1000.0,1000.0,1000.0, 1e+36, 1e+36,1000.0,1000.0,1000.0}, 
     { 1e+36, 400.0, 400.0, 400.0,1000.0, 600.0, 1e+36, 1e+36, 400.0, 400.0, 400.0}, 
     { 1e+36,1000.0, 400.0, 400.0,1000.0, 600.0, 1e+36, 1e+36, 800.0, 600.0, 600.0},
     { 1e+36,1000.0,1000.0, 400.0,1500.0, 600.0, 1e+36, 1e+36, 800.0,1000.0, 800.0},
     { 1e+36,1000.0, 500.0, 500.0,1500.0, 700.0, 1e+36, 1e+36, 600.0, 800.0, 800.0} };
  auto rclo_h = Kokkos::create_mirror_view(data.rclo);
  for (int i = 0; i < NSeas; ++i) {
    for (int j = 0; j < NLUse; ++j) {
        rclo_h(i, j) = rclo_a[i][j];
    }
  }
  Kokkos::deep_copy(data.rclo, rclo_h);
  
  Real rcls_a[NSeas][NLUse] = {
     { 1e+36,2000.0,2000.0,2000.0,2000.0,2000.0, 1e+36, 1e+36,2500.0,2000.0,4000.0}, 
     { 1e+36,9000.0,9000.0,9000.0,2000.0,4000.0, 1e+36, 1e+36,9000.0,9000.0,9000.0},
     { 1e+36, 1e+36,9000.0,9000.0,3000.0,6000.0, 1e+36, 1e+36,9000.0,9000.0,9000.0},
     { 1e+36, 1e+36, 1e+36,9000.0, 200.0, 400.0, 1e+36, 1e+36,9000.0, 1e+36,9000.0}, 
     { 1e+36,4000.0,4000.0,4000.0,2000.0,3000.0, 1e+36, 1e+36,4000.0,4000.0,8000.0} };
  auto rcls_h = Kokkos::create_mirror_view(data.rcls);
  for (int i = 0; i < NSeas; ++i) {
    for (int j = 0; j < NLUse; ++j) {
        rcls_h(i, j) = rcls_a[i][j];
    }
  }
  Kokkos::deep_copy(data.rcls, rcls_h);

  Real rgso_a[NSeas][NLUse] = {
     { 300.0, 150.0, 200.0, 200.0, 200.0, 300.0,2000.0, 400.0,1000.0, 180.0, 200.0},
     { 300.0, 150.0, 200.0, 200.0, 200.0, 300.0,2000.0, 400.0, 800.0, 180.0, 200.0}, 
     { 300.0, 150.0, 200.0, 200.0, 200.0, 300.0,2000.0, 400.0,1000.0, 180.0, 200.0}, 
     { 600.0,3500.0,3500.0,3500.0,3500.0,3500.0,2000.0, 400.0,3500.0,3500.0,3500.0},
     { 300.0, 150.0, 200.0, 200.0, 200.0, 300.0,2000.0, 400.0,1000.0, 180.0, 200.0} };
  auto rgso_h = Kokkos::create_mirror_view(data.rgso);
  for (int i = 0; i < NSeas; ++i) {
    for (int j = 0; j < NLUse; ++j) {
        rgso_h(i, j) = rgso_a[i][j];
    }
  }
  Kokkos::deep_copy(data.rgso, rgso_h);

  Real rgss_a[NSeas][NLUse] = {
     { 400.0, 150.0, 350.0, 500.0, 500.0, 100.0,   0.0,1000.0,   0.0, 220.0, 400.0}, 
     { 400.0, 200.0, 350.0, 500.0, 500.0, 100.0,   0.0,1000.0,   0.0, 300.0, 400.0},
     { 400.0, 150.0, 350.0, 500.0, 500.0, 200.0,   0.0,1000.0,   0.0, 200.0, 400.0}, 
     { 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,   0.0,1000.0, 100.0, 100.0,  50.0},
     { 500.0, 150.0, 350.0, 500.0, 500.0, 200.0,   0.0,1000.0,   0.0, 250.0, 400.0} };
  auto rgss_h = Kokkos::create_mirror_view(data.rgss);
  for (int i = 0; i < NSeas; ++i) {
    for (int j = 0; j < NLUse; ++j) {
        rgss_h(i, j) = rgss_a[i][j];
    }
  }
  Kokkos::deep_copy(data.rgss, rgss_h);

  Real ri_a[NSeas][NLUse] = {
     { 1e+36,  60.0, 120.0,  70.0, 130.0, 100.0, 1e+36, 1e+36,  80.0, 100.0, 150.0}, 
     { 1e+36, 1e+36, 1e+36, 1e+36, 250.0, 500.0, 1e+36, 1e+36, 1e+36, 1e+36, 1e+36}, 
     { 1e+36, 1e+36, 1e+36, 1e+36, 250.0, 500.0, 1e+36, 1e+36, 1e+36, 1e+36, 1e+36},
     { 1e+36, 1e+36, 1e+36, 1e+36, 400.0, 800.0, 1e+36, 1e+36, 1e+36, 1e+36, 1e+36},
     { 1e+36, 120.0, 240.0, 140.0, 250.0, 190.0, 1e+36, 1e+36, 160.0, 200.0, 300.0} };
  auto ri_h = Kokkos::create_mirror_view(data.ri);
  for (int i = 0; i < NSeas; ++i) {
    for (int j = 0; j < NLUse; ++j) {
        ri_h(i, j) = ri_a[i][j];
    }
  }
  Kokkos::deep_copy(data.ri, ri_h);

  Real rlu_a[NSeas][NLUse] = {
     { 1e+36,2000.0,2000.0,2000.0,2000.0,2000.0, 1e+36, 1e+36,2500.0,2000.0,4000.0}, 
     { 1e+36,9000.0,9000.0,9000.0,4000.0,8000.0, 1e+36, 1e+36,9000.0,9000.0,9000.0}, 
     { 1e+36, 1e+36,9000.0,9000.0,4000.0,8000.0, 1e+36, 1e+36,9000.0,9000.0,9000.0}, 
     { 1e+36, 1e+36, 1e+36, 1e+36,6000.0,9000.0, 1e+36, 1e+36,9000.0,9000.0,9000.0}, 
     { 1e+36,4000.0,4000.0,4000.0,2000.0,3000.0, 1e+36, 1e+36,4000.0,4000.0,8000.0} };
  auto rlu_h = Kokkos::create_mirror_view(data.rlu);
  for (int i = 0; i < NSeas; ++i) {
    for (int j = 0; j < NLUse; ++j) {
        rlu_h(i, j) = rlu_a[i][j];
    }
  }
  Kokkos::deep_copy(data.rlu, rlu_h);

  Real z0_a[NSeas][NLUse] = {
     { 1.000, 0.250, 0.050, 1.000, 1.000, 1.000,0.0006, 0.002, 0.150, 0.100, 0.100}, 
     { 1.000, 0.100, 0.050, 1.000, 1.000, 1.000,0.0006, 0.002, 0.100, 0.080, 0.080}, 
     { 1.000, 0.005, 0.050, 1.000, 1.000, 1.000,0.0006, 0.002, 0.100, 0.020, 0.060}, 
     { 1.000, 0.001, 0.001, 1.000, 1.000, 1.000,0.0006, 0.002, 0.001, 0.001, 0.040},
     { 1.000, 0.030, 0.020, 1.000, 1.000, 1.000,0.0006, 0.002, 0.010, 0.030, 0.060} };
  auto z0_h = Kokkos::create_mirror_view(data.z0);
  for (int i = 0; i < NSeas; ++i) {
    for (int j = 0; j < NLUse; ++j) {
        z0_h(i, j) = z0_a[i][j];
    }
  }
  Kokkos::deep_copy(data.z0, z0_h);

  // has_dvel maps species in solsym to true iff they appear in drydep_list
  bool has_dvel_a[gas_pcnst] = {}; // all false by default
  has_dvel_a[1] = true;            // H2O2
  has_dvel_a[2] = true;            // H2SO4
  has_dvel_a[3] = true;            // SO2
  ViewBool1DHost has_dvel_h(has_dvel_a, gas_pcnst);
  Kokkos::deep_copy(data.has_dvel, has_dvel_h);

  // map_dvel maps indices of species in solsym to those in drydep_list
  int map_dvel_a[gas_pcnst] = {};
  for (int i = 0; i < gas_pcnst; ++i) {
    map_dvel_a[i] = -1; // most indices unmapped
  }
  map_dvel_a[1] = 0; // H2O2
  map_dvel_a[2] = 1; // H2SO4
  map_dvel_a[3] = 2; // SO2
  ViewInt1DHost map_dvel_h(map_dvel_a, gas_pcnst);
  Kokkos::deep_copy(data.map_dvel, map_dvel_h);

  // index of "SO2" within solsym above
  data.so2_ndx = 3;

  return data;
}


} // namespace mam4::seq_drydep

#endif
