// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_WATER_UPTAKE_HPP
#define MAM4XX_WATER_UPTAKE_HPP

#include <Kokkos_Complex.hpp>
#include <haero/atmosphere.hpp>
#include <haero/surface.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/convproc.hpp>
#include <mam4xx/utils.hpp>
#include <mam4xx/wv_sat_methods.hpp>
namespace mam4 {
class Water_Uptake {

public:
  struct Config {

    Config(){};

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  Config config_;

public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 wet deposition"; }

  static constexpr Real eps = 1e-4; // Bad constant

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config());

  // validate -- validates the given atmospheric state and prognostics against
  // assumptions made by this implementation, returning true if the states are
  // valid, false if not
  KOKKOS_INLINE_FUNCTION
  bool validate(const AeroConfig &config, const ThreadTeam &team,
                const Atmosphere &atm, const Surface &sfc,
                const Prognostics &progs) const;

  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atm,
                          const Surface &sfc, const Prognostics &progs,
                          const Diagnostics &diags,
                          const Tendencies &tends) const;
};

namespace water_uptake {
//  number of variables in state_q
constexpr int nvars = 40;
constexpr int maxd_aspectype = 14;
constexpr Real small_value_30 = 1e-30; // (Bad Constant)
constexpr Real small_value_31 = 1e-31; // (Bad Constant)
//-----------------------------------------------------------------------
// compute aerosol wet density
//-----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void modal_aero_water_uptake_wetdens(
    const Real wetvol[AeroConfig::num_modes()],
    const Real wtrvol[AeroConfig::num_modes()],
    const Real drymass[AeroConfig::num_modes()],
    const Real specdens_1[AeroConfig::num_modes()],
    Real wetdens[AeroConfig::num_modes()]) {

  // compute aerosol wet density (kg/m3)
  // looping over densities
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    static constexpr Real small_value_30 = 1.0e-30; // (BAD CONSTANT)
    if (wetvol[imode] > small_value_30) {
      // ! wet density
      wetdens[imode] =
          (drymass[imode] + Constants::density_h2o * wtrvol[imode]) /
          wetvol[imode];
    } else {
      // dry density
      wetdens[imode] = specdens_1[imode];
    }
  }
}

//----------------------------------------------------------------------
//  find the smallest real solution from the polynomial solver
//----------------------------------------------------------------------
//
KOKKOS_INLINE_FUNCTION
void find_real_solution(const Real rdry, const Kokkos::complex<Real> cx[4],
                        Real &rwet, int &nsol) {
  rwet = 1000.0 * rdry; // (BAD CONSTANT or UNIT CONVERSION)
  nsol = 0;
  for (int nn = 0; nn < 4; ++nn) {
    Real xr = cx[nn].real();
    Real xi = cx[nn].imag();
    if (haero::abs(xi) > haero::abs(xr) * Water_Uptake::eps) {
      continue;
    }
    if (xr > rwet) {
      continue;
    }
    if (xr < rdry * (1.0 - Water_Uptake::eps)) {
      continue;
    }
    if (haero::isnan(xr)) {
      continue;
    }

    rwet = xr;
    nsol = nn;
  };
}

//-----------------------------------------------------------------------
//     solves x**4 + p3 x**3 + p2 x**2 + p1 x + p0 = 0
//     where p0, p1, p2, p3 are real
//-----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void makoh_quartic(Kokkos::complex<Real> cx[4], const Real p3, const Real p2,
                   const Real p1, const Real p0) {

  // set complex zeros and 1/3 values
  Kokkos::complex<Real> czero = {};

  Real qq = -p2 * p2 / 36.0 + (p3 * p1 - 4.0 * p0) / 12.0;
  Real rr = -haero::cube(p2 / 6.0) + p2 * (p3 * p1 - 4.0 * p0) / 48.0 +
            (4.0 * p0 * p2 - p0 * p3 * p3 - p1 * p1) / 16.0;

  Kokkos::complex<Real> crad = Kokkos::sqrt(rr * rr + qq * qq * qq);
  Kokkos::complex<Real> cb = rr - crad;

  if (cb == czero) {
    // insoluble particle
    cx[0] = haero::cbrt(-p1);
    cx[1] = cx[0];
    cx[2] = cx[0];
    cx[3] = cx[0];
  } else {
    cb = Kokkos::pow(cb, 1.0 / 3.0);
    Kokkos::complex<Real> cy = -cb + qq / cb + p2 / 6.0;
    Kokkos::complex<Real> cb0 = Kokkos::sqrt(cy * cy - p0);
    Kokkos::complex<Real> cb1 = (p3 * cy - p1) / (2.0 * cb0);

    cb = p3 / 2.0 + cb1;
    crad = Kokkos::sqrt(cb * cb - 4.0 * (cy + cb0));
    cx[0] = (-cb + crad) / 2.0;
    cx[1] = (-cb - crad) / 2.0;

    cb = p3 / 2.0 - cb1;
    crad = Kokkos::sqrt(cb * cb - 4.0 * (cy - cb0));
    cx[2] = (-cb + crad) / 2.0;
    cx[3] = (-cb - crad) / 2.0;
  }
}

// calculates equlibrium radius r of haze droplets as function of
// dry particle mass and relative humidity s using kohler solution
// given in pruppacher and klett (eqn 6-35)
//
// for multiple aerosol types, assumes an internal mixture of aerosols
//-----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void modal_aero_kohler(const Real rdry_in, const Real hygro, const Real rh,
                       Real &rwet_out) {

  static constexpr Real rhow = 1.0;      // (BAD CONSTANT)
  static constexpr Real surften = 76.0;  // (BAD CONSTANT)
  static constexpr Real mw = 18.0;       // (BAD CONSTANT)
  static constexpr Real tair = 273.0;    // (BAD CONSTANT)
  static constexpr Real ugascon = 8.3e7; // (BAD CONSTANT)
  static constexpr Real factor_um2m =
      1.e-6; // (BAD CONSTANT) convert micron to m
  static constexpr Real factor_m2um =
      1.e6; // (BAD CONSTANT) convert m to micron
  static constexpr Real small_value_10 = 1.e-10; // (BAD CONSTANT)
  static constexpr Real rmax = 30.0;             // (BAD CONSTANT)

  // effect of organics on surface tension is neglected'
  const Real aa =
      2.0e4 * mw * surften / (ugascon * tair * rhow); // (BAD CONSTANT)

  const Real rdry = rdry_in * factor_m2um; // convert (m) to (microns)
  const Real vol = haero::cube(rdry);      // vol is r**3, not volume
  const Real bb = vol * hygro;

  // quartic
  const Real ss =
      utils::min_max_bound(small_value_10, 1.0 - Water_Uptake::eps, rh);

  const Real slog = haero::log(ss);
  const Real p43 = -aa / slog;
  const Real p42 = 0.0;
  const Real p41 = bb / slog - vol;
  const Real p40 = aa * vol / slog;

  const Real pp = haero::abs(-bb / aa) / (rdry * rdry);
  Real rwet = 0.0;
  int nsol = 0;
  Kokkos::complex<Real> cx4[4] = {};
  if (pp < Water_Uptake::eps) {
    // approximate solution for small particles
    rwet = rdry * (1.0 + pp * (1.0 / 3.0) / (1.0 - slog * rdry / aa));
  } else {
    makoh_quartic(cx4, p43, p42, p41, p40);
    find_real_solution(rdry, cx4, rwet, nsol);
  }

  // bound and convert from microns to m
  rwet = haero::min(rwet, rmax); // upper bound based on 1 day lifetime
  rwet_out = rwet * factor_um2m;
}

//-----------------------------------------------------------------------
//
// Purpose: Compute aerosol wet radius and other properties
//
// Method:  Kohler theory
//
// Author:  S. Ghan
//
//-----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void modal_aero_water_uptake_wetaer(
    Real rhcrystal[AeroConfig::num_modes()],
    Real rhdeliques[AeroConfig::num_modes()],
    Real dgncur_a[AeroConfig::num_modes()],
    Real dryrad[AeroConfig::num_modes()], Real hygro[AeroConfig::num_modes()],
    const Real rh, Real naer[AeroConfig::num_modes()],
    Real dryvol[AeroConfig::num_modes()], Real wetrad[AeroConfig::num_modes()],
    Real wetvol[AeroConfig::num_modes()], Real wtrvol[AeroConfig::num_modes()],
    Real dgncur_awet[AeroConfig::num_modes()],
    Real qaerwat[AeroConfig::num_modes()]) {

  //-----------------------------------------------------------------------
  // loop over all aerosol modes
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {

    const Real hystfac =
        1.0 / haero::max(1.0e-5, (rhdeliques[imode] -
                                  rhcrystal[imode])); // (BAD CONSTANT)

    water_uptake::modal_aero_kohler(dryrad[imode], hygro[imode], rh,
                                    wetrad[imode]);

    wetrad[imode] = haero::max(wetrad[imode], dryrad[imode]);
    wetvol[imode] = (Constants::pi * 4.0 / 3.0) * haero::cube(wetrad[imode]);
    wetvol[imode] = haero::max(wetvol[imode], dryvol[imode]);
    wtrvol[imode] = wetvol[imode] - dryvol[imode];
    wtrvol[imode] = haero::max(wtrvol[imode], 0.0);

    // apply simple treatment of deliquesence/crystallization hysteresis
    // for rhcrystal < rh < rhdeliques, aerosol water is a fraction of
    // the "upper curve" value, and the fraction is a linear function of rh
    if (rh < rhcrystal[imode]) {
      wetrad[imode] = dryrad[imode];
      wetvol[imode] = dryvol[imode];
      wtrvol[imode] = 0.0;
    } else if (rh < rhdeliques[imode]) {
      wtrvol[imode] = wtrvol[imode] * hystfac * (rh - rhcrystal[imode]);
      wtrvol[imode] = haero::max(wtrvol[imode], 0.0);
      wetvol[imode] = dryvol[imode] + wtrvol[imode];
      wetrad[imode] = haero::cbrt(wetvol[imode] / (4.0 / 3.0 * Constants::pi));
    }

    // calculate wet aerosol diameter and aerosol water
    dgncur_awet[imode] = dgncur_a[imode] * (wetrad[imode] / dryrad[imode]);
    qaerwat[imode] = Constants::density_h2o * naer[imode] * wtrvol[imode];
  }
}

//-----------------------------------------------------------------------
// estimate clear air relative humidity using cloud fraction
//-----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void modal_aero_water_uptake_rh_clearair(const Real temperature,
                                         const Real pmid, const Real h2ommr,
                                         const Real cldn, Real &rh) {

  Real es = 0.0;
  Real qs = 0.0;

  wv_sat_methods::wv_sat_qsat_water(temperature, pmid, es, qs);

  static constexpr Real rh_max = 0.98; // (BAD CONSTANT)
  if (qs > h2ommr) {
    rh = h2ommr / qs;
  } else {
    rh = rh_max;
  }
  rh = utils::min_max_bound(0.0, rh_max, rh);

  static constexpr Real cldn_thresh = 1.0; // (BAD CONSTANT)
  if (cldn < cldn_thresh) {
    rh = (rh - cldn) / (1.0 - cldn); // RH of clear portion
  }
  rh = haero::max(rh, 0.0);
}

KOKKOS_INLINE_FUNCTION
void get_e3sm_parameters(
    int nspec_amode[AeroConfig::num_modes()],
    int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
    Real specdens_amode[maxd_aspectype], Real spechygro[maxd_aspectype]) {

  const int ntot_amode = AeroConfig::num_modes();
  int nspec_amode_temp[ntot_amode] = {7, 4, 7, 3};

  for (int i = 0; i < ntot_amode; ++i) {
    nspec_amode[i] = nspec_amode_temp[i];
  }

  Real specdens_amode_temp[maxd_aspectype] = {
      0.1770000000E+04, 0.1797693135 + 309, 0.1797693135 + 309,
      0.1000000000E+04, 0.1000000000E+04,   0.1700000000E+04,
      0.1900000000E+04, 0.2600000000E+04,   0.1601000000E+04,
      0.0000000000E+00, 0.0000000000E+00,   0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00};
  Real spechygro_temp[maxd_aspectype] = {
      0.5070000000E+00, 0.1797693135 + 309, 0.1797693135 + 309,
      0.1000000083E-09, 0.1400000000E+00,   0.1000000013E-09,
      0.1160000000E+01, 0.6800000000E-01,   0.1000000015E+00,
      0.0000000000E+00, 0.0000000000E+00,   0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00};

  for (int i = 0; i < maxd_aspectype; ++i) {
    specdens_amode[i] = specdens_amode_temp[i];
    spechygro[i] = spechygro_temp[i];
  }

  const int lspectype_amode_1d[ntot_amode * maxd_aspectype] = {
      1, 4, 5, 6, 8, 7, 9, 0, 0, 0, 0, 0, 0, 0, 1, 5, 7, 9, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 1, 6, 4, 5, 9, 0, 0, 0,
      0, 0, 0, 0, 4, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  int count = 0;
  for (int i = 0; i < ntot_amode; ++i) {
    for (int j = 0; j < maxd_aspectype; ++j) {
      lspectype_amode[j][i] = lspectype_amode_1d[count];
      count++;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void modal_aero_water_uptake_dryaer(
    int nspec_amode[AeroConfig::num_modes()],
    Real specdens_amode[maxd_aspectype], Real spechygro[maxd_aspectype],
    int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
    Real state_q[nvars], Real dgncur_a[AeroConfig::num_modes()],
    Real hygro[AeroConfig::num_modes()], Real naer[AeroConfig::num_modes()],
    Real dryrad[AeroConfig::num_modes()], Real dryvol[AeroConfig::num_modes()],
    Real drymass[AeroConfig::num_modes()],
    Real rhcrystal[AeroConfig::num_modes()],
    Real rhdeliques[AeroConfig::num_modes()],
    Real specdens_1[AeroConfig::num_modes()]) {

  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    hygro[imode] = 0.0;

    Real dryvolmr = 0.0;
    Real maer = 0.0;

    const Real sigmag = modes(imode).mean_std_dev;
    rhcrystal[imode] = modes(imode).crystallization_pt;
    rhdeliques[imode] = modes(imode).deliquescence_pt;
    const int nspec = nspec_amode[imode];
    int type_idx = lspectype_amode[0][imode] - 1; // Fortran to C++ indexing
    specdens_1[imode] = specdens_amode[type_idx];
    const Real spechygro_1 = spechygro[type_idx];

    const Real alnsg = haero::log(sigmag);

    for (int ispec = 0; ispec < nspec; ++ispec) {
      type_idx = lspectype_amode[ispec][imode] - 1;
      const Real spechygro_i = spechygro[type_idx];
      const Real specdens = specdens_amode[type_idx];

      int la, lc;
      convproc::assign_la_lc(imode, ispec, la, lc);
      const Real raer = state_q[la];
      const Real vol_tmp = raer / specdens;
      maer = maer + raer;
      dryvolmr += vol_tmp;

      // hygro currently is sum(hygro * volume) of each species,
      // need to divided by sum(volume) later to get mean hygro for all species
      hygro[imode] += vol_tmp * spechygro_i;
    } // end loop over species
    if (dryvolmr > small_value_30) {
      hygro[imode] = hygro[imode] / dryvolmr;
    } else {

      hygro[imode] = spechygro_1;
    }

    const Real v2ncur_a =
        1.0 / ((Constants::pi / 6.0) * haero::cube(dgncur_a[imode]) *
               haero::exp(4.5 * haero::square(alnsg)));
    // naer = aerosol number (#/kg)
    naer[imode] = dryvolmr * v2ncur_a;

    // compute mean (1 particle) dry volume and mass for each mode
    // old coding is replaced because the new (1/v2ncur_a) is equal to
    // the mean particle volume
    // also moletomass forces maer >= 1.0e-30, so (maer/dryvolmr)
    // should never cause problems (but check for maer < 1.0e-31 anyway)
    Real drydens;
    if (maer > small_value_31) {
      drydens = maer / dryvolmr;
    } else {
      drydens = 1.0;
    }

    // C++ porting note: these are output but defined in the module
    // thus not in the subroutine output
    dryvol[imode] = 1.0 / v2ncur_a;
    drymass[imode] = drydens * dryvol[imode];
    dryrad[imode] = haero::cbrt(dryvol[imode] / (Constants::pi * 4.0 / 3.0));
  }
}

KOKKOS_INLINE_FUNCTION
void modal_aero_water_uptake_dr_b4_wetdens(
    int nspec_amode[AeroConfig::num_modes()],
    Real specdens_amode[maxd_aspectype], Real spechygro[maxd_aspectype],
    int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
    Real state_q[nvars], Real temperature, Real pmid, Real cldn,
    Real dgncur_a[AeroConfig::num_modes()],
    Real dgncur_awet[AeroConfig::num_modes()],
    Real wetvol[AeroConfig::num_modes()], Real wtrvol[AeroConfig::num_modes()],
    Real drymass[AeroConfig::num_modes()],
    Real specdens_1[AeroConfig::num_modes()]) {

  //----------------------------------------------------------------------------
  // retreive aerosol properties

  Real hygro[AeroConfig::num_modes()];
  Real naer[AeroConfig::num_modes()];
  Real dryrad[AeroConfig::num_modes()];
  Real dryvol[AeroConfig::num_modes()];
  Real rhcrystal[AeroConfig::num_modes()];
  Real rhdeliques[AeroConfig::num_modes()];

  modal_aero_water_uptake_dryaer(nspec_amode, specdens_amode, spechygro,
                                 lspectype_amode, state_q, dgncur_a, hygro,
                                 naer, dryrad, dryvol, drymass, rhcrystal,
                                 rhdeliques, specdens_1);

  // ----------------------------------------------------------------------------
  // estimate clear air relative humidity using cloud fraction
  Real rh;
  modal_aero_water_uptake_rh_clearair(temperature, pmid, state_q[0], cldn, rh);

  //----------------------------------------------------------------------------
  // compute wet aerosol properties

  // compute aerosol wet radius, volume, diameter and aerosol water
  Real wetrad[AeroConfig::num_modes()];
  Real qaerwat[AeroConfig::num_modes()];
  modal_aero_water_uptake_wetaer(rhcrystal, rhdeliques, dgncur_a, dryrad, hygro,
                                 rh, naer, dryvol, wetrad, wetvol, wtrvol,
                                 dgncur_awet, qaerwat);
}

KOKKOS_INLINE_FUNCTION
void modal_aero_water_uptake_dr(
    int nspec_amode[AeroConfig::num_modes()],
    Real specdens_amode[maxd_aspectype], Real spechygro[maxd_aspectype],
    int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
    Real state_q[nvars], Real temperature, Real pmid, Real cldn,
    Real dgncur_a[AeroConfig::num_modes()],
    Real dgncur_awet[AeroConfig::num_modes()],
    Real wetdens[AeroConfig::num_modes()]) {

  // This function is a port modal_aero_wateruptake_dr
  // with the optional computation of the wetdensity.

  Real drymass[AeroConfig::num_modes()];
  Real specdens_1[AeroConfig::num_modes()];
  Real wetvol[AeroConfig::num_modes()];
  Real wtrvol[AeroConfig::num_modes()];

  modal_aero_water_uptake_dr_b4_wetdens(nspec_amode, specdens_amode, spechygro,
                                        lspectype_amode, state_q, temperature,
                                        pmid, cldn, dgncur_a, dgncur_awet,
                                        wetvol, wtrvol, drymass, specdens_1);

  // compute wet aerosol density
  modal_aero_water_uptake_wetdens(wetvol, wtrvol, drymass, specdens_1, wetdens);
}

KOKKOS_INLINE_FUNCTION
void modal_aero_water_uptake_dr(
    int nspec_amode[AeroConfig::num_modes()],
    Real specdens_amode[maxd_aspectype], Real spechygro[maxd_aspectype],
    int lspectype_amode[maxd_aspectype][AeroConfig::num_modes()],
    Real state_q[nvars], Real temperature, Real pmid, Real cldn,
    Real dgncur_a[AeroConfig::num_modes()],
    Real dgncur_awet[AeroConfig::num_modes()]) {

  // This function is a port modal_aero_wateruptake_dr
  // without the computation of the wetdensity.

  Real drymass[AeroConfig::num_modes()];
  Real specdens_1[AeroConfig::num_modes()];
  Real wetvol[AeroConfig::num_modes()];
  Real wtrvol[AeroConfig::num_modes()];

  modal_aero_water_uptake_dr_b4_wetdens(nspec_amode, specdens_amode, spechygro,
                                        lspectype_amode, state_q, temperature,
                                        pmid, cldn, dgncur_a, dgncur_awet,
                                        wetvol, wtrvol, drymass, specdens_1);
}

}; // namespace water_uptake

// init -- initializes the implementation with MAM4's configuration
inline void Water_Uptake::init(const AeroConfig &aero_config,
                               const Config &process_config) {

  config_ = process_config;
};

} // namespace mam4

#endif
