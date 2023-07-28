// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_WATER_UPTAKE_HPP
#define MAM4XX_WATER_UPTAKE_HPP

#include <haero/atmosphere.hpp>
#include <haero/surface.hpp>
#include <kokkos/Kokkos_Complex.hpp>
#include <mam4xx/aero_config.hpp>
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

//-----------------------------------------------------------------------
// compute aerosol wet density
//-----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void modal_aero_wateruptake_wetdens(
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
void modal_aero_wateruptake_wetaer(
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

}; // namespace water_uptake
} // namespace mam4

#endif