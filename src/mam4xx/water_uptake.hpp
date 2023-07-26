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

}; // namespace water_uptake
} // namespace mam4

#endif