// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_DRYDEP_HPP
#define MAM4XX_DRYDEP_HPP

#include <haero/atmosphere.hpp>
#include <mam4xx/aero_config.hpp>

namespace mam4 {

class DryDep {

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
  const char *name() const { return "MAM4 dry deposition"; }

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config());

  // validate -- validates the given atmospheric state and prognostics against
  // assumptions made by this implementation, returning true if the states are
  // valid, false if not
  KOKKOS_INLINE_FUNCTION
  bool validate(const AeroConfig &config, const ThreadTeam &team,
                const Atmosphere &atm, const Prognostics &progs) const;

  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atm,
                          const Prognostics &progs, const Diagnostics &diags,
                          const Tendencies &tends) const;
};

namespace drydep {

//==============================================================================
// Calculate the radius for a moment of a lognormal size distribution
//==============================================================================
KOKKOS_INLINE_FUNCTION
Real radius_for_moment(const int moment, const Real sig_part,
                       const Real radius_part, const Real radius_max) {
  const Real lnsig = haero::log(sig_part);
  return haero::min(radius_max, radius_part) *
         haero::exp((moment - 1.5) * lnsig * lnsig);
}

//==========================================================================
// Calculate dynamic viscosity of air, unit [kg m-1 s-1]. See RoY94 p. 102
//==========================================================================
KOKKOS_INLINE_FUNCTION
Real air_dynamic_viscosity(const Real temp) {
  // (BAD CONSTANTS)
  return 1.72e-5 * (haero::pow(temp / 273.0, 1.5)) * 393.0 / (temp + 120.0);
}

//==========================================================================
// Calculate kinematic viscosity of air, unit [m2 s-1]
//==========================================================================
KOKKOS_INLINE_FUNCTION
Real air_kinematic_viscosity(const Real temp, const Real pres) {
  const Real vsc_dyn_atm = air_dynamic_viscosity(temp);
  const Real rho = pres / Constants::r_gas_dry_air / temp;
  return vsc_dyn_atm / rho;
}

//======================================================
// Slip correction factor [unitless].
// See, e.g., SeP97 p. 464 and Zhang L. et al. (2001),
// DOI: 10.1016/S1352-2310(00)00326-5, Eq. (3).
// ======================================================
KOKKOS_INLINE_FUNCTION
Real slip_correction_factor(const Real dyn_visc, const Real pres,
                            const Real temp, const Real particle_radius) {

  // [m]
  const Real mean_free_path =
      2.0 * dyn_visc /
      (pres * haero::sqrt(8.0 / (Constants::pi * Constants::r_gas_dry_air *
                                 temp))); // (BAD CONSTANTS)

  const Real slip_correction_factor =
      1.0 +
      mean_free_path *
          (1.257 + 0.4 * haero::exp(-1.1 * particle_radius / mean_free_path)) /
          particle_radius; // (BAD CONSTANTS)

  return slip_correction_factor;
}

//====================================================================
// Calculate the Schmidt number of air [unitless], see SeP97 p.972
//====================================================================
KOKKOS_INLINE_FUNCTION
Real schmidt_number(const Real temp, const Real pres, const Real radius,
                    const Real vsc_dyn_atm, const Real vsc_knm_atm) {

  //  slip correction factor [unitless]
  const Real slp_crc = slip_correction_factor(vsc_dyn_atm, pres, temp, radius);

  // Brownian diffusivity of particle [m2/s], see SeP97 p.474
  const Real dff_aer = Constants::boltzmann * temp * slp_crc /
                       (6.0 * Constants::pi * vsc_dyn_atm * radius);

  return vsc_knm_atm / dff_aer;
}

//=======================================================================================
// Calculate the bulk gravitational settling velocity [m s-1]
//  - using the terminal velocity of sphere falling in a fluid based on Stokes's
//    law and
//  - taking into account the influces of size distribution.
//=======================================================================================
KOKKOS_INLINE_FUNCTION
Real gravit_settling_velocity(const Real particle_radius,
                              const Real particle_density,
                              const Real slip_correction,
                              const Real dynamic_viscosity,
                              const Real particle_sig) {

  // Calculate terminal velocity following, e.g.,
  //  -  Seinfeld and Pandis (1997),  p. 466
  //  - Zhang L. et al. (2001), DOI: 10.1016/S1352-2310(00)00326-5, Eq. 2.

  const Real gravit_settling_velocity =
      (4.0 / 18.0) * particle_radius * particle_radius * particle_density *
      Constants::gravity * slip_correction / dynamic_viscosity;

  // Account for size distribution (i.e., we are calculating the bulk velocity
  // for a particle population instead of a single particle).

  const Real lnsig = haero::log(particle_sig);
  const Real dispersion = haero::exp(2.0 * lnsig * lnsig);

  return gravit_settling_velocity * dispersion;
}

} // namespace drydep

} // namespace mam4

#endif