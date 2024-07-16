// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_DRYDEP_HPP
#define MAM4XX_DRYDEP_HPP

#include <haero/atmosphere.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/spitfire_transport.hpp>

#include <mam4xx/convproc.hpp>

namespace mam4 {

class DryDeposition {

public:
  // The value of n_land_type is taken from mozart as defined in mo_drydep.F90
  // BAD CONSTANT
  static constexpr int n_land_type = 11;

  struct Config {
    Config(){};

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;

    // BAD CONSTANT
    Real fraction_landuse[n_land_type] = {
        0.20918898065265040e-02, 0.10112323792561469e+00,
        0.19104123086831826e+00, 0.56703179010502225e+00,
        0.00000000000000000e+00, 0.42019237748858657e-01,
        0.85693761223933115e-01, 0.66234294754917442e-02,
        0.00000000000000000e+00, 0.00000000000000000e+00,
        0.43754228462347953e-02};
  };

  // aerosol_categories denotes four different
  // "categories" of aerosols. That is
  //   0 - interstitial aerosol, 0th moment (i.e., number)
  //   1 - interstitial aerosol, 3rd moment (i.e., volume/mass)
  //   2 - cloud-borne aerosol,  0th moment (i.e., number)
  //   3 - cloud-borne aerosol,  3rd moment (i.e., volume/mass)
  static constexpr int aerosol_categories = 4;

  // starting index of interstitial aerosols in state_q array.
  static constexpr int index_interstitial_aerosols = 15;

private:
  Config config_;
  Kokkos::View<Real *> rho;

  Kokkos::View<Real *> vlc_dry[AeroConfig::num_modes()][aerosol_categories];
  Real vlc_trb[AeroConfig::num_modes()][aerosol_categories];
  Kokkos::View<Real *> vlc_grv[AeroConfig::num_modes()][aerosol_categories];
  Kokkos::View<Real *> dqdt_tmp[aero_model::pcnst];

  // Computed tendenciesColumnView for
  // modal cloudborne aerosol number mixing ratios and
  // cloudborne aerosol mass mixing ratios within each mode.
  Kokkos::View<Real *> qqcw_tends[aero_model::pcnst] = {};

public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 Dry Deposition"; }

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config()) {
    config_ = process_config;
    const int nlev = mam4::nlev;
    Kokkos::resize(rho, nlev);
    Kokkos::deep_copy(rho, 0);
    for (int j = 0; j < AeroConfig::num_modes(); ++j) {
      for (int i = 0; i < aerosol_categories; ++i) {
        Kokkos::resize(vlc_dry[j][i], nlev);
        Kokkos::resize(vlc_grv[j][i], nlev);
        Kokkos::deep_copy(vlc_dry[j][i], 0);
        Kokkos::deep_copy(vlc_grv[j][i], 0);
        vlc_trb[j][i] = 0;
      }
    }
    for (int j = 0; j < aero_model::pcnst; ++j) {
      Kokkos::resize(dqdt_tmp[j], nlev);
      Kokkos::deep_copy(dqdt_tmp[j], 0);
      Kokkos::resize(qqcw_tends[j], nlev);
      Kokkos::deep_copy(qqcw_tends[j], 0);
    }
  }

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
                          const Surface &surf, const Prognostics &progs,
                          const Diagnostics &diags,
                          const Tendencies &tends) const;
};

namespace drydep {

// ##############################################################################
//  Given a coordinate xw, an interpolating polynomial ff and its derivative
//  fdot, calculate the value of the polynomial (psistar) at xin.
// ##############################################################################
KOKKOS_INLINE_FUNCTION
Real cfint2(const haero::ConstColumnView xw /*nlev+1*/,
            const Real ff[mam4::nlev + 1], const Real fdot[mam4::nlev + 1],
            const Real xin) {
  const int nlev = mam4::nlev;
  const Real xins = spitfire::median(xw[0], xin, xw[nlev]);
  int intz = -1;

  // first find the interval
  for (int kk = 0; kk < nlev; ++kk) {
    if (0 <= (xins - xw[kk]) * (xw[kk + 1] - xins)) {
      intz = kk;
    }
  }
  if (intz < 0) {
    printf(" mo_spitfire_transport: cfint2 -- interval was not found\n");
    Kokkos::abort(" mo_spitfire_transport: cfint2 -- interval was not found");
  }

  // now interpolate

  const int kk = intz;
  const Real dx = (xw[kk + 1] - xw[kk]);
  const Real ss = (ff[kk + 1] - ff[kk]) / dx;
  const Real c2 = (3 * ss - 2 * fdot[kk] - fdot[kk + 1]) / dx;
  const Real c3 = (fdot[kk] + fdot[kk + 1] - 2 * ss) / (dx * dx);
  const Real xx = (xins - xw[kk]);
  // const Real fxdot =  (3*c3*xx + 2*c2)*xx + fdot[kk];
  // const Real fxdd  = 6*c3*xx + 2*c2;
  const Real cfint = ((c3 * xx + c2) * xx + fdot[kk]) * xx + ff[kk];

  // limit the interpolant

  const Real psi1 = ff[kk] + (ff[kk + 1] - ff[kk]) * xx / dx;
  const Real psi2 =
      (kk == 0) ? ff[0]
                : ff[kk] + (ff[kk] - ff[kk - 1]) * xx / (xw[kk] - xw[kk - 1]);
  const Real psi3 = (kk == nlev - 1)
                        ? ff[nlev]
                        : ff[kk + 1] - (ff[kk + 2] - ff[kk + 1]) * (dx - xx) /
                                           (xw[kk + 2] - xw[kk + 1]);

  const Real psim = spitfire::median(psi1, psi2, psi3);
  const Real cfnew = spitfire::median(cfint, psi1, psim);
  const Real psistar = cfnew;

  return psistar;
}

// ##############################################################################
//  Calculate the derivative for the interpolating polynomial.
//  Multi column version.
// ##############################################################################
KOKKOS_INLINE_FUNCTION
void cfdotmc_pro(const haero::ConstColumnView xw /*nlev+1*/,
                 const Real ff[mam4::nlev + 1], Real fdot[mam4::nlev + 1]) {
  // clang-format off
  /*   
    Real   xw(nelv+1)   // coordinate variable
    Real   ff(nelv+1)   // value at notes
    Real fdot(nelv+1)   // derivative at nodes

  // Assumed variable distribution (staggering)
  //     xw1.....xw2......xw3......xw4......xw5.......xw6    1,nlev+1 points
  //     ff1.....ff2......ff3......ff4......ff5.......ff6    1,nlev+1 points
  //     ...sh1.......sh2......sh3......sh4......sh5....     1,nlev points
  //     ........dd2......dd3......dd4......dd5.........     2,nlev points
  //     ........ss2......ss3......ss4......ss5.........     2,nlev points
  //     .............dh2......dh3......dh4.............     2,nlev-1 points
  //     .............eh2......eh3......eh4.............     2,nlev-1 points
  //     .................ee3.......ee4.................     3,nlev-1 points
  //     .................ppl3......ppl4................     3,nlev-1 points
  //     .................ppr3......ppr4................     3,nlev-1 points
  //     .................tt3.......tt4.................     3,nlev-1 points
  //     ................fdot3.....fdot4................     3,nlev-1 points
  */
  // clang-format on
  const int nlev = mam4::nlev;
  Real delxh[nlev] = {};
  Real sh[nlev] = {};  // first divided differences between nodes
  Real ss[nlev] = {};  // first divided differences at nodes
  Real dd[nlev] = {};  // second divided differences at nodes
  Real dh[nlev] = {};  // second divided differences between nodes
  Real ee[nlev] = {};  // third divided differences at nodes
  Real eh[nlev] = {};  // third divided differences between nodes
  Real ppl[nlev] = {}; // p prime on left
  Real ppr[nlev] = {}; // p prime on right

  // -----------------
  for (int kk = 0; kk < nlev; ++kk) {
    // First divided differences between nodes
    delxh[kk] = xw[kk + 1] - xw[kk];
    sh[kk] = (ff[kk + 1] - ff[kk]) / delxh[kk];

    // First and second divided differences at nodes
    if (0 < kk) {
      dd[kk] = (sh[kk] - sh[kk - 1]) / (xw[kk + 1] - xw[kk - 1]);
      ss[kk] = spitfire::minmod(sh[kk], sh[kk - 1]);
    }
  }

  // Second and third divided diffs between nodes

  for (int kk = 1; kk < nlev - 1; ++kk) {
    eh[kk] = (dd[kk + 1] - dd[kk]) / (xw[kk + 2] - xw[kk - 1]);
    dh[kk] = spitfire::minmod(dd[kk], dd[kk + 1]);
  }

  // Treat the boundaries

  ee[1] = eh[1];
  ee[nlev - 1] = eh[nlev - 2];

  // Outside level

  fdot[0] = sh[0] - dd[1] * delxh[0] - eh[1] * delxh[0] * (xw[0] - xw[2]);
  fdot[1] = spitfire::minmod(fdot[0], 3 * sh[0]);

  fdot[nlev] = sh[nlev - 1] + dd[nlev - 1] * delxh[nlev - 1] +
               eh[nlev - 2] * delxh[nlev - 1] * (xw[nlev] - xw[nlev - 2]);
  fdot[nlev] = spitfire::minmod(fdot[nlev], 3 * sh[nlev - 1]);

  // One in from boundary

  fdot[1] = sh[0] + dd[1] * delxh[0] - eh[1] * delxh[0] * delxh[1];
  fdot[1] = spitfire::minmod(fdot[1], 3 * ss[1]);

  fdot[nlev - 1] = sh[nlev - 1] - dd[nlev - 1] * delxh[nlev - 1] -
                   eh[nlev - 1 - 1] * delxh[nlev - 1] * delxh[nlev - 1 - 1];
  fdot[nlev - 1] = spitfire::minmod(fdot[nlev - 1], 3 * ss[nlev - 1]);

  for (int kk = 2; kk < nlev - 1; ++kk) {
    ee[kk] = spitfire::minmod(eh[kk], eh[kk - 1]);
  }
  for (int kk = 2; kk < nlev - 1; ++kk) {
    // p prime at k-0.5
    ppl[kk] = sh[kk - 1] + dh[kk - 1] * delxh[kk - 1];
    // p prime at k+0.5
    ppr[kk] = sh[kk] - dh[kk] * delxh[kk];

    const Real tt = spitfire::minmod(ppl[kk], ppr[kk]);

    // derivate from parabola thru f(i,k-1), f(i,k), and f(i,k+1)
    const Real pp = sh[kk - 1] + dd[kk] * delxh[kk - 1];

    // quartic estimate of fdot
    fdot[kk] = pp - delxh[kk - 1] * delxh[kk] *
                        (eh[kk - 1] * (xw[kk + 2] - xw[kk]) +
                         eh[kk] * (xw[kk] - xw[kk - 2])) /
                        (xw[kk + 2] - xw[kk - 2]);

    // now limit it
    const Real qpl =
        sh[kk - 1] +
        delxh[kk - 1] *
            spitfire::minmod(dd[kk - 1] + ee[kk - 1] * (xw[kk] - xw[kk - 2]),
                             dd[kk] - ee[kk] * delxh[kk]);
    const Real qpr =
        sh[kk] +
        delxh[kk] *
            spitfire::minmod(dd[kk] + ee[kk] * delxh[kk - 1],
                             dd[kk + 1] + ee[kk + 1] * (xw[kk] - xw[kk + 2]));

    fdot[kk] = spitfire::median(fdot[kk], qpl, qpr);

    const Real ttlmt = spitfire::minmod(qpl, qpr);
    const Real tmin =
        haero::min(haero::min(0.0, 3 * ss[kk]), haero::min(1.5 * tt, ttlmt));
    const Real tmax =
        haero::max(haero::max(0.0, 3 * ss[kk]), haero::max(1.5 * tt, ttlmt));

    fdot[kk] = fdot[kk] + spitfire::minmod(tmin - fdot[kk], tmax - fdot[kk]);
  }
}

//===============================================================================
// Calculate tracer fluxes across cell boundaries using the 1D SPITFIRE
// (SPlit Implementation of Transport using Flux Integral REpresentation)
// algorithm of Rasch and Lawrence (1998):
//   Rasch, P. J., and M. Lawrence, Recent development in transport methods
//   at NCAR, MPI-Rep. 265, pp. 65 – 75, Max-Planck-Inst. fuer Meteorol.,
//   Hamburg, Germany, 1998.
//===============================================================================
template <typename VIEWTYPE>
KOKKOS_INLINE_FUNCTION void
getflx(const haero::ConstColumnView xw /*nlev+1*/, const VIEWTYPE phi /*nlev*/,
       const Real vel[mam4::nlev + 1], const Real deltat,
       Real flux[mam4::nlev + 1]) {
  // clang-format off
  // -----------------------------------------------------------------
  //  Assumed grid staggering:
  // 
  //  Input:
  // 
  //      xw1.......xw2.......xw3.......xw4.......xw5.......xw6
  //     vel1......vel2......vel3......vel4......vel5......vel6
  //      ....phi1......phi2.......phi3.....phi4.......phi5....
  // 
  //  Work arrays:
  // 
  //     psi1......psi2......psi3......psi4......psi5......psi6
  // 
  //  Output:
  // 
  //    flux1.....flux2.....flux3.....flux4.....flux5.....flux6
  // -----------------------------------------------------------------

  /*   
  in :: xw[nlev+1]    coordinate variable, values at layer interfaces. In EAM this is pint.
  in :: phi[nlev]     grid cell mean tracer mixing ratio
  in :: vel[nlev+1]   velocity in the xw coordinate. In EAM this is grav * rho * v
                      where v is velocity in the height (z) coordinate
  in :: deltat

  out :: flux[nlev+1]
  */

  // clang-format on

  // Set fluxes at boundaries to zero
  const int nlev = mam4::nlev;
  flux[0] = 0.0;
  flux[nlev] = 0.0;

  // Get the vertical integral of phi.
  // See Rasch and Lawrence (1998), Eq (3) but note we are using a pressure
  // coordinate here.

  Real psi[nlev + 1] = {}; // integral of phi along the xw coordinate
  for (int kk = 1; kk < nlev + 1; ++kk) {
    psi[kk] = phi[kk - 1] * (xw[kk] - xw[kk - 1]) + psi[kk - 1];
  }

  // Calculate the derivatives for the interpolating polynomial
  Real fdot[nlev + 1] = {}; // derivative of interpolating polynomial
  cfdotmc_pro(xw, psi, fdot);

  // Calculate fluxes at interior interfaces
  for (int kk = 1; kk < nlev; ++kk) {
    // Find departure point. Rasch and Lawrence (1998), Eq (4)
    const Real xxk = xw[kk] - vel[kk] * deltat;
    // Calculate the integral, psistar, at the departure point xxk.
    const Real psistar = cfint2(xw, psi, fdot, xxk);
    // Calculate the flux at interface kk. Rasch and Lawrence (1998), Eq (5)
    flux[kk] = psi[kk] - psistar;
  }
}

//-----------------------------------------------------------------------
// Numerically solve the sedimentation equation for 1 tracer
//-----------------------------------------------------------------------
template <typename VIEWTYPE>
KOKKOS_INLINE_FUNCTION Real sedimentation_solver_for_1_tracer(
    const Real dt, const Kokkos::View<Real *> sed_vel /*nlev*/,
    const VIEWTYPE qq_in /*nlev*/, const ColumnView rho /*nlev*/,
    const haero::ConstColumnView tair /*nlev*/,
    const haero::ConstColumnView pint /*nlev+1*/,
    const haero::ConstColumnView pmid /*nlev*/,
    const haero::ConstColumnView pdel /*nlev*/, ColumnView dqdt_sed /*nlev*/) {
  // clang-format off
  /*
  in :: dt
  in :: rho[nlev]       // air density [kg/m3]
  in :: tair[nlev]      // air temperature [K]
  in :: pint[nlev+1]    // air pressure at layer interfaces [Pa]
  in :: pmid[nlev]      // air pressure at layer midpoints  [Pa]
  in :: pdel[nlev]      // pressure layer thickness [Pa]
  in :: sed_vel[nlev]   // deposition velocity [m/s]
  in :: qq_in[nlev]     // tracer mixing ratio, [kg/kg] or [1/kg]

  out :: dqdt_sed[nlev] // tracer mixing ratio tendency [kg/kg/s] or [1/kg/s]
  out :: sflx           // deposition flux at the Earth's surface [kg/m2/s] or [1/m2/s]
  */
  // clang-format on
  // BAD CONSTANT
  const Real mxsedfac = 0.99; // maximum sedimentation flux factor
  const Real gravit = Constants::gravity;
  // ---------------------------------------------------------------------------------------
  //  Set sedimentation velocity to zero at the top interface of the model
  //  domain.
  const int nlev = mam4::nlev;
  Real pvmzaer[nlev + 1] = {}; // sedimentation velocity in Pa (positive = down)

  //  Assume the sedimentation velocities passed in are velocities
  //  at the bottom interface of each model layer, like an upwind scheme.
  for (int i = 1; i < nlev + 1; ++i)
    pvmzaer[i] = sed_vel[i - 1];

  //  Convert velocity from height coordinate to pressure coordinate;
  //  units: convert from meters/sec to pascals/sec.
  //  (This was referred to as "Phil's method" in the code before refactoring.)
  for (int i = 1; i < nlev + 1; ++i)
    pvmzaer[i] *= rho[i - 1] * gravit;

  // ------------------------------------------------------
  //  Calculate mass flux * dt at each layer interface
  // ------------------------------------------------------
  // dt * mass fluxes at layer interfaces (positive = down)
  Real dtmassflux[nlev + 1] = {};
  getflx(pint, qq_in, pvmzaer, dt, dtmassflux);

  // Filter out any negative fluxes from the getflx routine

  for (int kk = 1; kk < nlev; ++kk)
    dtmassflux[kk] = haero::max(0.0, dtmassflux[kk]);

  //  Set values for the upper and lower boundaries
  // no flux at model top
  dtmassflux[0] = 0;
  // surface flux by upwind scheme
  dtmassflux[nlev] = qq_in[nlev - 1] * pvmzaer[nlev] * dt;

  //  Limit the flux out of the bottom of each column:
  //  apply mxsedfac to prevent generating very small negative mixing ratio.
  //  *** Should we include the flux through the top interface, to accommodate
  //  thin surface layers?
  for (int kk = 0; kk < nlev; ++kk)
    dtmassflux[kk + 1] =
        haero::min(dtmassflux[kk + 1], mxsedfac * qq_in[kk] * pdel[kk]);

  // -----------------------------------------------------------------------
  //  Calculate the mixing ratio tendencies resulting from flux divergence
  // -----------------------------------------------------------------------
  for (int kk = 0; kk < nlev; ++kk)
    dqdt_sed[kk] = (dtmassflux[kk] - dtmassflux[kk + 1]) / (dt * pdel[kk]);
  // -----------------------------------------------------------------------
  //  Convert flux out the bottom to mass units [kg/m2/s]
  // -----------------------------------------------------------------------
  const Real sflx = dtmassflux[nlev] / (dt * gravit);
  return sflx;
}

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

KOKKOS_INLINE_FUNCTION
Real gamma(const int n_land_type) {
  // BAD CONSTANT
  const Real gamma_array[DryDeposition::n_land_type] = {
      0.56, 0.54, 0.54, 0.56, 0.56, 0.56, 0.50, 0.54, 0.54, 0.54, 0.54};
  return gamma_array[n_land_type];
}

KOKKOS_INLINE_FUNCTION
Real alpha(const int n_land_type) {
  // BAD CONSTANT
  const Real alpha_array[DryDeposition::n_land_type] = {
      1.50, 1.20, 1.20, 0.80, 1.00, 0.80, 100.0, 50.0, 2.0, 1.2, 50.0};
  return alpha_array[n_land_type];
}

KOKKOS_INLINE_FUNCTION
Real radius_collector(const int n_land_type) {
  // BAD CONSTANT
  const Real radius_collector_array[DryDeposition::n_land_type] = {
      10.0e-3, 3.5e-3, 3.5e-3,  5.1e-3, 2.0e-3, 5.0e-3,
      -1.0e0,  -1.0e0, 10.0e-3, 3.5e-3, -1.0e+0};
  return radius_collector_array[n_land_type];
}

KOKKOS_INLINE_FUNCTION
int iwet(const int n_land_type) {
  const int iwet_array[DryDeposition::n_land_type] = {-1, -1, -1, -1, -1, -1,
                                                      1,  -1, 1,  -1, -1};
  return iwet_array[n_land_type];
}

KOKKOS_INLINE_FUNCTION
void modal_aero_turb_drydep_velocity(
    const int moment, const Real fraction_landuse[DryDeposition::n_land_type],
    const Real radius_max, const Real tair, const Real pmid,
    const Real radius_part, const Real density_part, const Real sig_part,
    const Real fricvel, const Real ram1, const Real vlc_grv, Real &vlc_trb,
    Real &vlc_dry) {

  /// TODO - figure out how/where we need to resolve

  // Calculate size-INdependent thermokinetic properties of the air
  const Real vsc_dyn_atm = air_dynamic_viscosity(tair);
  const Real vsc_knm_atm = air_kinematic_viscosity(tair, pmid);

  // Calculate the mean radius and Schmidt number of the moment
  const Real radius_moment =
      radius_for_moment(moment, sig_part, radius_part, radius_max);
  const Real shm_nbr =
      schmidt_number(tair, pmid, radius_moment, vsc_dyn_atm, vsc_knm_atm);

  // Initialize deposition velocities averages over different land surface types
  Real vlc_trb_wgtsum = 0.0;
  Real vlc_dry_wgtsum = 0.0;

  // Loop over different land surface types. Calculate deposition velocities of
  // those different surface types. The overall deposition velocity of a grid
  // cell is the area-weighted average of those land-type-specific velocities.
  for (int lt = 0; lt < DryDeposition::n_land_type; ++lt) {

    const Real lnd_frc = fraction_landuse[lt];

    if (lnd_frc != 0.0) {
      //----------------------------------------------------------------------
      // Collection efficiency of deposition mechanism 1 - Brownian diffusion
      //----------------------------------------------------------------------
      const Real brownian = haero::pow(shm_nbr, (-gamma(lt)));

      //----------------------------------------------------------------------
      // Collection efficiency of deposition mechanism 2 - interception
      //----------------------------------------------------------------------
      Real interception = 0.0;
      const Real rc = radius_collector(lt);
      if (rc > 0.0) {
        // vegetated surface
        interception = 2.0 * haero::square(radius_moment / rc);
      }

      //----------------------------------------------------------------------
      // Collection efficiency of deposition mechanism 3 - impaction
      //----------------------------------------------------------------------
      Real stk_nbr = 0.0;
      if (rc > 0.0) {
        // vegetated surface
        stk_nbr = vlc_grv * fricvel / (Constants::gravity * rc);
      } else {
        // non-vegetated surface
        stk_nbr = vlc_grv * fricvel * fricvel /
                  (Constants::gravity * vsc_knm_atm); //  SeP97 p.965
      }

      static constexpr Real beta =
          2.0; // (BAD CONSTANT) empirical parameter $\beta$ in Eq. (7c) of
               // Zhang L. et al. (2001)
      const Real impaction =
          haero::pow(stk_nbr / (alpha(lt) + stk_nbr),
                     beta); // Eq. (7c) of Zhang L. et al.  (2001)

      //-----------------------------------------------------
      // Stick fraction, Eq. (10) of Zhang L. et al.  (2001)
      //-----------------------------------------------------
      Real stickfrac = 1.0;
      static constexpr Real stickfrac_lowerbnd =
          1.0e-10; // (BAD CONSTANT) lower bound of stick fraction
      if (iwet(lt) < 0) {
        stickfrac =
            haero::max(stickfrac_lowerbnd, haero::exp(-haero::sqrt(stk_nbr)));
      }

      //----------------------------------------------------------------------------------
      // Using the numbers calculated above, compute the quasi-laminar layer
      // resistance following Zhang L. et al. (2001), Eq. (5)
      //----------------------------------------------------------------------------------
      static constexpr Real eps0 =
          3.0; // (BAD CONSTANT) empirical parameter $\varepsilon_0$ in Eq. (5)
               // of Zhang L. et al. (2001)
      const Real rss_lmn = 1.0 / (eps0 * fricvel * stickfrac *
                                  (brownian + interception + impaction));

      //--------------------------------------------------------------------
      // Total resistence and deposition velocity of turbulent deposition,
      // see Eq. (21) of Zender et al. (2003)
      //--------------------------------------------------------------------
      const Real rss_trb = ram1 + rss_lmn + ram1 * rss_lmn * vlc_grv;
      const Real vlc_trb_ontype = 1.0 / rss_trb;

      //--------------------------------------------------------------------------------
      // Contributions to the single-value bulk deposition velocities of the
      // grid cell
      //--------------------------------------------------------------------------------
      vlc_trb_wgtsum += lnd_frc * (vlc_trb_ontype);
      vlc_dry_wgtsum += lnd_frc * (vlc_trb_ontype + vlc_grv);
    }
  } // lt=0,n_land_type-1

  vlc_trb = vlc_trb_wgtsum;
  vlc_dry = vlc_dry_wgtsum;
}

//==========================================================================
// Calculate particle velocity of gravitational settling
//==========================================================================
KOKKOS_INLINE_FUNCTION
Real modal_aero_gravit_settling_velocity(
    const int moment, const Real radius_max, const Real tair, const Real pmid,
    const Real radius_part, const Real density_part, const Real sig_part) {

  const Real vsc_dyn_atm = air_dynamic_viscosity(tair);

  const Real radius_moment =
      radius_for_moment(moment, sig_part, radius_part, radius_max);

  const Real slp_crc =
      slip_correction_factor(vsc_dyn_atm, pmid, tair, radius_moment);

  const Real vlc_grv = gravit_settling_velocity(radius_moment, density_part,
                                                slp_crc, vsc_dyn_atm, sig_part);
  return vlc_grv;
}

//---------------------------------------------------------------------------------
// !DESCRIPTION:
//
// Calc aerodynamic resistance over oceans and sea ice from Seinfeld and Pandis,
// p.963.
//
// Author: Natalie Mahowald
// Code refactor: Hui Wan, 2023
//---------------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void calcram(const Real landfrac, const Real icefrac, const Real ocnfrac,
             const Real obklen, const Real ustar, const Real tair,
             const Real pmid, const Real pdel, const Real ram1_in,
             const Real fv_in, Real &ram1_out, Real &fv_out) {

  static constexpr Real lndfrc_threshold =
      0.000000001; // (BAD CONSTANT) fraction, unitless
  static constexpr Real zzocn =
      0.0001; // (BAD CONSTANT) Ocean aerodynamic roughness length
  static constexpr Real zzsice =
      0.0400; // (BAD CONSTANT) Sea ice aerodynamic roughness length
  static constexpr Real xkar = 0.4; // (BAD CONSTANT) Von Karman constant
  //---------------------------------------------------------------------------
  // Friction velocity:
  //  - If the grid cell has a land fraction larger than a threshold (~zero),
  //    then use cam_in%fv.
  //  - Otherwise, use the ustar calculated in the atmosphere.
  //---------------------------------------------------------------------------
  if (landfrac > lndfrc_threshold) {
    fv_out = fv_in;
  } else {
    fv_out = ustar;
  }

  // fvitt -- fv == 0 causes a floating point exception in
  // dry dep of sea salts and dust

  if (fv_out == 0.0) {
    // BAD CONSTANT
    fv_out = 1.e-12;
  }

  //-------------------------------------------------------------------
  // Aerodynamic resistence
  //-------------------------------------------------------------------

  if (landfrac > lndfrc_threshold) {
    // If the grid cell has a land fraction larger than a threshold (~zero),
    // simply use cam_in%ram1
    ram1_out = ram1_in;
  } else {
    // If the grid cell has a land fraction smaller than the threshold,
    // calculate aerodynamic resistence

    // calculate psi, psi0, temp
    // use half the layer height like Ganzefeld and Lelieveld, 1995
    const Real zz = pdel * Constants::r_gas_dry_air * tair / pmid /
                    Constants::gravity / 2.0;

    Real psi = 0.0;
    Real psi0 = 0.0;
    if (obklen != 0.0) {
      psi = utils::min_max_bound(-1, 1, zz / obklen);
      psi0 = utils::min_max_bound(-1, 1, zzocn / obklen);
    }
    Real temp = zz / zzocn;

    // special treatment for ice-dominant cells
    if (icefrac > 0.5) {
      if (obklen > 0.0) {
        psi0 = utils::min_max_bound(-1, 1, zzsice / obklen);
      } else {
        psi0 = 0.0;
      }
      temp = zz / zzsice;
    }

    // calculate aerodynamic resistence
    if (psi > 0.0) {
      ram1_out = 1.0 / xkar / ustar * (haero::log(temp) + 4.7 * (psi - psi0));
    } else {
      const Real nu = haero::pow(1.00 - 15.000 * psi, 0.25);
      const Real nu0 = haero::pow(1.00 - 15.000 * psi0, 0.25);

      if (ustar != 0.0) {
        ram1_out =
            1.0 / xkar / ustar *
            (haero::log(temp) +
             haero::log(
                 ((haero::square(nu0) + 1.0) * haero::square(nu0 + 1.0)) /
                 ((haero::square(nu) + 1.0) * haero::square(nu + 1.0))) +
             2.0 * (haero::atan(nu) - haero::atan(nu0)));
      } else {
        ram1_out = 0.0;
      }
    }
  }
}

//==========================================================================================
// Calculate deposition velocities caused by turbulent dry deposition and
// gravitational settling of aerosol particles
//
// Reference:
//  L. Zhang, S. Gong, J. Padro, and L. Barrie:
//  A size-seggregated particle dry deposition scheme for an atmospheric aerosol
//  module Atmospheric Environment, 35, 549-560, 2001.
//
// History:
//  - Original version by X. Liu.
//  - Calculations for gravitational and turbulent dry deposition separated into
//   different subroutines by Hui Wan, 2023.
//==========================================================================================
KOKKOS_INLINE_FUNCTION
void modal_aero_depvel_part(
    const bool lowest_model_layer,
    const Real fraction_landuse[DryDeposition::n_land_type], const Real tair,
    const Real pmid, const Real ram1, const Real fricvel,
    const Real radius_part, const Real density_part, const Real sig_part,
    const int moment, Real &vlc_dry, Real &vlc_trb, Real &vlc_grv) {

  // use a maximum radius of 50 microns when calculating deposition velocity
  static constexpr Real radius_max = 50.0e-6; //(BAD CONSTANT)

  //------------------------------------------------------------------------------------
  // Calculate deposition velocity of gravitational settling in all grid layers
  //------------------------------------------------------------------------------------
  vlc_grv = modal_aero_gravit_settling_velocity(
      moment, radius_max, tair, pmid, radius_part, density_part, sig_part);

  // vlc_dry is just the gravitational settling velocity for now.
  vlc_dry = vlc_grv;

  //------------------------------------------------------------------------------------
  // For the lowest model layer:
  //  - Calculate turbulent dry deposition velocity, vlc_trb.
  //  - Add vlc_trb to vlc_grv to give the total deposition velocity, vlc_dry.
  //------------------------------------------------------------------------------------
  if (lowest_model_layer)
    modal_aero_turb_drydep_velocity(moment, fraction_landuse, radius_max, tair,
                                    pmid, radius_part, density_part, sig_part,
                                    fricvel, ram1, vlc_grv, vlc_trb, vlc_dry);
}

} // namespace drydep

// =============================================================================
//  Main subroutine of aerosol dry deposition parameterization.
//  Also serves as the interface routine called by EAM's physics driver.
// =============================================================================
KOKKOS_INLINE_FUNCTION
void aero_model_drydep(
    // inputs
    const ThreadTeam &team,
    const Real fraction_landuse[DryDeposition::n_land_type],
    const ConstColumnView tair, const ConstColumnView pmid,
    const ConstColumnView pint, const ConstColumnView pdel,
    const Diagnostics::ColumnTracerView state_q,
    const ConstColumnView dgncur_awet[AeroConfig::num_modes()],
    const ConstColumnView wetdens[AeroConfig::num_modes()], const Real obklen,
    const Real ustar, const Real landfrac, const Real icefrac,
    const Real ocnfrac, const Real fricvelin, const Real ram1in, const Real dt,
    // Input-outputs
    const ColumnView qqcw[aero_model::pcnst],
    // outputs
    const Diagnostics::ColumnTracerView ptend_q,
    bool ptend_lq[aero_model::pcnst], const ColumnView aerdepdrycw,
    const ColumnView aerdepdryis,
    // work arrays
    const ColumnView rho,
    const ColumnView vlc_dry[AeroConfig::num_modes()]
                            [DryDeposition::aerosol_categories],
    Real vlc_trb[AeroConfig::num_modes()][DryDeposition::aerosol_categories],
    const ColumnView vlc_grv[AeroConfig::num_modes()]
                            [DryDeposition::aerosol_categories],
    const ColumnView dqdt_tmp[aero_model::pcnst]) {
  // clang-format off
  /*   
    // Arguments
    in :: tair(nlev)        : air temperture [k]
    in :: pmid(nlev)        : air pressure at layer midpoint [Pa]
    in :: pint(1+nlev)      : air pressure at layer interface [Pa]
    in :: pdel(nlev)        : layer thickness [Pa]
    in :: state_q(nlev)     : mixing ratios [kg/kg or 1/kg]
    in :: dgncur_awet(nlev) : geometric mean wet diameter for number distribution [m]
    in :: wetdens(nlev)     : wet density of interstitial aerosol [kg/m3]
    in :: obklen            : Obukhov length [m]
    in :: ustar             : sfc friction velocity [m/s]
    in :: landfrac          : land fraction [unitless]
    in :: icefrac           : ice fraction [unitless]
    in :: ocnfrac           : ocean fraction [unitless]
    in :: fricvelin         : friction velocity from land model [m/s]
    in :: ram1in            : aerodynamical resistance from land model [s/m]
    in :: dt                : time step [s]
    inout  :: qqcw(nlev)    : Cloud borne aerosols mixing ratios [kg/kg or 1/kg]
    out    :: ptend_q (nlev, aero_model::pcnst) : diagnostics.d_tracer_mixing_ratio_dt

    // Scratch Space
    rho(nlev)               : air density [kg/m3]
    dqdt_tmp(nlev)          : temporary array to hold tendency for 1 species, [kg/kg/s] or [1/kg/s]

      Deposition velocities. The last dimension (size = 4) corresponds to the
      two attachment states and two moments:
        0 - interstitial aerosol, 0th moment (i.e., number)
        1 - interstitial aerosol, 3rd moment (i.e., volume/mass)
        2 - cloud-borne aerosol,  0th moment (i.e., number)
        3 - cloud-borne aerosol,  3rd moment (i.e., volume/mass)
    vlc_grv(nlev)     : dep velocity of gravitational settling [m/s]
    vlc_trb     : dep velocity of turbulent dry deposition [m/s]
    vlc_dry(nlev)     : dep velocity, sum of vlc_grv and vlc_trb [m/s]
  */  
  // clang-format on 
  auto printb = [](const std::string &name, const double &val) {
    //std::cout<<name<<":"<<std::setprecision (15)<<val<<std::endl;
  };
  const int nlev = mam4::nlev;
  // Calculate rho:
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, nlev), KOKKOS_LAMBDA(int kk) {
        const Real rair = Constants::r_gas_dry_air;
        rho[kk] = pmid[kk] / (rair * tair[kk]);
      });
  static constexpr int kprnt=63;
  printb("RHO",rho[kprnt]);
  // --------------------------------------------------------------------------------
  //  For turbulent dry deposition: calculate ram and fricvel over ocean and sea
  //  ice; copy values over land
  // --------------------------------------------------------------------------------
  // ram1    : aerodynamical resistance used in the calculaiton of turbulent dry deposition velocity [s/m]
  // fricvel : friction velocity used in the calculaiton of turbulent dry deposition velocity [m/s]
  Real ram1=0, fricvel=0;
  drydep::calcram(landfrac,          // in: cam_in%landfrac
                  icefrac,           // in: cam_in%icefrac
                  ocnfrac,           // in: cam_in%ocnfrac
                  obklen,            // in: calculated in tphysac
                  ustar,             // in: calculated in tphysac
                  tair[nlev-1],      // in. note: bottom level only
                  pmid[nlev-1],      // in. note: bottom level only
                  pdel[nlev-1],      // in. note: bottom level only
                  ram1in,            // in: cam_in%ram1
                  fricvelin,         // in: cam_in%fv
                  ram1,              //  out: aerodynamical resistance (s/m)
                  fricvel            //  out: bulk friction velocity of a grid cell
  );

    printb("ram1",ram1);
  printb("fricvel",fricvel);
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, nlev), KOKKOS_LAMBDA(int kk) {
        // imnt  : moment of the aerosol size distribution. 0 = number; 3 = volume
        int imnt = -1; 
        // jvlc  : index for last dimension of vlc_xxx arrays
        //   0 - interstitial aerosol, 0th moment (i.e., number)
        //   1 - interstitial aerosol, 3rd moment (i.e., volume/mass)
        //   2 - cloud-borne aerosol,  0th moment (i.e., number)
        //   3 - cloud-borne aerosol,  3rd moment (i.e., volume/mass)
        int jvlc = -1;
        // ======================
        //  cloud-borne aerosols
        // ---------------------------------------------------------------------------------------
        //  Calculate gravitational settling and dry deposition velocities for
        //  cloud droplets (and hence the cloud-borne aerosols therein). There
        //  is one set of velocities for number mixing ratios of all aerosol
        //  modes and one set of velocities for all mass mixing ratios of all
        //  modes.
        // ---------------------------------------------------------------------------------------
        //  *** mean drop radius should eventually be computed from ndrop and
        //  qcldwtr

        // rad_drop : cloud droplet radius [m]
        // BAD CONSTANT
        const Real rad_drop = 5.0e-6;
        const Real rhoh2o = haero::Constants::density_h2o;
        // dens_drop  : cloud droplet density [kg/m3]
        const Real dens_drop = rhoh2o;
        // sg_drop  : assumed geometric standard deviation of droplet size distribution
        const Real sg_drop = 1.46; // (BAD CONSTANTS)

        // moment of the aerosol size distribution. 0 = number; 3 = volume
        imnt = 0; // cloud-borne aerosol number
        // index for last dimension of vlc_xxx arrays
        jvlc = 2;
        const bool lowest_model_layer = (kk == nlev-1);
        drydep::modal_aero_depvel_part(lowest_model_layer,
	                               fraction_landuse, tair[kk], pmid[kk],
                                       ram1,
                                       fricvel, // in
                                       rad_drop, dens_drop, sg_drop,
                                       imnt,              // in
                                       vlc_dry[0][jvlc][kk],  // out
                                       vlc_trb[0][jvlc],  // out
                                       vlc_grv[0][jvlc][kk]); // out

        // moment of the aerosol size distribution. 0 = number; 3 = volume
        imnt = 3; // cloud-borne aerosol volume/mass
        // index for last dimension of vlc_xxx arrays
        jvlc = 3;
        drydep::modal_aero_depvel_part(lowest_model_layer,
	                               fraction_landuse, tair[kk], pmid[kk],
                                       ram1,
                                       fricvel, // in
                                       rad_drop, dens_drop, sg_drop,
                                       imnt,                // in
                                       vlc_dry[0][jvlc][kk],  // out
                                       vlc_trb[0][jvlc],  // out
                                       vlc_grv[0][jvlc][kk]); // out
      });
  team.team_barrier();
  printb("vlc_trb[0][2]:", vlc_trb[0][2]);
  printb("vlc_trb[0][3]:", vlc_trb[0][3]);
  printb("vlc_grv[0][2][kk]:", vlc_grv[0][2][kprnt]);
  printb("vlc_grv[0][3][kk]:", vlc_grv[0][3][kprnt]);
  printb("vlc_dry[0][2][kk]:", vlc_dry[0][2][kprnt]);
  printb("vlc_dry[0][3][kk]:", vlc_dry[0][3][kprnt]);
  // ----------------------------------------------------------------------------------
  //  Loop over all modes and all aerosol tracers (number + mass species).
  //  Calculate the drydep-induced tendencies, then update the mixing ratios.
  // ----------------------------------------------------------------------------------
  static constexpr int ntot_amode = AeroConfig::num_modes();
  // The number of species is currently 7:
  const int max_species = static_cast<int>(AeroId::None);
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, ntot_amode * (1 + max_species)),
      KOKKOS_LAMBDA(int ic) {
        const int imode = ic / (1 + max_species);
        const int lspec = ic % (1 + max_species) - 1;
        const int icnst = (lspec == -1)
                              ? ConvProc::numptrcw_amode(imode)
                              : ConvProc::lmassptrcw_amode(lspec, imode);
	const int jvlc = (lspec == -1) ? 2 : 3;
        if (-1 < icnst) {
          // qq : mixing ratio of a single tracer [kg/kg] or [1/kg]
          auto qq = qqcw[icnst];
          printb("qq_bef:",qq[kprnt]);
          // sflx : surface deposition flux of a single species [kg/m2/s] or [1/m2/s]
          const Real sflx = drydep::sedimentation_solver_for_1_tracer(
              dt, vlc_dry[0][jvlc], qq, rho, tair, pint, pmid, pdel, //input
              dqdt_tmp[ic]); //output (Note:sflx is also an output)
          // aerdepdrycw  : surface deposition flux of cloud-borne  aerosols, [kg/m2/s] or [1/m2/s]
          aerdepdrycw[icnst] = sflx;
          // Update mixing ratios here. Recall that mixing ratios of cloud-borne
          // aerosols are stored in pbuf, not as part of the state variable
          for (int klev = 0; klev < nlev; ++klev)
            qq[klev] += dqdt_tmp[ic][klev] * dt;
        printb("aerdepdrycw:",aerdepdrycw[icnst]);
        printb("qq:",qq[kprnt]);
        printb("dqtmp:",dqdt_tmp[ic][kprnt]);
        }
      }); //parallel_for for ic (constituents)


  // =====================
  //  interstial aerosols
  // =====================
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, nlev), KOKKOS_LAMBDA(int kk) {
        // imnt  : moment of the aerosol size distribution. 0 = number; 3 = volume
        int imnt = -1; 
        // jvlc  : index for last dimension of vlc_xxx arrays
        //   0 - interstitial aerosol, 0th moment (i.e., number)
        //   1 - interstitial aerosol, 3rd moment (i.e., volume/mass)
        //   2 - cloud-borne aerosol,  0th moment (i.e., number)
        //   3 - cloud-borne aerosol,  3rd moment (i.e., volume/mass)
        int jvlc = -1;
        // loop over aerosol modes
        for (int imode = 0; imode < ntot_amode; ++imode) {
          // -----------------------------------------------------------------
          //  Calculate gravitational settling and dry deposition velocities for
          //  interstitial aerosol particles in a single lognormal mode. Note:
          //   One set of velocities for number mixing ratio of the mode;
          //   One set of velocities for all mass mixing ratios of the mode.
          // -----------------------------------------------------------------
          // sigmag_amode : assumed geometric standard deviation of particle size distribution
          const Real sigmag_amode = modes(imode).mean_std_dev;

          const Real alnsg_amode = haero::log(sigmag_amode);
          // rad_aer  // volume mean wet radius of interstitial aerosols [m]
          const Real rad_aer = 0.5 * dgncur_awet[imode][kk] *
                               haero::exp(1.5 * haero::square(alnsg_amode));
          // dens_aer : wet density of interstitial aerosols [kg/m3]
          const Real dens_aer = wetdens[imode][kk];

          imnt = 0; // interstitial aerosol number
          jvlc = 0;
          const bool lowest_model_layer = (kk == nlev-1);
          drydep::modal_aero_depvel_part( lowest_model_layer,
	      fraction_landuse, tair[kk], pmid[kk], ram1, fricvel, // in
              rad_aer, dens_aer, sigmag_amode, imnt,         // in
              vlc_dry[imode][jvlc][kk],                      // out
              vlc_trb[imode][jvlc],                          // out
              vlc_grv[imode][jvlc][kk]);                     // out
        if(kk==71){
          printb("tair:",tair[kk]);
          printb("pmid:",pmid[kk]);
          printb("ram1:",ram1);
          printb("fricvel:", fricvel);
          printb("rad_aer:",rad_aer);
           printb("dens_aer:",dens_aer);
           printb("sigmag_amode:",sigmag_amode);
          printb("---->vlc_trb[0][0]:", vlc_trb[imode][0]);
        }
          imnt = 3; // interstitial aerosol volume/mass
          jvlc = 1;
          drydep::modal_aero_depvel_part(lowest_model_layer,
	      fraction_landuse, tair[kk], pmid[kk], ram1, fricvel, // in
              rad_aer, dens_aer, sigmag_amode, imnt,         // in
              vlc_dry[imode][jvlc][kk],                      // out
              vlc_trb[imode][jvlc],                          // out
              vlc_grv[imode][jvlc][kk]);                     // out
        }
      });
  team.team_barrier();
  printb("->vlc_trb[0][0]:", vlc_trb[0][0]);
  printb("->vlc_trb[0][1]:", vlc_trb[0][1]);
  printb("->vlc_grv[0][0][kk]:", vlc_grv[0][0][kprnt]);
  printb("->vlc_grv[0][1][kk]:", vlc_grv[0][1][kprnt]);
  printb("->vlc_dry[0][0][kk]:", vlc_dry[0][0][kprnt]);
  printb("->vlc_dry[0][1][kk]:", vlc_dry[0][1][kprnt]);
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, ntot_amode * (1 + max_species)),
      KOKKOS_LAMBDA(int kk) {
        // -----------------------------------------------------------
        //  Loop over number + mass species of the mode.
        //  Calculate drydep-induced tendencies
        // -----------------------------------------------------------
        const int imode = kk / (1 + max_species);
        const int lspec = kk % (1 + max_species) - 1;
        const int icnst = (lspec == -1)
                              ? ConvProc::numptrcw_amode(imode)
                              : ConvProc::lmassptrcw_amode(lspec, imode);
        
	const int jvlc = (lspec == -1) ? 0 : 1;
        if (-1 < icnst) {
          auto qq = Kokkos::subview(state_q, Kokkos::ALL(), icnst);
          // sflx : surface deposition flux of a single species [kg/m2/s] or [1/m2/s]
          const Real sflx = drydep::sedimentation_solver_for_1_tracer(
              dt, vlc_dry[imode][jvlc], qq,  // in
              rho, tair, pint, pmid, pdel, // in
              dqdt_tmp[kk]);               // out
          // aerdepdryis  : surface deposition flux of interstitial aerosols, [kg/m2/s] or [1/m2/s]
          aerdepdryis[icnst] = sflx;
          ptend_lq[icnst] = true;
          for (int i = 0; i < nlev; ++i)
            ptend_q(i,icnst) = dqdt_tmp[kk][i];
        printb("aerdepdryis:",aerdepdryis[icnst]);
        printb("ptend_q:",ptend_q(kprnt,icnst));
        printb("dqtmp:",dqdt_tmp[kk][kprnt]);

        }
      }); 
}
// compute_tendencies -- computes tendencies and updates diagnostics
// NOTE: that both diags and tends are const below--this means their views
// NOTE: are fixed, but the data in those views is allowed to vary.
KOKKOS_INLINE_FUNCTION
void DryDeposition::compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                        Real t, Real dt, const Atmosphere &atm, const Surface &surf,
                        const Prognostics &progs, const Diagnostics &diags,
                        const Tendencies &tends) const {
  // Time tendency of tracer mixing ratio (TMR) [kg/kg/s]

  auto fraction_landuse = this->config_.fraction_landuse;
  auto tair = atm.temperature;
  auto pmid = atm.pressure;
  auto pint = atm.interface_pressure;
  auto pdel = atm.hydrostatic_dp;
  auto state_q = diags.tracer_mixing_ratio; 
  auto dgncur_awet = diags.wet_geometric_mean_diameter_i;
  auto wetdens = diags.wet_density;
  static constexpr int ntot_amode = AeroConfig::num_modes();
  const int num_aerosol = AeroConfig::num_aerosol_ids();
  // Extract Prognostics
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, nlev), KOKKOS_LAMBDA(int kk) {
    for (int m=0; m<ntot_amode; ++m) {
      qqcw_tends[ConvProc::numptrcw_amode(m)][kk] = progs.n_mode_c[m][kk];
      for (int a=0; a<num_aerosol; ++a) 
        if (-1 < ConvProc::lmassptrcw_amode(a,m))
          qqcw_tends[ConvProc::lmassptrcw_amode(a,m)][kk] = progs.q_aero_c[m][a][kk];
    }
  });
  const Real obklen = diags.Obukhov_length;
  const Real ustar = diags.surface_friction_velocty;
  const Real landfrac = diags.land_fraction;
  const Real icefrac = diags.ice_fraction;
  const Real ocnfrac = diags.ocean_fraction;
  const Real fricvelin = diags.friction_velocity;
  const Real ram1in = diags.aerodynamical_resistance;
  auto ptend_q = diags.d_tracer_mixing_ratio_dt;
  bool ptend_lq[aero_model::pcnst];
  auto aerdepdrycw = diags.deposition_flux_of_cloud_borne_aerosols;
  auto aerdepdryis = diags.deposition_flux_of_interstitial_aerosols;
  auto rho     = this->rho;
  auto vlc_dry = this->vlc_dry;
  auto vlc_trb = this->vlc_trb;
  auto vlc_grv = this->vlc_grv;
  auto dqdt_tmp= this->dqdt_tmp;

  /*mam4::aero_model_drydep(
     team, fraction_landuse, tair, pmid, pint, pdel, state_q,
     dgncur_awet, wetdens, qqcw_tends, obklen, ustar, landfrac, icefrac,
     ocnfrac, fricvelin, ram1in, ptend_q, ptend_lq, dt, aerdepdrycw,
     aerdepdryis, rho, vlc_dry, vlc_trb, vlc_grv, dqdt_tmp);

  // Update Tendencies
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, nlev), KOKKOS_LAMBDA(int kk) {
    for (int m=0; m<ntot_amode; ++m) {
      tends.n_mode_c[m][kk] = qqcw_tends[ConvProc::numptrcw_amode(m)][kk]/dt; 
      for (int a=0; a<num_aerosol; ++a) 
        if (-1 < ConvProc::lmassptrcw_amode(a,m))
          tends.q_aero_c[m][a][kk] = qqcw_tends[ConvProc::lmassptrcw_amode(a,m)][kk] /dt;
    }
  });*/
}
} // namespace mam4

#endif
