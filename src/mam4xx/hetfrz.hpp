// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_HETFZR_HPP
#define MAM4XX_HETFZR_HPP

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>
#include <mam4xx/wv_sat_methods.hpp>

namespace mam4 {

class Hetfrz {

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
  const char *name() const { return "MAM4 heterogeneous freezing"; }

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config());

  static constexpr Real amu = 1.66053886e-27; // (BAD CONSTANT)

  // frequ. of vibration [s-1] higher freq. (as in P&K, consistent with Anupam's
  // data)
  static constexpr Real nus = 1.0e13; // (BAD CONSTANT)

  // 98% RH in mixed-phase clouds (Korolev & Isaac, JAS 2006)
  static constexpr Real rhwincloud = 0.98; // (BAD CONSTANT)

  // max. ice nucleating fraction soot
  static constexpr Real limfacbc = 0.01; // (BAD CONSTANT)

  static constexpr Real mincld =
      0.0001; // Do we need ot read this (BAD CONSTANT)

  // Wang et al., 2014 fitting parameters
  // freezing parameters for immersion freezing
  static constexpr Real theta_imm_bc =
      48.0; // contact angle [deg], converted to rad later !DeMott et al (1990)
            // (BAD CONSTANT)
  static constexpr Real dga_imm_bc =
      14.15E-20; // activation energy [J] (BAD CONSTANT)
  static constexpr Real theta_imm_dust =
      46.0; // contact angle [deg], converted to rad later !DeMott et al (2011)
            // (BAD CONSTANT) SD
  static constexpr Real dga_imm_dust =
      14.75E-20; // activation energy [J] (BAD CONSTANT)

  // freezing parameters for deposition nucleation
  static constexpr Real theta_dep_dust =
      20.0; // contact angle [deg], converted to rad later !Koehler et al (2010)
            // (BAD CONSTANT) SD
  static constexpr Real dga_dep_dust =
      -8.1e-21; // activation energy [J] (BAD CONSTANT)
  static constexpr Real theta_dep_bc =
      28.0; // contact angle [deg], converted to rad later !Moehler et al
            // (2005), soot (BAD CONSTANT)
  static constexpr Real dga_dep_bc =
      -2.e-19; // activation energy [J] (BAD CONSTANT)

  static constexpr Real num_m3_to_cm3 =
      1.0e-6; // volume unit conversion, #/m^3 to #/cm^3
  static constexpr Real num_cm3_to_m3 =
      1.0e6; // volume unit conversion, #/cm^3 to #/m^3
  static constexpr Real frz_cm3_to_m3 =
      1.0e6; // het. frz. unit conversion [cm^-3 s^-1 to m^-3 s^-1]

  // Index ids and number of species involved in heterogenous freezing
  static constexpr int id_bc = 0;
  static constexpr int id_dst1 = 1;
  static constexpr int id_dst3 = 2;
  static constexpr int hetfrz_aer_nspec = 3;

  // parameters for PDF theta model
  static constexpr int pdf_n_theta = 301;
  static constexpr int itheta_bin_beg = 52;
  static constexpr int itheta_bin_end = 112;
  static constexpr Real pdf_d_theta =
      (179. - 1.) / 180. * Constants::pi / (pdf_n_theta - 1);

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

namespace hetfrz {

KOKKOS_INLINE_FUNCTION
Real get_air_viscosity(const Real tc) {
  // tc temperature [deg C]
  // dynamic viscosity of air [kg/m/s]

  const Real coeff_a = -1.2e-5;
  const Real coeff_b = 0.0049;
  const Real coeff_c = 1.718;

  return (coeff_c + coeff_b * tc + coeff_a * tc * tc) * 1.e-5;
}

KOKKOS_INLINE_FUNCTION
Real get_latent_heat_vapor(const Real tc) {
  // tc temperature [deg C]
  // latent heat of vaporization [J/kg]

  const Real coeff_a = -0.0000614342;
  const Real coeff_b = 0.00158927;
  const Real coeff_c = -2.36418;
  const Real coeff_d = 2500.79;

  const Real latvap =
      1000 * (coeff_a * haero::cube(tc) + coeff_b * haero::square(tc) +
              coeff_c * tc + coeff_d);

  return latvap;
}

KOKKOS_INLINE_FUNCTION
Real get_reynolds_num(const Real r3lx, const Real rho_air,
                      const Real viscos_air) {

  const Real coeff_vlc_a = 8.8462e2;
  const Real coeff_vlc_b = 9.7593e7;
  const Real coeff_vlc_c = -3.4249e-11;
  const Real coeff_adj_a = 3.1250e-1;
  const Real coeff_adj_b = 1.0552e-3;
  const Real coeff_adj_c = -2.4023;

  // droplet terminal velocity after Chen & Liu, QJRMS 2004
  const Real vlc_drop_adjfunc = haero::exp(
      haero::exp(coeff_adj_a + coeff_adj_b * haero::cube(haero::log(r3lx)) +
                 coeff_adj_c * haero::pow(rho_air, 1.5)));
  const Real vlc_drop =
      (coeff_vlc_a + (coeff_vlc_b + coeff_vlc_c * r3lx) * r3lx) * r3lx *
      vlc_drop_adjfunc;

  // Reynolds number
  const Real Re = 2 * vlc_drop * r3lx * rho_air / viscos_air;

  return Re;
}

KOKKOS_INLINE_FUNCTION
Real get_temperature_diff(const Real temperature, const Real pressure,
                          const Real eswtr, const Real latvap,
                          const Real Ktherm_air) {

  // water vapor diffusivity: Pruppacher & Klett 13-3
  const Real Dvap = 0.211e-4 * (temperature / 273.15) * (101325. / pressure);

  const Real rhoh2o = Constants::density_h2o;
  const Real rh2o =
      461.50463982015992; // (BAD CONSTANT) water vapor gas constant

  // G-factor = rhoh2o*Xi in Rogers & Yau, p. 104

  const Real G_factor = rhoh2o / ((latvap / (rh2o * temperature) - 1) * latvap *
                                      rhoh2o / (Ktherm_air * temperature) +
                                  rhoh2o * rh2o * temperature / (Dvap * eswtr));

  // calculate T-Tc as in Cotton et al.
  const Real Tdiff =
      -G_factor * (Hetfrz::rhwincloud - 1.) * latvap / Ktherm_air;

  return Tdiff;
}

KOKKOS_INLINE_FUNCTION
void calculate_collkernel_sub(const Real temperature, const Real pressure,
                              const Real rho_air, const Real r3lx,
                              const Real r_a, const Real lambda,
                              const Real latvap, const Real viscos_air,
                              const Real Re, const Real Ktherm_air,
                              const Real Ktherm, const Real Pr, Real Tdiff,
                              Real &K_total) {

  // Note for C++ port: Due to BFB for Fortran code, we have to declare
  // Boltzmann constant here again. This value is only used in this
  // subroutine.
  constexpr Real kboltz2 = 1.38065e-23;                // BAD CONSTANT
  constexpr Real pi = 3.1415926535897931;              // BAD CONSTANT
  constexpr Real r_gas_h2o_vapor = 461.50463982015992; // BAD CONSTANT
  // Knudsen number (Seinfeld & Pandis 8.1)
  const Real Kn = lambda / r_a;

  // aerosol diffusivity
  const Real Daer =
      kboltz2 * temperature * (1 + Kn) / (6.0 * pi * r_a * viscos_air);

  // Schmidt number
  const Real Sc = viscos_air / (Daer * rho_air);
  //  Young (1974) first equ. on page 771
  // Todo (the 0.33 power here should probably be 1/3, but that wouldn't be
  // BFB.)

  const Real K_brownian = 4.0 * pi * r3lx * Daer *
                          (1.0 + 0.3 * haero::sqrt(Re) * haero::pow(Sc, 0.33));

  // form factor
  const Real f_t =
      0.4 * (1.0 + 1.45 * Kn + 0.4 * Kn * haero::exp(-1. / Kn)) *
      (Ktherm_air + 2.5 * Kn * Ktherm) /
      ((1.0 + 3.0 * Kn) * (2.0 * Ktherm_air + 5.0 * Kn * Ktherm + Ktherm));
  const Real Q_heat = Ktherm_air / r3lx *
                      (1.0 + 0.3 * haero::sqrt(Re) * haero::pow(Pr, 0.33)) *
                      Tdiff;
  const Real K_thermo_cotton =
      4.0 * pi * haero::square(r3lx) * f_t * Q_heat / pressure;
  const Real K_diffusio_cotton =
      -(1.0 / f_t) * (r_gas_h2o_vapor * temperature / latvap) * K_thermo_cotton;

  K_total = 1.e6 * (K_brownian + K_thermo_cotton +
                    K_diffusio_cotton); // convert m3/s -> cm3/s
  K_total = haero::max(0.0, K_total);
}

KOKKOS_INLINE_FUNCTION
void collkernel(const Real temperature, const Real pressure, const Real eswtr,
                const Real r3lx, const Real r_bc, const Real r_dust_a1,
                const Real r_dust_a3, Real &Kcoll_bc, Real &Kcoll_dust_a1,
                Real &Kcoll_dust_a3) {

  // thermal conductivities from Seinfeld & Pandis, Table 8.6
  constexpr Real Ktherm_bc = 4.2;    // Carbon (BAD CONSTANT)
  constexpr Real Ktherm_dust = 0.72; // Clay (BAD CONSTANT)

  const Real tc = temperature - Constants::freezing_pt_h2o;

  // air viscosity for tc<0,
  const Real viscos_air = get_air_viscosity(tc);
  const Real r_gas_dry_air = 287.04231136504870; // (BAD CONSTANT)
  // air density
  const Real rho_air = pressure / (r_gas_dry_air * temperature);

  // mean free path: Seinfeld & Pandis 8.6
  const Real lambda =
      2.0 * viscos_air /
      (pressure *
       haero::sqrt(8.0 / (Constants::pi * r_gas_dry_air * temperature)));

  // latent heat of vaporization, varies with T
  const Real latvap = get_latent_heat_vapor(tc);

  // Reynolds number
  const Real Re = get_reynolds_num(r3lx, rho_air, viscos_air);

  // thermal conductivity of air: Seinfeld & Pandis eq. 15.75
  const Real Ktherm_air = 1.0e-3 * (4.39 + 0.071 * temperature); // J/(m s K)

  // Prandtl number
  const Real Pr = viscos_air * Constants::cp_dry_air / Ktherm_air;

  // calculate T-Tc as in Cotton et al.
  const Real Tdiff =
      get_temperature_diff(temperature, pressure, eswtr, latvap, Ktherm_air);

  calculate_collkernel_sub(temperature, pressure, rho_air, r3lx, r_bc, lambda,
                           latvap, viscos_air, Re, Ktherm_air, Ktherm_bc, Pr,
                           Tdiff, Kcoll_bc);

  calculate_collkernel_sub(temperature, pressure, rho_air, r3lx, r_dust_a1,
                           lambda, latvap, viscos_air, Re, Ktherm_air,
                           Ktherm_dust, Pr, Tdiff, Kcoll_dust_a1);

  calculate_collkernel_sub(temperature, pressure, rho_air, r3lx, r_dust_a3,
                           lambda, latvap, viscos_air, Re, Ktherm_air,
                           Ktherm_dust, Pr, Tdiff, Kcoll_dust_a3);
}

KOKKOS_INLINE_FUNCTION
Real get_form_factor(const Real alpha) {
  const Real v_cos = haero::cos(alpha);
  return (2.0 + v_cos) * haero::square(1.0 - v_cos) / 4.0;
}

KOKKOS_INLINE_FUNCTION
Real get_dg0imm(const Real sigma_iw, const Real rgimm) {
  return 4.0 * Constants::pi / 3.0 * sigma_iw * haero::square(rgimm);
}

KOKKOS_INLINE_FUNCTION
Real get_Aimm(const Real vwice, const Real rgimm, const Real temperature,
              const Real dg0imm) {
  constexpr Real n1 = 1e19; // number of water molecules in contact with unit
                            // area of substrate [m-2] (BAD CONSTANT)
  constexpr Real hplanck = 6.63e-34; // Planck constant (BAD CONSTANT)
  constexpr Real rhplanck = 1.0 / hplanck;

  constexpr Real bad_boltzmann = 1.38e-23; // (BAD CONSTANT)

  return n1 * ((vwice * rhplanck) / haero::cube(rgimm) *
               haero::sqrt(3.0 / Constants::pi * bad_boltzmann * temperature *
                           dg0imm));
}

KOKKOS_INLINE_FUNCTION
void calculate_hetfrz_contact_nucleation(
    const Real deltat, const Real temperature,
    Real uncoated_aer_num[Hetfrz::hetfrz_aer_nspec], const Real icnlx,
    const Real sigma_iv, const Real eswtr, const Real rgimm, const Real r_bc,
    const Real r_dust_a1, const Real r_dust_a3, const Real Kcoll_bc,
    const Real Kcoll_dust_a1, const Real Kcoll_dust_a3, const bool do_bc,
    const bool do_dst1, const bool do_dst3, Real &frzbccnt, Real &frzducnt) {

  frzbccnt = 0.0;
  frzducnt = 0.0;

  // form factor
  const Real f_cnt_bc =
      get_form_factor(Hetfrz::theta_dep_bc * Constants::pi / 180.0);
  const Real f_cnt_dust_a1 =
      get_form_factor(Hetfrz::theta_dep_dust * Constants::pi / 180.0);
  const Real f_cnt_dust_a3 =
      get_form_factor(Hetfrz::theta_dep_dust * Constants::pi / 180.0);

  // homogeneous energy of germ formation
  const Real dg0cnt =
      4.0 * Constants::pi / 3.0 * sigma_iv * haero::square(rgimm);

  // prefactor
  // attention: division of small numbers
  constexpr Real bad_boltzmann = 1.38e-23; // Boltzmann Constant (BAD CONSTANT)
  constexpr Real mwh2o = Constants::molec_weight_h2o *
                         1000.0; // Molecular weight water (BAD CONSTANT)
  const Real Acnt =
      Hetfrz::rhwincloud * eswtr * 4.0 * Constants::pi /
      (Hetfrz::nus * haero::sqrt(2.0 * Constants::pi * mwh2o * Hetfrz::amu *
                                 bad_boltzmann * temperature));

  // nucleation rate per particle
  const Real Jcnt_bc = Acnt * haero::square(r_bc) *
                       haero::exp((-Hetfrz::dga_dep_bc - f_cnt_bc * dg0cnt) /
                                  (bad_boltzmann * temperature)) *
                       Kcoll_bc * icnlx;

  const Real Jcnt_dust_a1 =
      Acnt * haero::square(r_dust_a1) *
      haero::exp((-Hetfrz::dga_dep_dust - f_cnt_dust_a1 * dg0cnt) /
                 (bad_boltzmann * temperature)) *
      Kcoll_dust_a1 * icnlx;

  const Real Jcnt_dust_a3 =
      Acnt * haero::square(r_dust_a3) *
      haero::exp((-Hetfrz::dga_dep_dust - f_cnt_dust_a3 * dg0cnt) /
                 (bad_boltzmann * temperature)) *
      Kcoll_dust_a3 * icnlx;

  // Limit to 1% of available potential IN (for BC), no limit for dust
  if (do_bc) {
    frzbccnt =
        frzbccnt +
        haero::min(Hetfrz::limfacbc * uncoated_aer_num[Hetfrz::id_bc] / deltat,
                   uncoated_aer_num[Hetfrz::id_bc] / deltat *
                       (1.0 - haero::exp(-Jcnt_bc * deltat)));
  }

  if (do_dst1) {
    frzducnt =
        frzducnt + haero::min(uncoated_aer_num[Hetfrz::id_dst1] / deltat,
                              uncoated_aer_num[Hetfrz::id_dst1] / deltat *
                                  (1.0 - haero::exp(-Jcnt_dust_a1 * deltat)));
  }

  if (do_dst3) {
    frzducnt =
        frzducnt + haero::min(uncoated_aer_num[Hetfrz::id_dst3] / deltat,
                              uncoated_aer_num[Hetfrz::id_dst3] / deltat *
                                  (1.0 - haero::exp(-Jcnt_dust_a3 * deltat)));
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_hetfrz_deposition_nucleation(
    const Real deltat, const Real temperature,
    Real uncoated_aer_num[Hetfrz::hetfrz_aer_nspec], const Real sigma_iv,
    const Real eswtr, const Real vwice, const Real rgdep, const Real r_bc,
    const Real r_dust_a1, const Real r_dust_a3, const bool do_bc,
    const bool do_dst1, const bool do_dst3, Real &frzbcdep, Real &frzdudep) {

  frzbcdep = 0.0;
  frzdudep = 0.0;

  // form factor
  const Real f_dep_bc =
      get_form_factor(Hetfrz::theta_dep_bc * Constants::pi / 180.0);
  const Real f_dep_dust_a1 =
      get_form_factor(Hetfrz::theta_dep_dust * Constants::pi / 180.0);
  const Real f_dep_dust_a3 =
      get_form_factor(Hetfrz::theta_dep_dust * Constants::pi / 180.0);

  // homogeneous energy of germ formation
  const Real dg0dep =
      4.0 * Constants::pi / 3.0 * sigma_iv * haero::square(rgdep);

  constexpr Real mwh2o = Constants::molec_weight_h2o *
                         1000.0; // Molecular weight water (BAD CONSTANT)

  // prefactor
  // attention: division of small numbers
  constexpr Real bad_boltzmann = 1.38e-23; // Boltzmann Constant (BAD CONSTANT)
  const Real Adep = haero::square(Hetfrz::rhwincloud * eswtr) *
                    (vwice / (mwh2o * Hetfrz::amu)) /
                    (bad_boltzmann * temperature * Hetfrz::nus) *
                    haero::sqrt(sigma_iv / (bad_boltzmann * temperature));

  Real Jdep_bc = 0.0;
  Real Jdep_dust_a1 = 0.0;
  Real Jdep_dust_a3 = 0.0;
  // nucleation rate per particle
  if (rgdep > 0) {
    Jdep_bc = Adep * haero::square(r_bc) / haero::sqrt(f_dep_bc) *
              haero::exp((-Hetfrz::dga_dep_bc - f_dep_bc * dg0dep) /
                         (bad_boltzmann * temperature));
    Jdep_dust_a1 = Adep * haero::square(r_dust_a1) /
                   haero::sqrt(f_dep_dust_a1) *
                   haero::exp((-Hetfrz::dga_dep_dust - f_dep_dust_a1 * dg0dep) /
                              (bad_boltzmann * temperature));
    Jdep_dust_a3 = Adep * haero::square(r_dust_a3) /
                   haero::sqrt(f_dep_dust_a3) *
                   haero::exp((-Hetfrz::dga_dep_dust - f_dep_dust_a3 * dg0dep) /
                              (bad_boltzmann * temperature));
  }

  // Limit to 1% of available potential IN (for BC), no limit for dust
  if (do_bc) {
    frzbcdep =
        frzbcdep +
        haero::min(Hetfrz::limfacbc * uncoated_aer_num[Hetfrz::id_bc] / deltat,
                   uncoated_aer_num[Hetfrz::id_bc] / deltat *
                       (1.0 - haero::exp(-Jdep_bc * deltat)));
  }

  if (do_dst1) {
    frzdudep =
        frzdudep + haero::min(uncoated_aer_num[Hetfrz::id_dst1] / deltat,
                              uncoated_aer_num[Hetfrz::id_dst1] / deltat *
                                  (1.0 - haero::exp(-Jdep_dust_a1 * deltat)));
  }

  if (do_dst3) {
    frzdudep =
        frzdudep + haero::min(uncoated_aer_num[Hetfrz::id_dst3] / deltat,
                              uncoated_aer_num[Hetfrz::id_dst3] / deltat *
                                  (1.0 - haero::exp(-Jdep_dust_a3 * deltat)));
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_hetfrz_immersion_nucleation(
    const Real deltat, const Real temperature,
    Real uncoated_aer_num[Hetfrz::hetfrz_aer_nspec],
    const Real total_interstitial_aer_num[Hetfrz::hetfrz_aer_nspec],
    const Real total_cloudborne_aer_num[Hetfrz::hetfrz_aer_nspec],
    const Real sigma_iw, const Real eswtr, const Real vwice,
    const Real dim_theta[Hetfrz::pdf_n_theta],
    const Real pdf_imm_theta[Hetfrz::pdf_n_theta], const Real rgimm_bc,
    const Real rgimm_dust_a1, const Real rgimm_dust_a3, const Real r_bc,
    const Real r_dust_a1, const Real r_dust_a3, const bool do_bc,
    const bool do_dst1, const bool do_dst3, Real &frzbcimm, Real &frzduimm) {

  frzbcimm = 0.0;
  frzduimm = 0.0;

  // form factor
  // only consider flat surfaces due to uncertainty of curved surfaces
  const Real f_imm_bc =
      get_form_factor(Hetfrz::theta_imm_bc * Constants::pi / 180.0);

  Real dim_f_imm_dust_a1[Hetfrz::pdf_n_theta] = {0.0};
  Real dim_f_imm_dust_a3[Hetfrz::pdf_n_theta] = {0.0};

  for (int ibin = Hetfrz::itheta_bin_beg; ibin <= Hetfrz::itheta_bin_end;
       ++ibin) {
    const Real ff = get_form_factor(dim_theta[ibin]);
    dim_f_imm_dust_a1[ibin] = ff;
    dim_f_imm_dust_a3[ibin] = ff;
  }

  // homogeneous energy of germ formation
  const Real dg0imm_bc = get_dg0imm(sigma_iw, rgimm_bc);
  const Real dg0imm_dust_a1 = get_dg0imm(sigma_iw, rgimm_dust_a1);
  const Real dg0imm_dust_a3 = get_dg0imm(sigma_iw, rgimm_dust_a3);

  // prefactor
  const Real Aimm_bc = get_Aimm(vwice, rgimm_bc, temperature, dg0imm_bc);
  const Real Aimm_dust_a1 =
      get_Aimm(vwice, rgimm_dust_a1, temperature, dg0imm_dust_a1);
  const Real Aimm_dust_a3 =
      get_Aimm(vwice, rgimm_dust_a3, temperature, dg0imm_dust_a3);

  // nucleation rate per particle
  constexpr Real bad_boltzmann = 1.38e-23; // (BAD CONSTANT)
  const Real Jimm_bc = Aimm_bc * haero::square(r_bc) / haero::sqrt(f_imm_bc) *
                       haero::exp((-Hetfrz::dga_imm_bc - f_imm_bc * dg0imm_bc) /
                                  (bad_boltzmann * temperature));

  Real dim_Jimm_dust_a1[Hetfrz::pdf_n_theta] = {0.0};
  Real dim_Jimm_dust_a3[Hetfrz::pdf_n_theta] = {0.0};

  for (int ibin = Hetfrz::itheta_bin_beg; ibin <= Hetfrz::itheta_bin_end;
       ++ibin) {
    dim_Jimm_dust_a1[ibin] =
        Aimm_dust_a1 * haero::square(r_dust_a1) /
        haero::sqrt(dim_f_imm_dust_a1[ibin]) *
        haero::exp(
            (-Hetfrz::dga_imm_dust - dim_f_imm_dust_a1[ibin] * dg0imm_dust_a1) /
            (bad_boltzmann * temperature));

    dim_Jimm_dust_a1[ibin] = haero::max(dim_Jimm_dust_a1[ibin], 0.0);

    dim_Jimm_dust_a3[ibin] =
        Aimm_dust_a3 * haero::square(r_dust_a3) /
        haero::sqrt(dim_f_imm_dust_a3[ibin]) *
        haero::exp(
            (-Hetfrz::dga_imm_dust - dim_f_imm_dust_a3[ibin] * dg0imm_dust_a3) /
            (bad_boltzmann * temperature));

    dim_Jimm_dust_a3[ibin] = haero::max(dim_Jimm_dust_a3[ibin], 0.0);
  }

  // Limit to 1% of available potential IN (for BC), no limit for dust
  Real sum_imm_dust_a1 = 0.0;
  Real sum_imm_dust_a3 = 0.0;
  for (int ibin = Hetfrz::itheta_bin_beg; ibin <= Hetfrz::itheta_bin_end - 1;
       ++ibin) {
    sum_imm_dust_a1 =
        sum_imm_dust_a1 +
        0.5 *
            ((pdf_imm_theta[ibin] *
                  haero::exp(-dim_Jimm_dust_a1[ibin] * deltat) +
              pdf_imm_theta[ibin + 1] *
                  haero::exp(-dim_Jimm_dust_a1[ibin + 1] * deltat))) *
            Hetfrz::pdf_d_theta;
    sum_imm_dust_a3 =
        sum_imm_dust_a3 +
        0.5 *
            ((pdf_imm_theta[ibin] *
                  haero::exp(-dim_Jimm_dust_a3[ibin] * deltat) +
              pdf_imm_theta[ibin + 1] *
                  haero::exp(-dim_Jimm_dust_a3[ibin + 1] * deltat))) *
            Hetfrz::pdf_d_theta;
  }

  if (sum_imm_dust_a1 > 0.99) {
    sum_imm_dust_a1 = 1.0;
  }
  if (sum_imm_dust_a3 > 0.99) {
    sum_imm_dust_a3 = 1.0;
  }

  if (do_bc) {
    // print do_bc
    const int id_bc = Hetfrz::id_bc;
    frzbcimm +=
        haero::min(Hetfrz::limfacbc * total_cloudborne_aer_num[id_bc] / deltat,
                   total_cloudborne_aer_num[id_bc] / deltat *
                       (1.0 - haero::exp(-Jimm_bc * deltat)));
  }

  if (do_dst1) {
    // print do_dst1
    const int id_dst1 = Hetfrz::id_dst1;
    frzduimm += haero::min(1.0 * total_cloudborne_aer_num[id_dst1] / deltat,
                           total_cloudborne_aer_num[id_dst1] / deltat *
                               (1.0 - sum_imm_dust_a1));
  }

  if (do_dst3) {
    // print do_dist3
    const int id_dst3 = Hetfrz::id_dst3;
    frzduimm += haero::min(1.0 * total_cloudborne_aer_num[id_dst3] / deltat,
                           total_cloudborne_aer_num[id_dst3] / deltat *
                               (1.0 - sum_imm_dust_a3));
  }

  if (temperature > 263.15) {
    frzduimm = 0.0;
    frzbcimm = 0.0;
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_rgimm_and_determine_spec_flag(
    const Real vwice, const Real sigma_iw, const Real temperature,
    const Real aw, const Real supersatice, Real &rgimm, bool &do_spec_flag) {

  // ****************************************************************************
  // calculate critical germ radius for immersion freezing and determine
  // flags for calculating ice nulceation for BC and dust
  // ****************************************************************************

  do_spec_flag = false;
  constexpr Real bad_boltzmann = 1.38e-23; // (BAD CONSTANT)
  // if aw*Si<=1, the freezing point depression is strong enough to prevent
  // freezing
  if (aw * supersatice > 1.0) {
    do_spec_flag = true;
    rgimm = 2 * vwice * sigma_iw /
            (bad_boltzmann * temperature * haero::log(aw * supersatice));
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_water_activity(
    Real total_interstitial_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real awcam[Hetfrz::hetfrz_aer_nspec], Real awfacm[Hetfrz::hetfrz_aer_nspec],
    const Real r3lx, Real aw[Hetfrz::hetfrz_aer_nspec]) {

  Real molal[Hetfrz::hetfrz_aer_nspec];
  constexpr Real mw_so4 = Constants::molec_weight_so4 * 1000.0; /// BAD CONSTANT
  constexpr Real coeff_c1 = 2.9244948e-2;
  constexpr Real coeff_c2 = 2.3141243e-3;
  constexpr Real coeff_c3 = 7.8184854e-7;

  for (int ispec = 0; ispec < Hetfrz::hetfrz_aer_nspec; ++ispec) {
    // calculate molality
    if (total_interstitial_aer_num[ispec] > 0.0) {
      molal[ispec] = (1.e-6 * awcam[ispec] * (1.0 - awfacm[ispec]) /
                      (mw_so4 * total_interstitial_aer_num[ispec] * 1.e6)) /
                     (4.0 * Constants::pi / 3.0 * Constants::density_h2o *
                      haero::cube(haero::max(r3lx, 4.0e-6)));

      aw[ispec] = 1.0 / (1.0 + coeff_c1 * molal[ispec] +
                         coeff_c2 * haero::square(molal[ispec]) +
                         coeff_c3 * haero::cube(molal[ispec]));
    }
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_vars_for_pdf_imm(Real dim_theta[Hetfrz::pdf_n_theta],
                                Real pdf_imm_theta[Hetfrz::pdf_n_theta]) {

  constexpr Real theta_min = 1.0 / 180.0 * Constants::pi;   // (BAD CONSTANT)
  constexpr Real theta_max = 179.0 / 180.0 * Constants::pi; // (BAD CONSTANT)
  constexpr Real imm_dust_mean_theta =
      46.0 / 180.0 * Constants::pi;         // (BAD CONSTANT)
  constexpr Real imm_dust_var_theta = 0.01; // (BAD CONSTANT)

  const Real ln_theta_min = haero::log(theta_min);
  const Real ln_theta_max = haero::log(theta_max);
  const Real ln_imm_dust_mean_theta = haero::log(imm_dust_mean_theta);

  // calculate the integral in the denominator
  const Real x1_imm = (ln_theta_min - ln_imm_dust_mean_theta) /
                      (haero::sqrt(2.0) * imm_dust_var_theta);
  const Real x2_imm = (ln_theta_max - ln_imm_dust_mean_theta) /
                      (haero::sqrt(2.0) * imm_dust_var_theta);

  const Real norm_theta_imm = (haero::erf(x2_imm) - haero::erf(x1_imm)) * 0.5;

  for (int ibin = 0; ibin < Hetfrz::pdf_n_theta; ++ibin) {
    dim_theta[ibin] = 0.0;
    pdf_imm_theta[ibin] = 0.0;
  }

  for (int ibin = Hetfrz::itheta_bin_beg; ibin <= Hetfrz::itheta_bin_end;
       ++ibin) {
    dim_theta[ibin] = 1.0 / 180.0 * Constants::pi + ibin * Hetfrz::pdf_d_theta;

    pdf_imm_theta[ibin] =
        haero::exp(-(haero::square(haero::log(dim_theta[ibin]) -
                                   ln_imm_dust_mean_theta)) /
                   (2.0 * haero::square(imm_dust_var_theta))) /
        (dim_theta[ibin] * imm_dust_var_theta *
         haero::sqrt(2.0 * Constants::pi)) /
        norm_theta_imm;
  }
}

KOKKOS_INLINE_FUNCTION
void hetfrz_classnuc_calc(
    const Real deltat, const Real temperature, const Real pressure,
    const Real supersatice, Real fn[Hetfrz::hetfrz_aer_nspec], const Real r3lx,
    const Real icnlx, Real hetraer[Hetfrz::hetfrz_aer_nspec],
    Real awcam[Hetfrz::hetfrz_aer_nspec], Real awfacm[Hetfrz::hetfrz_aer_nspec],
    Real dstcoat[Hetfrz::hetfrz_aer_nspec],
    Real total_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real coated_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real uncoated_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real total_interstitial_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real total_cloudborne_aer_num[Hetfrz::hetfrz_aer_nspec], Real &frzbcimm,
    Real &frzduimm, Real &frzbccnt, Real &frzducnt, Real &frzbcdep,
    Real &frzdudep) {

  // *****************************************************************************
  //                 PDF theta model
  // *****************************************************************************
  // some variables for PDF theta model
  // immersion freezing
  //
  // With the original value of pdf_n_theta set to 101 the dust activation
  // fraction between -15 and 0 C could be overestimated.  This problem was
  // eliminated by increasing pdf_n_theta to 301.  To reduce the expense of
  // computing the dust activation fraction the integral is only evaluated
  // where dim_theta is non-zero.  This was determined to be between
  // dim_theta index values of 53 through 113.  These loop bounds are
  // hardcoded in the variables itheta_bin_beg and itheta_bin_end.
  //

  Real dim_theta[Hetfrz::pdf_n_theta];
  Real pdf_imm_theta[Hetfrz::pdf_n_theta];

  calculate_vars_for_pdf_imm(dim_theta, pdf_imm_theta);

  // get saturation vapor pressures
  const Real eswtr = wv_sat_methods::svp_water(temperature);

  const Real tc = temperature - Constants::freezing_pt_h2o;
  const Real rhoice = 916.7 - 0.175 * tc - 5.e-4 * haero::square(tc);
  const Real vwice =
      (1000.0 * Constants::molec_weight_h2o) * Hetfrz::amu / rhoice;

  const Real sigma_iw = (28.5 + 0.25 * tc) * 1e-3;
  const Real sigma_iv = (76.1 - 0.155 * tc + 28.5 + 0.25 * tc) * 1.0e-3;

  // get mass mean radius
  const Real r_bc = hetraer[0];
  const Real r_dust_a1 = hetraer[1];
  const Real r_dust_a3 = hetraer[2];

  // calculate collision kernels as a function of environmental parameters and
  // aerosol/droplet sizes
  Real Kcoll_bc, Kcoll_dust_a1, Kcoll_dust_a3;
  collkernel(temperature, pressure, eswtr, r3lx, r_bc, r_dust_a1, r_dust_a3,
             Kcoll_bc, Kcoll_dust_a1, Kcoll_dust_a3);

  // *****************************************************************************
  //                take water activity into account
  // *****************************************************************************
  //   solute effect
  Real aw[Hetfrz::hetfrz_aer_nspec] = {1.0};

  // The heterogeneous ice freezing temperatures of all IN generally decrease
  // with increasing total solute mole fraction. Therefore, the large solution
  // concentration will cause the freezing point depression and the ice freezing
  // temperatures of all IN will get close to the homogeneous ice freezing
  // temperatures. Since we take into account water activity for three
  // heterogeneous freezing modes(immersion, deposition, and contact), we
  // utilize interstitial aerosols(not cloudborne aerosols) to calculate water
  // activity. If the index of IN is 0, it means three freezing modes of this
  // aerosol are depressed.

  calculate_water_activity(total_interstitial_aer_num, awcam, awfacm, r3lx, aw);

  // *****************************************************************************
  //                immersion freezing begin
  // *****************************************************************************

  // critical germ size
  constexpr Real bad_boltzmann = 1.38e-23; // (BAD CONSTANT)
  const Real rgimm = 2.0 * vwice * sigma_iw /
                     (bad_boltzmann * temperature * haero::log(supersatice));

  // take solute effect into account
  Real rgimm_bc = rgimm;
  Real rgimm_dust_a1 = rgimm;
  Real rgimm_dust_a3 = rgimm;
  // print out rgimm

  bool do_bc, do_dst1, do_dst3;
  calculate_rgimm_and_determine_spec_flag(vwice, sigma_iw, temperature,
                                          aw[Hetfrz::id_bc], supersatice,
                                          rgimm_bc, do_bc);

  calculate_rgimm_and_determine_spec_flag(vwice, sigma_iw, temperature,
                                          aw[Hetfrz::id_dst1], supersatice,
                                          rgimm_dust_a1, do_dst1);

  calculate_rgimm_and_determine_spec_flag(vwice, sigma_iw, temperature,
                                          aw[Hetfrz::id_dst3], supersatice,
                                          rgimm_dust_a3, do_dst3);

  calculate_hetfrz_immersion_nucleation(
      deltat, temperature, uncoated_aer_num, total_interstitial_aer_num,
      total_cloudborne_aer_num, sigma_iw, eswtr, vwice, dim_theta,
      pdf_imm_theta, rgimm_bc, rgimm_dust_a1, rgimm_dust_a3, r_bc, r_dust_a1,
      r_dust_a3, do_bc, do_dst1, do_dst3, frzbcimm, frzduimm);

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //  Deposition nucleation
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  // critical germ size
  // assume 98% RH in mixed-phase clouds (Korolev & Isaac, JAS 2006)
  const Real rgdep = 2.0 * vwice * sigma_iv /
                     (bad_boltzmann * temperature *
                      haero::log(Hetfrz::rhwincloud * supersatice));

  calculate_hetfrz_deposition_nucleation(
      deltat, temperature, uncoated_aer_num, sigma_iv, eswtr, vwice, rgdep,
      r_bc, r_dust_a1, r_dust_a3, do_bc, do_dst1, do_dst3, frzbcdep, frzdudep);

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // contact nucleation
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  calculate_hetfrz_contact_nucleation(
      deltat, temperature, uncoated_aer_num, icnlx, sigma_iv, eswtr, rgimm,
      r_bc, r_dust_a1, r_dust_a3, Kcoll_bc, Kcoll_dust_a1, Kcoll_dust_a3, do_bc,
      do_dst1, do_dst3, frzbccnt, frzducnt);
}

KOKKOS_INLINE_FUNCTION
void calculate_interstitial_aer_num(
    const Real bcmac, const Real dmac, const Real bcmpc, const Real dmc,
    const Real ssmc, const Real mommc, const Real bcmc, const Real pommc,
    const Real soammc, const Real ncoarse,
    Real total_interstital_aer_num[Hetfrz::hetfrz_aer_nspec]) {

  // fixed ratio converting BC mass to number (based on BC emission) [#/kg]
  constexpr Real bc_kg_to_num = 4.669152e+17; // (BAD CONSTANT)

  // fixed ratio converting accum mode dust mass to number [#/kg]
  constexpr Real dst1_kg_to_num = 3.484e+15; // (BAD CONSTANT)

  total_interstital_aer_num[0] = bcmac * bc_kg_to_num * Hetfrz::num_m3_to_cm3;
  total_interstital_aer_num[0] += bcmpc * bc_kg_to_num * Hetfrz::num_m3_to_cm3;
  total_interstital_aer_num[1] = dmac * dst1_kg_to_num * Hetfrz::num_m3_to_cm3;

  if (dmc > 0.0) {
    total_interstital_aer_num[2] =
        dmc / (ssmc + dmc + bcmc + pommc + soammc + mommc) * ncoarse *
        Hetfrz::num_m3_to_cm3;
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_cloudborne_aer_num(
    const Real dmac_cb, const Real ssmac_cb, const Real so4mac_cb,
    const Real bcmac_cb, const Real pommac_cb, const Real soamac_cb,
    const Real mommac_cb, const Real num_accum_cb, const Real dmc_cb,
    const Real ssmc_cb, const Real mommc_cb, const Real bcmc_cb,
    const Real pommc_cb, const Real soamc_cb, const Real num_coarse_cb,
    Real total_cloudborne_aer_num[Hetfrz::hetfrz_aer_nspec]) {
  // ***************************************************
  // calculate cloudborne aerosol concentrations for
  // BC and dust
  // ***************************************************
  if (bcmac_cb > 0.0) {
    total_cloudborne_aer_num[0] = bcmac_cb /
                                  (so4mac_cb + bcmac_cb + pommac_cb +
                                   soamac_cb + ssmac_cb + dmac_cb + mommac_cb) *
                                  num_accum_cb *
                                  Hetfrz::num_m3_to_cm3; // #/cm^3
  }

  if (dmac_cb > 0.0) {
    total_cloudborne_aer_num[1] = dmac_cb /
                                  (so4mac_cb + bcmac_cb + pommac_cb +
                                   soamac_cb + ssmac_cb + dmac_cb + mommac_cb) *
                                  num_accum_cb *
                                  Hetfrz::num_m3_to_cm3; // #/cm^3
  }

  if (dmc_cb > 0.0) {
    total_cloudborne_aer_num[2] =
        dmc_cb / (dmc_cb + ssmc_cb + mommc_cb + bcmc_cb + pommc_cb + soamc_cb) *
        num_coarse_cb * Hetfrz::num_m3_to_cm3;
  }
}

KOKKOS_INLINE_FUNCTION
Real get_aer_radius(const Real specdens, const Real aermc, const Real aernum) {
  // Compute radius for an aerosol species given mass and number concentrations
  // Note for C++ port: This function may already be present in the C++ utility
  // functions
  return haero::cbrt(3.0 / (4.0 * Constants::pi * specdens) * aermc / aernum);
}

KOKKOS_INLINE_FUNCTION
void calculate_mass_mean_radius(
    const Real bcmac, const Real bcmpc, const Real dmac, const Real dmc,
    Real total_interstitial_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real hetraer[Hetfrz::hetfrz_aer_nspec]) {

  const Real aermc_min_threshold = 1.0e-30;
  const Real aernum_min_threshold = 1.0e-3;
  const Real r_bc_prescribed = 0.067e-6;
  const Real r_dust_a1_prescribed = 0.258e-6;
  const Real r_dust_a3_prescribed = 1.576e-6;

  const Real bc_num = total_interstitial_aer_num[0];
  const Real dst1_num = total_interstitial_aer_num[1];
  const Real dst3_num = total_interstitial_aer_num[2];

  // Initialize hetraer with prescribed radius
  hetraer[0] = r_bc_prescribed;
  hetraer[1] = r_dust_a1_prescribed;
  hetraer[2] = r_dust_a3_prescribed;

  const Real specdens_bc = aero_species(int(AeroId::BC)).density;
  const Real specdens_dust = aero_species(int(AeroId::DST)).density;
  // BC
  if (((bcmac + bcmpc) * 1.0e-3 > aermc_min_threshold) &
      (bc_num > aernum_min_threshold)) {
    hetraer[0] = get_aer_radius(specdens_bc, bcmac + bcmpc,
                                bc_num * Hetfrz::num_cm3_to_m3);
  }

  // fine dust a1
  if ((dmac * 1e-3 > aermc_min_threshold) & (dst1_num > aernum_min_threshold)) {
    hetraer[1] =
        get_aer_radius(specdens_dust, dmac, dst1_num * Hetfrz::num_cm3_to_m3);
  }

  // coarse dust a3
  if ((dmc * 1e-3 > aermc_min_threshold) & (dst3_num > aernum_min_threshold)) {
    hetraer[2] =
        get_aer_radius(specdens_dust, dmc, dst3_num * Hetfrz::num_cm3_to_m3);
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_coated_fraction(
    const Real air_density, const Real so4mac, const Real pommac,
    const Real mommac, const Real soamac, const Real dmac, const Real bcmac,
    const Real mommpc, const Real pommpc, const Real bcmpc, const Real so4mc,
    const Real pommc, const Real soamc, const Real mommc, const Real dmc,
    Real total_interstitial_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real total_cloudborne_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real hetraer[Hetfrz::hetfrz_aer_nspec],
    Real total_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real coated_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real uncoated_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real dstcoat[Hetfrz::hetfrz_aer_nspec], Real &na500, Real &tot_na500) {

  // ***************************************************
  // calculate total, coated, uncoated number
  // concentration for BC and dust
  // ***************************************************

  const Real r_bc = hetraer[0];
  const Real r_dust_a1 = hetraer[1];
  const Real r_dust_a3 = hetraer[2];

  const Real alnsg_mode_accum =
      haero::log(modes(int(ModeIndex::Accumulation)).mean_std_dev);
  const Real alnsg_mode_coarse =
      haero::log(modes(int(ModeIndex::Coarse)).mean_std_dev);
  const Real alnsg_mode_pcarbon =
      haero::log(modes(int(ModeIndex::PrimaryCarbon)).mean_std_dev);

  const Real fac_volsfc_dust_a1 =
      haero::exp(2.5 * haero::square(alnsg_mode_accum));
  const Real fac_volsfc_dust_a3 =
      haero::exp(2.5 * haero::square(alnsg_mode_coarse));
  const Real fac_volsfc_pcarbon =
      haero::exp(2.5 * haero::square(alnsg_mode_pcarbon));

  Real vol_shell[Hetfrz::hetfrz_aer_nspec];
  Real vol_core[Hetfrz::hetfrz_aer_nspec];

  const Real spechygro_so4 = 0.507; // Sulfate hygroscopicity
  const Real spechygro_soa = 0.14;  // SOA hygroscopicity
  const Real spechygro_pom = 0.1;   // POM hygroscopicity
  const Real spechygro_mom = 0.1;   // MOM hygroscopicity
  const Real soa_equivso4_factor = spechygro_soa / spechygro_so4;
  const Real pom_equivso4_factor = spechygro_pom / spechygro_so4;
  const Real mom_equivso4_factor = spechygro_mom / spechygro_so4;

  const Real specdens_so4 = aero_species(int(AeroId::SO4)).density;
  const Real specdens_pom = aero_species(int(AeroId::POM)).density;
  const Real specdens_mom = aero_species(int(AeroId::MOM)).density;
  const Real specdens_soa = aero_species(int(AeroId::SOA)).density;
  const Real specdens_dst = aero_species(int(AeroId::DST)).density;
  const Real specdens_bc = aero_species(int(AeroId::BC)).density;

  vol_shell[1] =
      (so4mac / specdens_so4 + pommac * pom_equivso4_factor / specdens_pom +
       mommac * mom_equivso4_factor / specdens_mom +
       soamac * soa_equivso4_factor / specdens_soa) /
      air_density;

  vol_core[1] = dmac / (specdens_dst * air_density);

  // bc
  vol_shell[0] = (pommpc * pom_equivso4_factor / specdens_pom +
                  mommpc * mom_equivso4_factor / specdens_mom) /
                 air_density;

  vol_core[0] = bcmpc / (specdens_bc * air_density);

  // dust_a1
  Real coat_ratio1 = vol_shell[0] * (r_bc * 2.0) * fac_volsfc_pcarbon;

  const Real n_so4_monolayers_dust = 1.0;
  const Real dr_so4_monolayers_dust = n_so4_monolayers_dust * 4.76e-10;
  Real coat_ratio2 =
      haero::max(6.0 * dr_so4_monolayers_dust * vol_core[0], 0.0);
  dstcoat[0] = coat_ratio1 / coat_ratio2;

  // dust_a1
  coat_ratio1 = vol_shell[1] * (r_dust_a1 * 2.0) * fac_volsfc_dust_a1;
  coat_ratio2 = haero::max(6.0 * dr_so4_monolayers_dust * vol_core[1], 0.0);
  dstcoat[1] = coat_ratio1 / coat_ratio2;

  // dust_a3
  vol_shell[2] = so4mc / (specdens_so4 * air_density) +
                 pommc / (specdens_pom * air_density) +
                 soamc / (specdens_soa * air_density) +
                 mommc / (specdens_mom * air_density);

  vol_core[2] = dmc / (specdens_dst * air_density);
  coat_ratio1 = vol_shell[2] * (r_dust_a3 * 2.0) * fac_volsfc_dust_a3;
  coat_ratio2 = haero::max(6.0 * dr_so4_monolayers_dust * vol_core[2], 0.0);
  dstcoat[2] = coat_ratio1 / coat_ratio2;

  for (int ispec = 0; ispec < Hetfrz::hetfrz_aer_nspec; ++ispec) {
    dstcoat[ispec] = utils::min_max_bound(0.001, 1.0, dstcoat[ispec]);
  }

  for (int ispec = 0; ispec < Hetfrz::hetfrz_aer_nspec; ++ispec) {
    total_aer_num[ispec] =
        total_interstitial_aer_num[ispec] + total_cloudborne_aer_num[ispec];
    coated_aer_num[ispec] = total_interstitial_aer_num[ispec] * dstcoat[ispec];
    uncoated_aer_num[ispec] =
        total_interstitial_aer_num[ispec] * (1.0 - dstcoat[ispec]);
  }

  const Real bc_kg_to_num = 4.669152e+17;
  coated_aer_num[0] =
      bcmpc * bc_kg_to_num * Hetfrz::num_m3_to_cm3 * dstcoat[0] +
      bcmac * bc_kg_to_num * Hetfrz::num_m3_to_cm3;

  uncoated_aer_num[0] =
      bcmpc * bc_kg_to_num * Hetfrz::num_m3_to_cm3 * (1.0 - dstcoat[0]);

  const Real dst1_0p5um_scale = 0.488;
  const Real bc_0p5um_scale = 0.0256;
  // scaled for D>0.5 um using Clarke et al., 1997; 2004; 2007: rg=0.1um,
  // sig=1.6
  tot_na500 = total_aer_num[0] * bc_0p5um_scale +
              total_aer_num[1] * dst1_0p5um_scale + total_aer_num[2];

  // scaled for D>0.5 um using Clarke et al., 1997; 2004; 2007: rg=0.1um,
  // sig=1.6
  na500 = total_interstitial_aer_num[0] * bc_0p5um_scale +
          total_interstitial_aer_num[1] * dst1_0p5um_scale +
          total_interstitial_aer_num[2];
}

KOKKOS_INLINE_FUNCTION
void calculate_vars_for_water_activity(
    const Real so4mac, const Real soamac, const Real bcmac, const Real mommac,
    const Real pommac, const Real num_accum, const Real so4mc, const Real mommc,
    const Real bcmc, const Real pommc, const Real soamc, const Real num_coarse,
    Real total_interstitial_aer_num[Hetfrz::hetfrz_aer_nspec],
    Real awcam[Hetfrz::hetfrz_aer_nspec],
    Real awfacm[Hetfrz::hetfrz_aer_nspec]) {

  const Real bc_num = total_interstitial_aer_num[0];
  const Real dst1_num = total_interstitial_aer_num[1];
  const Real dst3_num = total_interstitial_aer_num[2];

  Real aermc_tmp = so4mac + soamac + pommac + bcmac + mommac;

  const Real mass_kg_to_mug = 1.0e9;
  // accumulation mode for dust_a1
  if (num_accum > 0.0) {
    awcam[1] = (dst1_num * Hetfrz::num_cm3_to_m3) / num_accum * aermc_tmp *
               mass_kg_to_mug;
  }

  if (awcam[1] > 0.0) {
    awfacm[1] = (bcmac + soamac + pommac + mommac) / aermc_tmp;
  }

  // accumulation mode for bc (if MAM4, primary carbon mode is insoluble)
  if (num_accum > 0.0) {
    awcam[0] = (bc_num * Hetfrz::num_cm3_to_m3) / num_accum * aermc_tmp *
               mass_kg_to_mug;
  }
  awfacm[0] = awfacm[1];

  aermc_tmp = so4mc + mommc + bcmc + pommc + soamc;

  // coarse mode for dust_a3
  if (num_coarse > 0.0) {
    awcam[2] = (dst3_num * Hetfrz::num_cm3_to_m3) / num_coarse * aermc_tmp *
               mass_kg_to_mug;
  }

  if (awcam[2] > 0.0) {
    awfacm[2] =
        (bcmc + soamc + pommc + mommc) / (soamc + pommc + so4mc + bcmc + mommc);
  }
}

KOKKOS_INLINE_FUNCTION
void hetfrz_rates_1box(const int k, const Real dt, const Atmosphere &atm,
                       const Prognostics &progs, const Diagnostics &diags,
                       const Tendencies &tends, const Hetfrz::Config &config) {
  const Real temp = atm.temperature(k);
  const Real pmid = atm.pressure(k);
  const Real qc = atm.liquid_mixing_ratio(k);
  const Real nc = atm.cloud_liquid_number_mixing_ratio(k);
  const Real ast = diags.stratiform_cloud_fraction(k);

  // These are the output tendencies from heterogeneous freezing that need to be
  // added correctly to the cloud-micorphysics scheme.
  auto &hetfrz_immersion_nucleation_tend =
      diags.hetfrz_immersion_nucleation_tend(k);
  hetfrz_immersion_nucleation_tend = 0;
  auto &hetfrz_contact_nucleation_tend =
      diags.hetfrz_contact_nucleation_tend(k);
  hetfrz_contact_nucleation_tend = 0;
  auto &hetfrz_depostion_nucleation_tend =
      diags.hetfrz_depostion_nucleation_tend(k);
  hetfrz_depostion_nucleation_tend = 0;

  // These fields are used for diagnostics
  auto &bc_num = diags.bc_num(k);
  bc_num = 0;
  auto &dst1_num = diags.dst1_num(k);
  dst1_num = 0;
  auto &dst3_num = diags.dst3_num(k);
  dst3_num = 0;
  auto &bcc_num = diags.bcc_num(k);
  bcc_num = 0;
  auto &dst1c_num = diags.dst1c_num(k);
  dst1c_num = 0;
  auto &dst3c_num = diags.dst3c_num(k);
  dst3c_num = 0;
  auto &bcuc_num = diags.bcuc_num(k);
  bcuc_num = 0;
  auto &dst1uc_num = diags.dst1uc_num(k);
  dst1uc_num = 0;
  auto &dst3uc_num = diags.dst3uc_num(k);
  dst3uc_num = 0;
  auto &bc_a1_num = diags.bc_a1_num(k);
  bc_a1_num = 0;
  auto &dst_a1_num = diags.dst_a1_num(k);
  dst_a1_num = 0;
  auto &dst_a3_num = diags.dst_a3_num(k);
  dst_a3_num = 0;
  auto &bc_c1_num = diags.bc_c1_num(k);
  bc_c1_num = 0;
  auto &dst_c1_num = diags.dst_c1_num(k);
  dst_c1_num = 0;
  auto &dst_c3_num = diags.dst_c3_num(k);
  dst_c3_num = 0;
  auto &fn_bc_c1_num = diags.fn_bc_c1_num(k);
  fn_bc_c1_num = 0;
  auto &fn_dst_c1_num = diags.fn_dst_c1_num(k);
  fn_dst_c1_num = 0;
  auto &fn_dst_c3_num = diags.fn_dst_c3_num(k);
  fn_dst_c3_num = 0;
  auto &na500 = diags.na500(k);
  na500 = 0;
  auto &totna500 = diags.totna500(k);
  totna500 = 0;
  auto &freqimm = diags.freqimm(k);
  freqimm = 0;
  auto &freqcnt = diags.freqcnt(k);
  freqcnt = 0;
  auto &freqdep = diags.freqdep(k);
  freqdep = 0;
  auto &freqmix = diags.freqmix(k);
  freqmix = 0;
  auto &dstfrezimm = diags.dstfrezimm(k);
  dstfrezimm = 0;
  auto &dstfrezcnt = diags.dstfrezcnt(k);
  dstfrezcnt = 0;
  auto &dstfrezdep = diags.dstfrezdep(k);
  dstfrezdep = 0;
  auto &bcfrezimm = diags.bcfrezimm(k);
  bcfrezimm = 0;
  auto &bcfrezcnt = diags.bcfrezcnt(k);
  bcfrezcnt = 0;
  auto &bcfrezdep = diags.bcfrezdep(k);
  bcfrezdep = 0;
  auto &nimix_imm = diags.nimix_imm(k);
  nimix_imm = 0;
  auto &nimix_cnt = diags.nimix_cnt(k);
  nimix_cnt = 0;
  auto &nimix_dep = diags.nimix_dep(k);
  nimix_dep = 0;
  auto &dstnidep = diags.dstnidep(k);
  dstnidep = 0;
  auto &dstnicnt = diags.dstnicnt(k);
  dstnicnt = 0;
  auto &dstniimm = diags.dstniimm(k);
  dstniimm = 0;
  auto &bcnidep = diags.bcnidep(k);
  bcnidep = 0;
  auto &bcnicnt = diags.bcnicnt(k);
  bcnicnt = 0;
  auto &bcniimm = diags.bcniimm(k);
  bcniimm = 0;
  auto &numice10s = diags.numice10s(k);
  numice10s = 0;
  auto &numimm10sdst = diags.numimm10sdst(k);
  numimm10sdst = 0;
  auto &numimm10sbc = diags.numimm10sbc(k);
  numimm10sbc = 0;
  // End fields used for diagnostics.

  const int coarse_idx = int(ModeIndex::Coarse);
  const int accum_idx = int(ModeIndex::Accumulation);
  const int pcarbon_idx = int(ModeIndex::PrimaryCarbon);

  auto &accum_bc = progs.q_aero_i[accum_idx][int(AeroId::BC)];
  auto &accum_dst = progs.q_aero_i[accum_idx][int(AeroId::DST)];
  auto &accum_soa = progs.q_aero_i[accum_idx][int(AeroId::SOA)];
  auto &accum_so4 = progs.q_aero_i[accum_idx][int(AeroId::SO4)];
  auto &accum_pom = progs.q_aero_i[accum_idx][int(AeroId::POM)];
  auto &accum_mom = progs.q_aero_i[accum_idx][int(AeroId::MOM)];
  auto &num_accum = progs.n_mode_i[accum_idx];

  auto &pcarbon_bc = progs.q_aero_i[pcarbon_idx][int(AeroId::BC)];
  auto &pcarbon_pom = progs.q_aero_i[pcarbon_idx][int(AeroId::POM)];
  auto &pcarbon_mom = progs.q_aero_i[pcarbon_idx][int(AeroId::MOM)];

  auto &coarse_dust = progs.q_aero_i[coarse_idx][int(AeroId::DST)];
  auto &coarse_ncl = progs.q_aero_i[coarse_idx][int(AeroId::NaCl)];
  auto &coarse_mom = progs.q_aero_i[coarse_idx][int(AeroId::MOM)];
  auto &coarse_bc = progs.q_aero_i[coarse_idx][int(AeroId::BC)];
  auto &coarse_pom = progs.q_aero_i[coarse_idx][int(AeroId::POM)];
  auto &coarse_soa = progs.q_aero_i[coarse_idx][int(AeroId::SOA)];
  auto &coarse_so4 = progs.q_aero_i[coarse_idx][int(AeroId::SO4)];

  auto &num_coarse = progs.n_mode_i[coarse_idx];

  // initialize rho
  const Real rair = 287.0423114; // Bad Constant
  const Real air_density = atm.pressure[k] / (atm.temperature[k] * rair);
  const Real lcldm = haero::max(ast, Hetfrz::mincld);

  const Real bcmac = accum_bc(k) * air_density;
  const Real dmac = accum_dst(k) * air_density;
  const Real soamac = accum_soa(k) * air_density;
  const Real so4mac = accum_so4(k) * air_density;
  const Real pommac = accum_pom(k) * air_density;
  const Real mommac = accum_mom(k) * air_density;

  const Real bcmpc = pcarbon_bc(k) * air_density;
  const Real pommpc = pcarbon_pom(k) * air_density;
  const Real mommpc = pcarbon_mom(k) * air_density;

  const Real dmc = coarse_dust(k) * air_density;
  const Real ssmc = coarse_ncl(k) * air_density;
  const Real mommc = coarse_mom(k) * air_density;
  const Real bcmc = coarse_bc(k) * air_density;
  const Real pommc = coarse_pom(k) * air_density;
  const Real soamc = coarse_soa(k) * air_density;
  const Real so4mc = coarse_so4(k) * air_density;

  Real total_interstitial_aer_num[Hetfrz::hetfrz_aer_nspec] = {0.0};

  calculate_interstitial_aer_num(bcmac, dmac, bcmpc, dmc, ssmc, mommc, bcmc,
                                 pommc, soamc, num_coarse[k],
                                 total_interstitial_aer_num);

  auto &accum_dst_cb = progs.q_aero_c[accum_idx][int(AeroId::DST)];
  auto &accum_ss_cb = progs.q_aero_c[accum_idx][int(AeroId::NaCl)];
  auto &accum_so4_cb = progs.q_aero_c[accum_idx][int(AeroId::SO4)];
  auto &accum_bc_cb = progs.q_aero_c[accum_idx][int(AeroId::BC)];
  auto &accum_pom_cb = progs.q_aero_c[accum_idx][int(AeroId::POM)];
  auto &accum_soa_cb = progs.q_aero_c[accum_idx][int(AeroId::SOA)];
  auto &accum_mom_cb = progs.q_aero_c[accum_idx][int(AeroId::MOM)];

  auto &num_accum_cb = progs.n_mode_c[accum_idx];

  const Real dmac_cb = accum_dst_cb[k] * air_density;
  const Real ssmac_cb = accum_ss_cb[k] * air_density;
  const Real so4mac_cb = accum_so4_cb[k] * air_density;
  const Real bcmaac_cb = accum_bc_cb[k] * air_density;
  const Real pommac_cb = accum_pom_cb[k] * air_density;
  const Real soamac_cb = accum_soa_cb[k] * air_density;
  const Real mommac_cb = accum_mom_cb[k] * air_density;

  auto &coarse_dust_cb = progs.q_aero_c[coarse_idx][int(AeroId::DST)];
  auto &coarse_ncl_cb = progs.q_aero_c[coarse_idx][int(AeroId::NaCl)];
  auto &coarse_mom_cb = progs.q_aero_c[coarse_idx][int(AeroId::MOM)];
  auto &coarse_bc_cb = progs.q_aero_c[coarse_idx][int(AeroId::BC)];
  auto &coarse_pom_cb = progs.q_aero_c[coarse_idx][int(AeroId::POM)];
  auto &coarse_soa_cb = progs.q_aero_c[coarse_idx][int(AeroId::SOA)];

  auto &num_coarse_cb = progs.n_mode_c[accum_idx];

  const Real dmc_cb = coarse_dust_cb[k] * air_density;
  const Real ssmc_cb = coarse_ncl_cb[k] * air_density;
  const Real mommc_cb = coarse_mom_cb[k] * air_density;
  const Real bcmc_cb = coarse_bc_cb[k] * air_density;
  const Real pommc_cb = coarse_pom_cb[k] * air_density;
  const Real soamc_cb = coarse_soa_cb[k] * air_density;

  Real total_cloudborne_aer_num[Hetfrz::hetfrz_aer_nspec] = {0.0};
  calculate_cloudborne_aer_num(
      dmac_cb, ssmac_cb, so4mac_cb, bcmaac_cb, pommac_cb, soamac_cb, mommac_cb,
      num_accum_cb[k], dmc_cb, ssmc_cb, mommc_cb, bcmc_cb, pommc_cb, soamc_cb,
      num_coarse_cb[k], total_cloudborne_aer_num);

  Real hetraer[Hetfrz::hetfrz_aer_nspec] = {0.0};

  calculate_mass_mean_radius(bcmac, bcmpc, dmac, dmc,
                             total_interstitial_aer_num, hetraer);

  Real total_aer_num[Hetfrz::hetfrz_aer_nspec] = {0.0};
  Real coated_aer_num[Hetfrz::hetfrz_aer_nspec] = {0.0};
  Real uncoated_aer_num[Hetfrz::hetfrz_aer_nspec] = {0.0};
  Real dstcoat[Hetfrz::hetfrz_aer_nspec] = {0.0};
  na500 = 0.0;
  totna500 = 0.0;

  calculate_coated_fraction(
      air_density, so4mac, pommac, mommac, soamac, dmac, bcmac, mommpc, pommpc,
      bcmpc, so4mc, pommc, soamc, mommc, dmc, total_interstitial_aer_num,
      total_cloudborne_aer_num, hetraer, total_aer_num, coated_aer_num,
      uncoated_aer_num, dstcoat, na500, totna500);

  Real awcam[Hetfrz::hetfrz_aer_nspec] = {0.0};
  Real awfacm[Hetfrz::hetfrz_aer_nspec] = {0.0};

  calculate_vars_for_water_activity(
      so4mac, soamac, bcmac, mommac, pommac, num_accum[k], so4mc, mommc, bcmc,
      pommc, soamc, num_coarse[k], total_interstitial_aer_num, awcam, awfacm);

  auto af_accum = diags.activation_fraction[accum_idx](k);
  auto af_coarse = diags.activation_fraction[coarse_idx](k);

  Real cloudborne_aer_num[Hetfrz::hetfrz_aer_nspec] = {0.0};
  cloudborne_aer_num[0] = total_aer_num[0] * af_accum;  // bc
  cloudborne_aer_num[1] = total_aer_num[1] * af_accum;  // dst_a1
  cloudborne_aer_num[2] = total_aer_num[2] * af_coarse; // dst_a3

  // Store local variables to diags
  bc_num = total_aer_num[0];
  dst1_num = total_aer_num[1];
  dst3_num = total_aer_num[2];

  bcc_num = coated_aer_num[0];
  dst1c_num = coated_aer_num[1];
  dst3c_num = coated_aer_num[2];

  bcuc_num = uncoated_aer_num[0];
  dst1uc_num = uncoated_aer_num[1];
  dst3uc_num = uncoated_aer_num[2];

  bc_a1_num = total_interstitial_aer_num[0];
  dst_a1_num = total_interstitial_aer_num[1];
  dst_a3_num = total_interstitial_aer_num[2];

  bc_c1_num = total_cloudborne_aer_num[0];
  dst_c1_num = total_cloudborne_aer_num[1];
  dst_c3_num = total_cloudborne_aer_num[2];

  fn_bc_c1_num = cloudborne_aer_num[0];
  fn_dst_c1_num = cloudborne_aer_num[1];
  fn_dst_c3_num = cloudborne_aer_num[2];

  // Finshed storing local variables to diags
  Real frzbcimm = 0.0;
  Real frzduimm = 0.0;
  Real frzbccnt = 0.0;
  Real frzducnt = 0.0;
  Real frzbcdep = 0.0;
  Real frzdudep = 0.0;

  if ((temp > 235.15) & (temp < 269.15)) {

    const Real qcic = haero::min((qc / lcldm), 5.0e-3);
    const Real ncic = haero::max((nc / lcldm), 0.0);

    const Real con1 = 1.0 / haero::pow((1.333 * Constants::pi), 0.333);
    Real r3lx = con1 * haero::pow(air_density * qcic /
                                      (Constants::density_h2o *
                                       haero::max(ncic * air_density, 1.0e6)),
                                  0.333); // in m
    r3lx = haero::max(4e-6, r3lx);
    const Real supersatice =
        wv_sat_methods::svp_water(temp) / wv_sat_methods::svp_ice(temp);

    Real fn[Hetfrz::hetfrz_aer_nspec] = {af_accum, af_accum, af_coarse};

    hetfrz::hetfrz_classnuc_calc(
        dt, temp, pmid, supersatice, fn, r3lx, ncic * air_density * 1e-6,
        hetraer, awcam, awfacm, dstcoat, total_aer_num, coated_aer_num,
        uncoated_aer_num, total_interstitial_aer_num, total_cloudborne_aer_num,
        frzbcimm, frzduimm, frzbccnt, frzducnt, frzbcdep, frzdudep);

    // These are the output tendencies from hetfrz that need to be properly
    // coupled into the cloud micorphysical scheme
    hetfrz_immersion_nucleation_tend = frzbcimm + frzduimm;
    hetfrz_contact_nucleation_tend = frzbccnt + frzducnt;
    hetfrz_depostion_nucleation_tend = frzbcdep + frzdudep;

    if (hetfrz_immersion_nucleation_tend > 0.0)
      freqimm = 1.0;
    if (hetfrz_contact_nucleation_tend > 0.0)
      freqcnt = 1.0;
    if (hetfrz_depostion_nucleation_tend > 0.0)
      freqdep = 1.0;
    if ((hetfrz_immersion_nucleation_tend + hetfrz_contact_nucleation_tend +
         hetfrz_depostion_nucleation_tend) > 0.0)
      freqmix = 1.0;
  }

  bcfrezimm = frzbcimm * Hetfrz::frz_cm3_to_m3 * ast;
  bcfrezcnt = frzbccnt * Hetfrz::frz_cm3_to_m3 * ast;
  bcfrezdep = frzbcdep * Hetfrz::frz_cm3_to_m3 * ast;

  dstfrezimm = frzduimm * Hetfrz::frz_cm3_to_m3 * ast;
  dstfrezcnt = frzducnt * Hetfrz::frz_cm3_to_m3 * ast;
  dstfrezdep = frzdudep * Hetfrz::frz_cm3_to_m3 * ast;

  nimix_imm = frzbcimm * Hetfrz::frz_cm3_to_m3 * dt;
  nimix_cnt = frzbccnt * Hetfrz::frz_cm3_to_m3 * dt;
  nimix_dep = frzbcdep * Hetfrz::frz_cm3_to_m3 * dt;

  dstnicnt = frzduimm * Hetfrz::frz_cm3_to_m3 * dt;
  dstnidep = frzducnt * Hetfrz::frz_cm3_to_m3 * dt;
  dstniimm = frzdudep * Hetfrz::frz_cm3_to_m3 * dt;

  numice10s =
      (hetfrz_immersion_nucleation_tend + hetfrz_contact_nucleation_tend +
       hetfrz_depostion_nucleation_tend) *
      Hetfrz::frz_cm3_to_m3 * dt * (10.0 / dt);
  numimm10sdst = hetfrz_immersion_nucleation_tend * Hetfrz::frz_cm3_to_m3 * dt *
                 (10.0 / dt);
  numimm10sbc = hetfrz_immersion_nucleation_tend * Hetfrz::frz_cm3_to_m3 * dt *
                (10.0 / dt);
}

} // namespace hetfrz

// init -- initializes the implementation with MAM4's configuration
inline void Hetfrz::init(const AeroConfig &aero_config,
                         const Config &process_config) {

  config_ = process_config;
};

// compute_tendencies -- computes tendencies and updates diagnostics
// NOTE: that both diags and tends are const below--this means their views
// NOTE: are fixed, but the data in those views is allowed to vary.
KOKKOS_INLINE_FUNCTION
void Hetfrz::compute_tendencies(const AeroConfig &config,
                                const ThreadTeam &team, Real t, Real dt,
                                const Atmosphere &atm, const Surface &sfc,
                                const Prognostics &progs,
                                const Diagnostics &diags,
                                const Tendencies &tends) const {

  // Hetergeneous freezing does not update tendienes directly, as hetergenoues
  // freezing does not contribute to aerosol-scheme specficic prognostic
  // variables, rather it it computes tendencies that are stored as dianostics,
  // namely hetfrz_immersion_nucleation_tend, hetfrz_contact_nucleation_tend,
  // and hetfrz_depostion_nucleation_tend. In order for heterogenous freezing to
  // have impacts on the model state, these tendencies must be correctly added
  // to the relevant cloud-microphysical parameterization.

  const int nk = atm.num_levels();
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, nk), KOKKOS_CLASS_LAMBDA(int k) {
        hetfrz::hetfrz_rates_1box(k, dt, atm, progs, diags, tends, config_);
      });
}

} // namespace mam4

#endif
