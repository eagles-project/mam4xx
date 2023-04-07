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

namespace mam4 {

class Hetfzr {

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
  const char *name() const { return "MAM4 aging"; }

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config());

  static constexpr Real amu = 1.66053886e-27;

  // frequ. of vibration [s-1] higher freq. (as in P&K, consistent with Anupam's
  // data)
  static constexpr Real nus = 1.0e13;

  // 98% RH in mixed-phase clouds (Korolev & Isaac, JAS 2006)
  static constexpr Real rhwincloud = 0.98;

  // max. ice nucleating fraction soot
  static constexpr Real limfacbc = 0.01;

  // Wang et al., 2014 fitting parameters
  // freezing parameters for immersion freezing
  static constexpr Real theta_imm_bc =
      48.0; // contact angle [deg], converted to rad later !DeMott et al (1990)
  static constexpr Real dga_imm_bc = 14.15E-20; // activation energy [J]
  static constexpr Real theta_imm_dust =
      46.0; // contact angle [deg], converted to rad later !DeMott et al (2011)
            // SD
  static constexpr Real dga_imm_dust = 14.75E-20; // activation energy [J]

  // freezing parameters for deposition nucleation
  static constexpr Real theta_dep_dust =
      20.0; // contact angle [deg], converted to rad later !Koehler et al (2010)
            // SD
  static constexpr Real dga_dep_dust = -8.1e-21; // activation energy [J]
  static constexpr Real theta_dep_bc =
      28.0; // contact angle [deg], converted to rad later !Moehler et al
            // (2005), soot
  static constexpr Real dga_dep_bc = -2.e-19; // activation energy [J]

  // Index ids and number of species involved in heterogenous freezing
  static constexpr int id_bc = 0;
  static constexpr int id_dst1 = 1;
  static constexpr int id_dst3 = 2;
  static constexpr int hetfzr_aer_nspec = 3;

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

namespace hetfzr {

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
  const Real vlc_drop_adjfunc =
      (haero::exp(coeff_adj_a + coeff_adj_b * haero::cube(haero::log(r3lx)) +
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
  const Real rh2o = Constants::r_gas_h2o_vapor;
  // G-factor = rhoh2o*Xi in Rogers & Yau, p. 104

  const Real G_factor = rhoh2o / ((latvap / (rh2o * temperature) - 1) * latvap *
                                      rhoh2o / (Ktherm_air * temperature) +
                                  rhoh2o * rh2o * temperature / (Dvap * eswtr));

  // calculate T-Tc as in Cotton et al.
  const Real Tdiff =
      -G_factor * (Hetfzr::rhwincloud - 1.) * latvap / Ktherm_air;

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
  constexpr Real kboltz2 = 1.38065e-23; // BAD CONSTANT

  // Knudsen number (Seinfeld & Pandis 8.1)
  const Real Kn = lambda / r_a;

  // aerosol diffusivity
  const Real Daer =
      kboltz2 * temperature * (1 + Kn) /
      (6.0 * Constants::pi * Constants::r_gas_dry_air * viscos_air);

  // Schmidt number
  const Real Sc = viscos_air / (Daer * rho_air);

  //  Young (1974) first equ. on page 771
  const Real K_brownian = 4.0 * Constants::pi * r3lx * Daer *
                          (1 + 0.3 * haero::sqrt(Re) * haero::cbrt(Sc));

  // form factor
  const Real f_t =
      0.4 * (1.0 + 1.45 * Kn + 0.4 * Kn * haero::exp(-1. / Kn)) *
      (Ktherm_air + 2.5 * Kn * Ktherm) /
      ((1.0 + 3.0 * Kn) * (2.0 * Ktherm_air + 5.0 * Kn * Ktherm + Ktherm));

  const Real Q_heat = Ktherm_air / r3lx *
                      (1.0 + 0.3 * haero::sqrt(Re) * haero::cbrt(Pr)) * Tdiff;

  const Real K_thermo_cotton =
      4.0 * Constants::pi * haero::square(r3lx) * f_t * Q_heat / pressure;

  const Real K_diffusio_cotton =
      -(1.0 / f_t) * (Constants::r_gas_h2o_vapor * temperature / latvap) *
      K_thermo_cotton;

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
  constexpr Real Ktherm_bc = 4.2;    // Carbon
  constexpr Real Ktherm_dust = 0.72; // Clay

  const Real tc = temperature - Constants::triple_pt_h2o;

  // air viscosity for tc<0,
  const Real viscos_air = get_air_viscosity(tc);

  // air density
  const Real rho_air = pressure / (Constants::r_gas_dry_air * temperature);

  // mean free path: Seinfeld & Pandis 8.6
  const Real lambda =
      2.0 * viscos_air /
      (pressure * haero::sqrt(8.0 / (Constants::pi * Constants::r_gas_dry_air *
                                     temperature)));

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
  const Real v_cos = cos(alpha); // This should resolve to haero cos
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
                            // area of substrate [m-2]
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
    Real uncoated_aer_num[Hetfzr::hetfzr_aer_nspec], const Real icnlx,
    const Real sigma_iv, const Real eswtr, const Real rgimm, const Real r_bc,
    const Real r_dust_a1, const Real r_dust_a3, const Real Kcoll_bc,
    const Real Kcoll_dust_a1, const Real Kcoll_dust_a3, const bool do_bc,
    const bool do_dst1, const bool do_dst3, Real &frzbccnt, Real &frzducnt) {

  frzbccnt = 0.0;
  frzducnt = 0.0;

  // form factor
  const Real f_cnt_bc =
      get_form_factor(Hetfzr::theta_dep_bc * Constants::pi / 180.0);
  const Real f_cnt_dust_a1 =
      get_form_factor(Hetfzr::theta_dep_dust * Constants::pi / 180.0);
  const Real f_cnt_dust_a3 =
      get_form_factor(Hetfzr::theta_dep_dust * Constants::pi / 180.0);

  // homogeneous energy of germ formation
  const Real dg0cnt =
      4.0 * Constants::pi / 3.0 * sigma_iv * haero::square(rgimm);

  // prefactor
  // attention: division of small numbers
  constexpr Real bad_boltzmann = 1.38e-23; // Boltzmann Constant (BAD CONSTANT)
  const Real Acnt =
      Hetfzr::rhwincloud * eswtr * 4.0 * Constants::pi /
      (Hetfzr::nus *
       haero::sqrt(2.0 * Constants::pi * Constants::molec_weight_h2o *
                   Hetfzr::amu * bad_boltzmann * temperature));

  // nucleation rate per particle
  const Real Jcnt_bc = Acnt * haero::square(r_bc) *
                       haero::exp((-Hetfzr::dga_dep_bc - f_cnt_bc * dg0cnt) /
                                  (bad_boltzmann * temperature)) *
                       Kcoll_bc * icnlx;
  const Real Jcnt_dust_a1 =
      Acnt * haero::square(r_dust_a1) *
      haero::exp((-Hetfzr::dga_dep_dust - f_cnt_dust_a1 * dg0cnt) /
                 (bad_boltzmann * temperature)) *
      Kcoll_dust_a1 * icnlx;
  const Real Jcnt_dust_a3 =
      Acnt * haero::square(r_dust_a3) *
      haero::exp((-Hetfzr::dga_dep_dust - f_cnt_dust_a3 * dg0cnt) /
                 (bad_boltzmann * temperature)) *
      Kcoll_dust_a3 * icnlx;

  // Limit to 1% of available potential IN (for BC), no limit for dust
  if (do_bc) {
    frzbccnt =
        frzbccnt +
        haero::min(Hetfzr::limfacbc * uncoated_aer_num[Hetfzr::id_bc] / deltat,
                   uncoated_aer_num[Hetfzr::id_bc] / deltat *
                       (1.0 - haero::exp(-Jcnt_bc * deltat)));
  }

  if (do_dst1) {
    frzducnt =
        frzducnt + haero::min(uncoated_aer_num[Hetfzr::id_dst1] / deltat,
                              uncoated_aer_num[Hetfzr::id_dst1] / deltat *
                                  (1.0 - haero::exp(-Jcnt_dust_a1 * deltat)));
  }

  if (do_dst3) {
    frzducnt =
        frzducnt + haero::min(uncoated_aer_num[Hetfzr::id_dst3] / deltat,
                              uncoated_aer_num[Hetfzr::id_dst3] / deltat *
                                  (1.0 - haero::exp(-Jcnt_dust_a3 * deltat)));
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_hetfrz_deposition_nucleation(
    const Real deltat, const Real temperature,
    Real uncoated_aer_num[Hetfzr::hetfzr_aer_nspec], const Real sigma_iv,
    const Real eswtr, const Real vwice, const Real rgdep, const Real r_bc,
    const Real r_dust_a1, const Real r_dust_a3, const bool do_bc,
    const bool do_dst1, const bool do_dst3, Real &frzbcdep, Real &frzdudep) {

  frzbcdep = 0.0;
  frzdudep = 0.0;

  // form factor
  const Real f_dep_bc =
      get_form_factor(Hetfzr::theta_dep_bc * Constants::pi / 180.0);
  const Real f_dep_dust_a1 =
      get_form_factor(Hetfzr::theta_dep_dust * Constants::pi / 180.0);
  const Real f_dep_dust_a3 =
      get_form_factor(Hetfzr::theta_dep_dust * Constants::pi / 180.0);

  // homogeneous energy of germ formation
  const Real dg0dep =
      4.0 * Constants::pi / 3.0 * sigma_iv * haero::square(rgdep);

  // prefactor
  // attention: division of small numbers
  constexpr Real bad_boltzmann = 1.38e-23; // Boltzmann Constant (BAD CONSTANT)
  const Real Adep = haero::square(Hetfzr::rhwincloud * eswtr) *
                    (vwice / (Constants::molec_weight_h2o * Hetfzr::amu)) /
                    (bad_boltzmann * temperature * Hetfzr::nus) *
                    haero::sqrt(sigma_iv / (bad_boltzmann * temperature));

  Real Jdep_bc = 0.0;
  Real Jdep_dust_a1 = 0.0;
  Real Jdep_dust_a3 = 0.0;
  // nucleation rate per particle
  if (rgdep > 0) {
    Jdep_bc = Adep * haero::square(r_bc) / haero::sqrt(f_dep_bc) *
              haero::exp((-Hetfzr::dga_dep_bc - f_dep_bc * dg0dep) /
                         (bad_boltzmann * temperature));
    Jdep_dust_a1 = Adep * haero::square(r_dust_a1) /
                   haero::sqrt(f_dep_dust_a1) *
                   haero::exp((-Hetfzr::dga_dep_dust - f_dep_dust_a1 * dg0dep) /
                              (bad_boltzmann * temperature));
    Jdep_dust_a3 = Adep * haero::square(r_dust_a3) /
                   haero::sqrt(f_dep_dust_a3) *
                   haero::exp((-Hetfzr::dga_dep_dust - f_dep_dust_a3 * dg0dep) /
                              (bad_boltzmann * temperature));
  }

  // Limit to 1% of available potential IN (for BC), no limit for dust
  if (do_bc) {
    frzbcdep =
        frzbcdep +
        haero::min(Hetfzr::limfacbc * uncoated_aer_num[Hetfzr::id_bc] / deltat,
                   uncoated_aer_num[Hetfzr::id_bc] / deltat *
                       (1.0 - haero::exp(-Jdep_bc * deltat)));
  }

  if (do_dst1) {
    frzdudep =
        frzdudep + haero::min(uncoated_aer_num[Hetfzr::id_dst1] / deltat,
                              uncoated_aer_num[Hetfzr::id_dst1] / deltat *
                                  (1.0 - haero::exp(-Jdep_dust_a1 * deltat)));
  }

  if (do_dst3) {
    frzdudep =
        frzdudep + haero::min(uncoated_aer_num[Hetfzr::id_dst3] / deltat,
                              uncoated_aer_num[Hetfzr::id_dst3] / deltat *
                                  (1.0 - haero::exp(-Jdep_dust_a3 * deltat)));
  }
}

KOKKOS_INLINE_FUNCTION
void calculate_hetfrz_immersion_nucleation(
    const Real deltat, const Real temperature,
    Real uncoated_aer_num[Hetfzr::hetfzr_aer_nspec],
    const Real total_interstitial_aer_num,
    const Real total_cloudborne_aer_num[Hetfzr::hetfzr_aer_nspec],
    const Real sigma_iw, const Real eswtr, const Real vwice,
    const Real dim_theta[Hetfzr::pdf_n_theta],
    const Real pdf_imm_theta[Hetfzr::pdf_n_theta], const Real rgimm_bc,
    const Real rgimm_dust_a1, const Real rgimm_dust_a3, const Real r_bc,
    const Real r_dust_a1, const Real r_dust_a3, const bool do_bc,
    const bool do_dst1, const bool do_dst3, Real &frzbcimm, Real &frzduimm) {

  frzbcimm = 0.0;
  frzduimm = 0.0;

  // form factor
  // only consider flat surfaces due to uncertainty of curved surfaces
  const Real f_imm_bc =
      get_form_factor(Hetfzr::theta_imm_bc * Constants::pi / 180.0);

  Real dim_f_imm_dust_a1[Hetfzr::pdf_n_theta] = {0.0};
  Real dim_f_imm_dust_a3[Hetfzr::pdf_n_theta] = {0.0};

  for (int ibin = Hetfzr::itheta_bin_beg; ibin <= Hetfzr::itheta_bin_end;
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
                       haero::exp((-Hetfzr::dga_imm_bc - f_imm_bc * dg0imm_bc) /
                                  (bad_boltzmann * temperature));

  Real dim_Jimm_dust_a1[Hetfzr::pdf_n_theta] = {0.0};
  Real dim_Jimm_dust_a3[Hetfzr::pdf_n_theta] = {0.0};

  for (int ibin = Hetfzr::itheta_bin_beg; ibin <= Hetfzr::itheta_bin_end;
       ++ibin) {
    dim_Jimm_dust_a1[ibin] =
        Aimm_dust_a1 * haero::square(r_dust_a1) /
        haero::sqrt(dim_f_imm_dust_a1[ibin]) *
        haero::exp(
            (-Hetfzr::dga_imm_dust - dim_f_imm_dust_a1[ibin] * dg0imm_dust_a1) /
            (bad_boltzmann * temperature));

    dim_Jimm_dust_a1[ibin] = haero::max(dim_Jimm_dust_a1[ibin], 0.0);

    dim_Jimm_dust_a3[ibin] =
        Aimm_dust_a3 * haero::square(r_dust_a3) /
        haero::sqrt(dim_f_imm_dust_a3[ibin]) *
        haero::exp(
            (-Hetfzr::dga_imm_dust - dim_f_imm_dust_a3[ibin] * dg0imm_dust_a3) /
            (bad_boltzmann * temperature));

    dim_Jimm_dust_a3[ibin] = haero::max(dim_Jimm_dust_a3[ibin], 0.0);
  }

  // Limit to 1% of available potential IN (for BC), no limit for dust
  Real sum_imm_dust_a1 = 0.0;
  Real sum_imm_dust_a3 = 0.0;
  for (int ibin = Hetfzr::itheta_bin_beg; ibin <= Hetfzr::itheta_bin_end - 1;
       ++ibin) {
    sum_imm_dust_a1 =
        sum_imm_dust_a1 +
        0.5 *
            ((pdf_imm_theta[ibin] *
                  haero::exp(-dim_Jimm_dust_a1[ibin] * deltat) +
              pdf_imm_theta[ibin + 1] *
                  haero::exp(-dim_Jimm_dust_a1[ibin + 1] * deltat))) *
            Hetfzr::pdf_d_theta;
    sum_imm_dust_a3 =
        sum_imm_dust_a3 +
        0.5 *
            ((pdf_imm_theta[ibin] *
                  haero::exp(-dim_Jimm_dust_a3[ibin] * deltat) +
              pdf_imm_theta[ibin + 1] *
                  haero::exp(-dim_Jimm_dust_a3[ibin + 1] * deltat))) *
            Hetfzr::pdf_d_theta;
  }

  if (sum_imm_dust_a1 > 0.99) {
    sum_imm_dust_a1 = 1.0;
  }
  if (sum_imm_dust_a3 > 0.99) {
    sum_imm_dust_a3 = 1.0;
  }

  if (do_bc) {
    const int id_bc = Hetfzr::id_bc;
    frzbcimm +=
        haero::min(Hetfzr::limfacbc * total_cloudborne_aer_num[id_bc] / deltat,
                   total_cloudborne_aer_num[id_bc] / deltat *
                       (1.0 - haero::exp(-Jimm_bc * deltat)));
  }

  if (do_dst1) {
    const int id_dst1 = Hetfzr::id_dst1;
    frzduimm += haero::min(1.0 * total_cloudborne_aer_num[id_dst1] / deltat,
                           total_cloudborne_aer_num[id_dst1] / deltat *
                               (1.0 - sum_imm_dust_a1));
  }

  if (do_dst3) {
    const int id_dst3 = Hetfzr::id_dst3;
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

} // namespace hetfzr
} // namespace mam4

#endif