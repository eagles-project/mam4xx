#ifndef MAM4XX_GASAEREXCH_HPP
#define MAM4XX_GASAEREXCH_HPP

#include <mam4xx/mam4_types.hpp>
#include <mam4xx/aero_config.hpp>

#include <haero/atmosphere.hpp>
#include <haero/haero.hpp>
#include <haero/constants.hpp>
#include <Kokkos_Array.hpp>

namespace mam4 {

using Pack = haero::PackType;

/// @class GasAerExch
/// This class implements MAM4's gas/aersol exchange  parameterization. Its
/// structure is defined by the usage of the impl_ member in the AeroProcess
/// class in
/// ../aero_process.hpp.
class GasAerExch {
  static constexpr int num_mode = AeroConfig::num_modes();
  static constexpr int num_gas = AeroConfig::num_gas_ids();
  static constexpr int num_aer = AeroConfig::num_aerosol_ids();
  static constexpr int nait = static_cast<int>(ModeIndex::Aitken);
  static constexpr int npca = static_cast<int>(ModeIndex::PrimaryCarbon);
  static constexpr int igas_h2so4 = static_cast<int>(GasId::H2SO4);
  static constexpr int igas_nh3 = static_cast<int>(GasId::NH3);
  static constexpr int iaer_h2so4 = static_cast<int>(GasId::H2SO4);
  static constexpr int iaer_so4 = static_cast<int>(AeroId::SO4);
  static constexpr int iaer_pom = static_cast<int>(AeroId::POM);
  static constexpr int igas_soag_bgn = static_cast<int>(AeroId::SOA);
  static constexpr int igas_soag_end = static_cast<int>(AeroId::SOA);
  // static const int npoa     = static_cast<int>(AeroId::POA);
  enum { NA, ANAL, IMPL };

  static constexpr Real mw_h2so4 = Constants::molec_weight_h2so4;

  // ratio of gas uptake coeff w.r.t. that of h2so4
  static constexpr Real soag_h2so4_uptake_coeff_ratio = 0.81; // for SOAG
  static constexpr Real nh3_h2so4_uptake_coeff_ratio = 2.08;  // for NH3

  bool l_gas_condense_to_mode[num_gas][num_mode]={};
  int eqn_and_numerics_category[num_gas]={};
  Real uptk_rate_factor[num_gas]={};
  Real modes_mean_std_dev[num_mode]={};
  //haero::DeviceType::view_1d<int> mode_aging_optaa;

public:
  // process-specific configuration data (if any)
  struct Config {
    using ColumnView = haero::ColumnView;
    Config() {}
    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
    bool l_mode_can_contain_species[num_aer][num_mode]={};
    bool l_mode_can_age[num_aer]={};
    int idx_gas_to_aer[num_gas]={};
    // qgas_netprod_otrproc = gas net production rate from other processes
    // such as gas-phase chemistry and emissions (mol/mol/s)
    // this allows the condensation (gasaerexch) routine to apply production and
    // condensation loss together, which is more accurate numerically
    // NOTE - must be >= zero, as numerical method can fail when it is negative
    // NOTE - currently only the values for h2so4 and nh3 should be non-zero
    Real qgas_netprod_otrproc[num_gas]={};
  };

  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 gas/aersol exchange"; }

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config()) {

    config_ = process_config;
    //Kokkos::resize(mode_aging_optaa, num_mode);

    //------------------------------------------------------------------
    // MAM currently assumes that the uptake rate of other gases
    // are proportional to the uptake rate of sulfuric acid gas (H2SO4).
    // Here the array uptk_rate_factor contains the uptake rate ratio
    // w.r.t. that of H2SO4.
    //------------------------------------------------------------------
    // Set default to 0, meaning no uptake
    for (int k = 0; k < num_gas; ++k)
      uptk_rate_factor[k] = 0.0;
    // H2SO4 is the ref species, so the ratio is 1
    uptk_rate_factor[igas_h2so4] = 1.0;
    // For NH3
    uptk_rate_factor[igas_nh3] = nh3_h2so4_uptake_coeff_ratio;
 
    for (int imode = 0; imode < num_mode; ++imode)
      modes_mean_std_dev[imode]=modes[imode].mean_std_dev;
    //-------------------------------------------------------------------
    // MAM currently uses a splitting method to deal with gas-aerosol
    // mass exchange. A quasi-analytical solution assuming timestep-wise
    // constant uptake rate is applied to nonvolatile species, while
    // an implicit time stepping method with adaptive step size
    // is applied to gas-phase SOA species which are assumed semi-volatile.
    // There are two different subroutines in this modules to deal
    // with the two categories of cases. The array
    // eqn_and_numerics_category flags which gas species belongs to
    // which category.
    //-------------------------------------------------------------------
    for (int k = 0; k < num_gas; ++k)
      eqn_and_numerics_category[k] = NA;
    for (int k = igas_soag_bgn; k <= igas_soag_end; ++k)
      eqn_and_numerics_category[k] = IMPL;
    eqn_and_numerics_category[igas_h2so4] = ANAL;
    eqn_and_numerics_category[igas_nh3] = ANAL;

    //-------------------------------------------------------------------
    // Determine whether specific gases will condense to specific modes
    //-------------------------------------------------------------------
    for (int igas = 0; igas < num_gas; ++igas)
      for (int imode = 0; imode < num_mode; ++imode)
        l_gas_condense_to_mode[igas][imode] = false;
    // loop through all registered gas species
    for (int igas = 0; igas < num_gas; ++igas) {
      // can this gas species condense?
      if (eqn_and_numerics_category[igas] != NA) {
        // what aerosol species does the gas become when condensing?
        const int iaer = config_.idx_gas_to_aer[igas];
        for (int imode = 0; imode < num_mode; ++imode)
          l_gas_condense_to_mode[igas][imode] =
              config_.l_mode_can_contain_species[iaer][imode] ||
              config_.l_mode_can_age[imode];
      }
    }

    // For SOAG. (igas_soag_bgn and igas_soag_end are the start- and
    // end-indices) Remove use of igas_soag_bgn but keep comments in 
    // case need to be added back 
    // uptk_rate_factor(igas_soag:igas_soagzz) =
    //   soag_h2so4_uptake_coeff_ratio
    //         igas_soag = ngas + 1
    //         igas_soagzz = ngas + nsoa
    //         uptk_rate_factor(igas_soag:igas_soagzz) =
    //         soag_h2so4_uptake_coeff_ratio
  }

  // validate -- validates the given atmospheric state and prognostics against
  // assumptions made by this implementation, returning true if the states are
  // valid, false if not
  KOKKOS_INLINE_FUNCTION
  bool validate(const AeroConfig &config, const ThreadTeam &team,
                const Atmosphere &atm, const Prognostics &progs) const {
    // Make sure relevant atmospheric quantities are physical.
    const int nk = PackInfo::num_packs(atm.num_levels());
    int violations = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, nk),
        KOKKOS_LAMBDA(int k, int &violation) {
          if ((atm.temperature(k) < 0).any() || (atm.pressure(k) < 0).any() ||
              (atm.vapor_mixing_ratio(k) < 0).any()) {
            violation = 1;
          }
        },
        violations);

    if (violations == 0) { // all clear so far
      // Check for negative mixing ratios.
      if (!progs.quantities_nonnegative(team)) {
        ++violations;
      }
    }
    return (violations > 0);
  }

  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atm,
                          const Prognostics &progs, const Diagnostics &diags,
                          const Tendencies &tends) const {
    // const int nghq = 2;  // set number of ghq points for direct ghq
    const bool l_calc_gas_uptake_coeff =
        config.calculate_gas_uptake_coefficient;
    const Real pstd = Constants::pressure_stp;
    const Real mw_h2so4 = Constants::molec_weight_h2so4;
    const Real mw_air = Constants::molec_weight_dry_air;
    const Real vol_molar_h2so4 = Constants::molec_weight_h2so4;
    const Real vol_molar_air = Constants::molec_weight_dry_air;
    const Real accom_coef_h2so4 = Constants::accom_coef_h2so4;;
    const Real r_universal = Constants::r_gas;
    const Real r_pi = Constants::pi;
    const Real beta_inp = 0; // quadrature parameter (--)
    const int nghq = config.number_gauss_points_for_integration;

    const int nk = PackInfo::num_packs(atm.num_levels());
    //====================================================================
    // Initialize the time-step mean gas concentration (explain why?)
    //====================================================================
    //Real qgas_avg[num_gas];
    //for (int k = 0; k < num_gas; ++k)
    //  qgas_avg[k] = 0.0;

    Real alnsg_aer[num_mode];
    for (int k = 0; k < num_mode; ++k)
      alnsg_aer[k] = log(modes_mean_std_dev[k]);

    const int h2so4 = igas_h2so4;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, nk), KOKKOS_CLASS_LAMBDA(int k) {
          //===============================================================
          // Calculate the reference uptake coefficient for all
          // aerosol modes using properties of the H2SO4 gas
          //===============================================================
          const Pack &temp = atm.temperature(k);
          const Pack &pmid = atm.pressure(k);
          Pack dgn_awet[num_mode] = {0, 0, 0, 0};
          for (int i = 0; i < num_mode; ++i)
            dgn_awet[i] = diags.wet_geometric_mean_diameter[i](k);
          if (l_calc_gas_uptake_coeff) {
            // do calcullation for ALL modes
            const bool l_condense_to_mode[num_mode] = {true, true, true, true};
            // initialize with zero (-> no uptake)
            Pack uptkaer_ref[num_mode] = {0, 0, 0, 0};
            gas_aer_uptkrates_1box1gas(
                l_condense_to_mode, temp, pmid, pstd, mw_h2so4, 1000 * mw_air,
                vol_molar_h2so4, vol_molar_air, accom_coef_h2so4, r_universal,
                r_pi, beta_inp, nghq, dgn_awet, alnsg_aer, uptkaer_ref);

            // -------------------------------------------------------------
            // Unit conversion: uptkrate is for number = 1 #/m3, so mult. by
            // number conc. (#/m3)
            //--------------------------------------------------------------
            Pack qnum_cur[4];
            for (int i = 0; i < 4; ++i) {
              qnum_cur[i] = progs.n_mode[i](k);
            }
            const Pack aircon = pmid / (r_universal * temp);
            for (int imode = 0; imode < num_mode; ++imode) {
              if (l_condense_to_mode[imode]) {
                uptkaer_ref[imode] *= qnum_cur[imode] * aircon;
              }
            }
            //===============================================================
            // Assign uptake rate to each gas species and each mode using the
            // ref. value uptkaer_ref calculated above and the uptake rate
            // factor specified as constants at the beginning of the module
            //===============================================================
            // gas to aerosol mass transfer rate (1/s)
            for (int igas = 0; igas < num_gas; ++igas)
              for (int imode = 0; imode < num_mode; ++imode)
                progs.uptkaer[igas][imode](k) = 0.0; // default is no uptake

            for (int igas = 0; igas < num_gas; ++igas) {
              for (int imode = 0; imode < num_mode; ++imode)
                if (l_gas_condense_to_mode[igas][imode])
                  progs.uptkaer[igas][imode](k) =
                      uptkaer_ref[imode] * uptk_rate_factor[igas];
            }
            // total uptake rate (sum of all aerosol modes) for h2so4.
            // Diagnosed for calling routine. Not used in this subroutne.
            diags.uptkrate_h2so4(k) = 0;
            for (int n = 0; n < num_mode; ++n)
              diags.uptkrate_h2so4(k) += progs.uptkaer[h2so4][n](k);
          }
          // extract gas mixing ratios
          Pack qgas_cur[num_gas], qgas_avg[num_gas], qaer_cur[num_aer][num_mode];
          for (int g = 0; g < num_gas; ++g) {
            qgas_cur[g] = progs.q_gas[g](k);
            qgas_avg[g] = 0;
          }
          for (int n = 0; n < num_mode; ++n)
            for (int g = 0; g < num_aer; ++g)
              qaer_cur[g][n] = progs.q_aero_i[n][g](k);
          for (int igas = 0; igas < num_gas; ++igas) {
            if (eqn_and_numerics_category[igas] == ANAL) {
              const int iaer = config_.idx_gas_to_aer[igas];
              const Real netprod = config_.qgas_netprod_otrproc[igas];
              Pack uptkaer[num_mode];
              for (int n = 0; n < num_mode; ++n)
                uptkaer[n] = progs.uptkaer[igas][n](k);
              Pack (&qaer)[num_mode] = qaer_cur[iaer];
              mam_gasaerexch_1subarea_1gas_nonvolatile(
                  dt, netprod, uptkaer,
                  qgas_cur[igas], qgas_avg[igas], qaer);
            }
          }
        });
  }
  KOKKOS_INLINE_FUNCTION
  void mam_gasaerexch_1subarea_1gas_nonvolatile(
    const Real dt, const Real netprod, Pack uptkaer[num_mode],
    Pack &qgas_cur, Pack &qgas_avg, Pack qaer_cur[num_mode]) const {

  }

  KOKKOS_INLINE_FUNCTION
  void gas_aer_uptkrates_1box1gas(
      const bool l_condense_to_mode[num_mode], const Pack &temp,
      const Pack &pmid, const Real pstd, const Real mw_gas, const Real mw_air,
      const Real vol_molar_gas, const Real vol_molar_air, const Real accom,
      const Real r_universal, const Real pi, const Real beta_inp,
      const int nghq, const Pack dgncur_awet[num_mode],
      const Real lnsg[num_mode],
      Pack uptkaer[num_mode]) const {
    //----------------------------------------------------------------------
    //  Computes   uptake rate parameter uptkaer[0:num_mode] =
    //  uptkrate[0:num_mode]
    //
    //                           /
    //  where      uptkrate(i) = | gas_conden_rate(Dp) n_i(lnDp) dlnDp
    //                           /
    //
    //  is the uptake rate for aerosol mode i with size distribution n_i(lnDp)
    //  and number concentration of = 1 #/m3; aernum(i) is the actual number
    //  mixing ratio of mode i in the unit of  #/kmol-air, and aircon is
    //  the air concentration in the unit of kmol/m3.
    //
    //  gas_conden_rate(D_p) = 2 * pi * gasdiffus * D_p * F(Kn,ac), with
    //          gasdiffus = gas diffusivity
    //          F(Kn,ac) = Fuchs-Sutugin correction factor
    //          Kn = Knudsen number (which is a function of Dp)
    //          ac = accomodation coefficient (constant for each gas species)
    //----------------------------------------------------------------------
    //  using Gauss-Hermite quadrature of order nghq=2
    //
    //      D_p = particle diameter (cm)
    //      x = ln(D_p)
    //      dN/dx = log-normal particle number density distribution
    //----------------------------------------------------------------------
    const Real tworootpi = 2 * Kokkos::Experimental::sqrt(pi);
    const Real root2 = Kokkos::Experimental::sqrt(2.0);
    const Real one = 1.0;
    const Real two = 2.0;

    // Dick's old version
    // integer, parameter :: nghq = 2
    // real(wp), save :: xghq(nghq), wghq(nghq) ! quadrature abscissae and
    // weights data xghq / 0.70710678, -0.70710678 / data wghq / 0.88622693,
    // 0.88622693 /

    // choose
    // nghq-----------------------------------------------------------------
    const Kokkos::Array<Real, 20> xghq_20 = {
        -5.3874808900112,  -4.6036824495507, -3.9447640401156, -3.3478545673832,
        -2.7888060584281,  -2.2549740020893, -1.7385377121166, -1.2340762153953,
        -0.73747372854539, -0.2453407083009, 0.2453407083009,  0.73747372854539,
        1.2340762153953,   1.7385377121166,  2.2549740020893,  2.7888060584281,
        3.3478545673832,   3.9447640401156,  4.6036824495507,  5.3874808900112};
    const Kokkos::Array<Real, 20> wghq_20 = {
        2.229393645534e-13, 4.399340992273e-10, 1.086069370769e-7,
        7.80255647853e-6,   2.283386360164e-4,  0.003243773342238,
        0.024810520887464,  0.10901720602002,   0.28667550536283,
        0.46224366960061,   0.46224366960061,   0.28667550536283,
        0.10901720602002,   0.024810520887464,  0.003243773342238,
        2.283386360164e-4,  7.80255647853e-6,   1.086069370769e-7,
        4.399340992273e-10, 2.229393645534e-13};
    const Kokkos::Array<Real, 10> xghq_10 = {
        -3.436159118837737603327,  -2.532731674232789796409,
        -1.756683649299881773451,  -1.036610829789513654178,
        -0.3429013272237046087892, 0.3429013272237046087892,
        1.036610829789513654178,   1.756683649299881773451,
        2.532731674232789796409,   3.436159118837737603327};
    const Kokkos::Array<Real, 10> wghq_10 = {
        7.64043285523262062916e-6,  0.001343645746781232692202,
        0.0338743944554810631362,   0.2401386110823146864165,
        0.6108626337353257987836,   0.6108626337353257987836,
        0.2401386110823146864165,   0.03387439445548106313616,
        0.001343645746781232692202, 7.64043285523262062916E-6};
    const Kokkos::Array<Real, 4> xghq_4 = {-1.6506801238858, -0.52464762327529,
                                           0.52464762327529, 1.6506801238858};
    const Kokkos::Array<Real, 4> wghq_4 = {0.081312835447245, 0.8049140900055,
                                           0.8049140900055, 0.081312835447245};
    const Kokkos::Array<Real, 2> xghq_2 = {-7.0710678118654746e-01,
                                           7.0710678118654746e-01};
    const Kokkos::Array<Real, 2> wghq_2 = {8.8622692545275794e-01,
                                           8.8622692545275794e-01};
    Real const *xghq = nullptr;
    Real const *wghq = nullptr;
    if (20 == nghq) {
      xghq = xghq_20.data();
      wghq = wghq_20.data();
    } else if (10 == nghq) {
      xghq = xghq_10.data();
      wghq = wghq_10.data();
    } else if (4 == nghq) {
      xghq = xghq_4.data();
      wghq = wghq_4.data();
    } else if (2 == nghq) {
      xghq = xghq_2.data();
      wghq = wghq_2.data();
    } else {
      printf(
          "nghq integration option is not available: %d, "
          "valid are 20, 10, 4, and 2\n",
          nghq);
      Kokkos::abort("Invalid integration order requested.");
    }
    //-----------------------------------------------------------------------

    // pressure (atmospheres)
    const Pack p_in_atm = pmid / pstd;
    // gas diffusivity (m2/s)
    const Pack gasdiffus = gas_diffusivity(temp, p_in_atm, mw_gas, mw_air,
                                           vol_molar_gas, vol_molar_air);
    // gas mean free path (m)
    const Pack gasfreepath =
        3.0 * gasdiffus / mean_molecular_speed(temp, mw_gas, r_universal, pi);

    const Real accomxp283 = accom * 0.283;
    const Real accomxp75 = accom * 0.75;

    // outermost loop over all modes
    for (int n = 0; n < num_mode; ++n) {
      const Pack lndpgn = ekat::log(dgncur_awet[n]); // (m)

      // beta = dln(uptake_rate)/dln(D_p)
      //      = 2.0 in free molecular regime, 1.0 in continuum regime
      // if uptake_rate ~= a * (D_p**beta), then the 2 point quadrature 
      // is very accurate
      Pack beta;
      if (abs(beta_inp - 1.5) > 0.5) {
        // D_p = dgncur_awet(n) * ekat::exp( 1.5*(lnsg[n]**2) )
        const Pack D_p = dgncur_awet[n];
        const Pack knudsen = two * gasfreepath / D_p;
        // tmpa = dln(fuchs_sutugin)/d(knudsen)
        const Pack tmpa =
            one / (one + knudsen) -
            (two * knudsen + one + accomxp283) /
                (knudsen * (knudsen + one + accomxp283) + accomxp75);
        beta = one - knudsen * tmpa;
        beta = max(one, min(two, beta));
      } else {
        beta = beta_inp;
      }

      const Pack constant =
          tworootpi *
          ekat::exp(beta * lndpgn + 0.5 * ekat::pow(beta * lnsg[n], 2.0));

      // sum over gauss-hermite quadrature points
      Pack sumghq = 0.0;
      for (int iq = 0; iq < nghq; ++iq) {
        const Pack lndp =
            lndpgn + beta * lnsg[n] * lnsg[n] + root2 * lnsg[n] * xghq[iq];
        const Pack D_p = ekat::exp(lndp);

        const Pack hh = fuchs_sutugin(D_p, gasfreepath, accomxp283, accomxp75);

        sumghq += wghq[iq] * D_p * hh / ekat::pow(D_p, beta);
      }
      const Pack uptkrate = constant * gasdiffus *
                            sumghq; // gas-to-aerosol mass transfer rates (1/s)
                                    // for number concentration = 1 #/m3

      // --------------------------------------------------------------------
      // Unit of uptkrate is for number = 1 #/m3.
      // --------------------------------------------------------------------
      uptkaer[n] =
          l_condense_to_mode[n] ? uptkrate : 0.0; // zero means no uptake
    }
  }

  //------------------------------------------------------------------------
  // gas_diffusivity       ! (m2/s)
  KOKKOS_INLINE_FUNCTION
  Pack gas_diffusivity(
      const Pack &T_in_K,   // temperature (K)
      const Pack &p_in_atm, // pressure (atmospheres)
      const Real mw_gas,    // molec. weight of the condensing gas (g/mol)
      const Real mw_air,    // molec. weight of air (g/mol)
      const Real vd_gas,    // molec. diffusion volume of the condensing gas
      const Real vd_air) const { // molec. diffusion volume of air

    const Real onethird = 1.0 / 3.0;

    const Pack gas_diffusivity =
        (1.0e-7 * ekat::pow(T_in_K, 1.75) *
         Kokkos::Experimental::sqrt(1.0 / mw_gas + 1.0 / mw_air)) /
        (p_in_atm * Kokkos::Experimental::pow(
                        Kokkos::Experimental::pow(vd_gas, onethird) +
                            Kokkos::Experimental::pow(vd_air, onethird),
                        2.0));

    return gas_diffusivity;
  }
  //-----------------------------------------------------------------------
  //  mean_molecular_speed    ! (m/s)
  KOKKOS_INLINE_FUNCTION
  Pack mean_molecular_speed(const Pack &temp,       // temperature (K)
                            const Real rmw,         // molec. weight (g/mol)
                            const Real r_universal, // universal gas constant
                            const Real pi) const {
    const Pack mean_molecular_speed =
        ekat::sqrt(8.0 * r_universal * temp / (pi * rmw));

    return mean_molecular_speed;
  }
  //------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  Pack fuchs_sutugin(const Pack &D_p, const Pack &gasfreepath,
                     const Real accomxp283, const Real accomxp75) const {
    const Pack knudsen = 2.0 * gasfreepath / D_p;

    // fkn = ( 0.75*accomcoef*(1. + xkn) ) /
    //       ( xkn*xhn + xkn + 0.283*xkn*accomcoef + 0.75*accomcoef )

    const Pack fuchs_sutugin =
        (accomxp75 * (1.0 + knudsen)) /
        (knudsen * (knudsen + 1.0 + accomxp283) + accomxp75);

    return fuchs_sutugin;
  }

private:
  // Gas-Aerosol-Exchange-specific configuration
  Config config_;
};

} // namespace mam4

#endif
