#ifndef MAM4XX_GASAEREXCH_HPP
#define MAM4XX_GASAEREXCH_HPP

#include <Kokkos_Array.hpp>
#include <haero/atmosphere.hpp>
#include <haero/haero.hpp>

#include "aero_config.hpp"

namespace mam4 {

using Atmosphere = haero::Atmosphere;
using Constants = haero::Constants;
using IntPack = haero::IntPackType;
using PackType = haero::PackType;
using PackInfo = haero::PackInfo;
using Real = haero::Real;
using ThreadTeam = haero::ThreadTeam;

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
  enum {
    NA,
    ANAL,
    IMPL
  };

  static constexpr Real mw_h2so4 = Constants::molec_weight_h2so4;

  // ratio of gas uptake coeff w.r.t. that of h2so4
  static constexpr Real soag_h2so4_uptake_coeff_ratio = 0.81;  // for SOAG
  static constexpr Real nh3_h2so4_uptake_coeff_ratio = 2.08;   // for NH3

  haero::DeviceType::view_2d<bool> l_mode_can_contain_species;
  haero::DeviceType::view_2d<bool> l_gas_condense_to_mode;
  haero::DeviceType::view_1d<bool> l_mode_can_age;
  haero::DeviceType::view_1d<int> idx_gas_to_aer;
  haero::DeviceType::view_1d<int> eqn_and_numerics_category;
  haero::DeviceType::view_1d<Real> uptk_rate_factor;
  haero::DeviceType::view_1d<Real> qgas_cur;
  haero::DeviceType::view_1d<Real> qgas_avg;
  haero::DeviceType::view_1d<Real> qgas_netprod_otrproc;
  haero::DeviceType::view_2d<PackType> uptkaer;
  haero::DeviceType::view_1d<Real> alnsg_aer;
  haero::DeviceType::view_1d<int> mode_aging_optaa;

 public:
  // process-specific configuration data (if any)
  struct Config {
    Config() {}
    Config(const Config &) = default;
    ~Config() = default;
    Config& operator=(const Config&) = default;
    haero::DeviceType::view_2d<bool> l_mode_can_contain_species;
    haero::DeviceType::view_1d<bool> l_mode_can_age;
  };

  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 gas/aersol exchange"; }

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config()) {
    Kokkos::resize(l_mode_can_contain_species, num_aer, num_mode);
    Kokkos::resize(l_gas_condense_to_mode, num_gas, num_mode);
    Kokkos::resize(l_mode_can_age, num_aer);
    Kokkos::resize(idx_gas_to_aer, num_gas);
    Kokkos::resize(eqn_and_numerics_category, num_gas);
    Kokkos::resize(uptk_rate_factor, num_gas);

    Kokkos::resize(qgas_cur, num_gas);
    Kokkos::resize(qgas_avg, num_gas);
    Kokkos::resize(qgas_netprod_otrproc, num_gas);

    Kokkos::resize(uptkaer, num_gas,
                   num_mode);  // gas to aerosol mass transfer rate (1/s)

    Kokkos::resize(alnsg_aer, num_mode);
    Kokkos::resize(mode_aging_optaa, num_mode);

    EKAT_ASSERT(l_mode_can_contain_species.extent(0) == process_config.l_mode_can_contain_species.extent(0));
    EKAT_ASSERT(l_mode_can_contain_species.extent(1) == process_config.l_mode_can_contain_species.extent(1));
    Kokkos::parallel_for( 1, KOKKOS_CLASS_LAMBDA(int) { 
      for (int iaer = 0; iaer < num_aer; ++iaer)
        for (int imode = 0; imode < num_mode; ++imode)
          l_mode_can_contain_species(iaer,num_mode) = 
	    process_config.l_mode_can_contain_species(iaer,num_mode);
    });
    EKAT_ASSERT(l_mode_can_age.extent(0) == process_config.l_mode_can_age.extent(0));
    Kokkos::parallel_for( 1, KOKKOS_CLASS_LAMBDA(int) { 
      for (int iaer = 0; iaer < num_aer; ++iaer)
        l_mode_can_age(iaer) = process_config.l_mode_can_age(iaer);
    });
    //------------------------------------------------------------------
    // MAM currently assumes that the uptake rate of other gases
    // are proportional to the uptake rate of sulfuric acid gas (H2SO4).
    // Here the array uptk_rate_factor contains the uptake rate ratio
    // w.r.t. that of H2SO4.
    //------------------------------------------------------------------
    // Set default to 0, meaning no uptake
    Kokkos::parallel_for(
        num_gas, KOKKOS_CLASS_LAMBDA(int k) { uptk_rate_factor(k) = 0.0; });

    const int h2so4 = igas_h2so4;
    const int nh3 = igas_nh3;
    Kokkos::parallel_for(
        1, KOKKOS_CLASS_LAMBDA(int) {
          // H2SO4 is the ref species, so the ratio is 1
          uptk_rate_factor(h2so4) = 1.0;
          // For NH3
          uptk_rate_factor(nh3) = nh3_h2so4_uptake_coeff_ratio;
        });

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
    Kokkos::parallel_for(
        num_gas, KOKKOS_CLASS_LAMBDA(int k) { 
      if (k==igas_h2so4)                                eqn_and_numerics_category(k) = ANAL;
      else if (igas_soag_bgn <= k && k <= igas_soag_end) eqn_and_numerics_category(k) = IMPL;
      else if (k==igas_nh3)                             eqn_and_numerics_category(k) = ANAL;
      else                                              eqn_and_numerics_category(k) = NA;
    });
 
 
    //-------------------------------------------------------------------
    // Determine whether specific gases will condense to specific modes
    //-------------------------------------------------------------------
    Kokkos::parallel_for( 1, KOKKOS_CLASS_LAMBDA(int) { 
      for (int igas = 0; igas < num_gas; ++igas)
        for (int imode = 0; imode < num_mode; ++imode)
          l_gas_condense_to_mode(igas,num_mode) = false; 
    });
 
    Kokkos::parallel_for( 1, KOKKOS_CLASS_LAMBDA(int) { 
      for (int igas = 0; igas < num_gas; ++igas) {   // loop through all registered gas species
        if (eqn_and_numerics_category(igas) != NA) { // this gas species can condense
          const int iaer = idx_gas_to_aer(igas);     // what aerosol species does the gas become when condensing?
          for (int imode = 0; imode < num_mode; ++imode)
            l_gas_condense_to_mode(igas,imode) = 
	      l_mode_can_contain_species(iaer,imode) || l_mode_can_age(imode);
        }
      }
    });

    // For SOAG. (igas_soag_bgn and igas_soag_end are the start- and
    // end-indices) Remove use of igas_soag_bgn but keep comments in case need
    // to be added back uptk_rate_factor(igas_soag:igas_soagzz) =
    // soag_h2so4_uptake_coeff_ratio
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

    if (violations == 0) {  // all clear so far
      // Check for negative mixing ratios.
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, nk),
          KOKKOS_LAMBDA(int k, int &violation) {
            for (int mode = 0; mode < 4; ++mode) {  // check mode mmrs
              if ((progs.n_mode[mode](k) < 0).any()) {
                ++violation;
              } else {
                for (int spec = 0; spec < 7; ++spec) {  // check aerosol mmrs
                  if ((progs.q_aero[mode][spec](k) < 0).any()) {
                    ++violation;
                    break;
                  }
                }
              }
              if (violation > 0) break;
            }
            if (violation == 0) {
              for (int gas = 0; gas < 13; ++gas) {  // check gas mmrs
                if ((progs.q_gas[gas](k) < 0).any()) ++violation;
              }
            }
          },
          violations);
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

    const int nk = PackInfo::num_packs(atm.num_levels());
    //=========================================================================
    // Initialize the time-step mean gas concentration (explain why?)
    //=========================================================================
    for (int k = 0; k < num_gas; ++k) qgas_avg(k) = 0.0;

    Kokkos::Array<Real, num_mode> alnsg_aer;
    for (int k = 0; k < num_mode; ++k)
      alnsg_aer[k] = log(modes[k].mean_std_dev);

    const int h2so4 = igas_h2so4;
    if (l_calc_gas_uptake_coeff) {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, nk), KOKKOS_CLASS_LAMBDA(int k) {
            //=========================================================================
            // Calculate the reference uptake coefficient for all aerosol modes
            // using properties of the H2SO4 gas
            //=========================================================================
            Kokkos::Array<PackType, num_mode> uptkaer_ref = {
                0, 0, 0, 0};  // initialize with zero (-> no uptake)
            Kokkos::Array<bool, num_mode> l_condense_to_mode = {
                true, true, true, true};  // do calcullation for ALL modes
            // const int igas = igas_h2so4; // use properties of the H2SO4 gas
            const PackType &temp = atm.temperature(k);
            const PackType &pmid = atm.pressure(k);
            Kokkos::Array<PackType, num_mode> dgn_awet = {0, 0, 0, 0};
            for (int i = 0; i < num_mode; ++i)
              dgn_awet[i] = diags.wet_geometric_mean_diameter[i](k);
            const Real pstd = Constants::pressure_stp;
            const Real mw_h2so4 = Constants::molec_weight_h2so4;
            const Real mw_air = Constants::molec_weight_dry_air;
            const Real vol_molar_h2so4 = Constants::molec_weight_h2so4;
            const Real vol_molar_air = Constants::molec_weight_dry_air;
            const Real accom_coef_h2so4 = 0.65;
            const Real r_universal = Constants::r_gas;
            const Real r_pi = Constants::pi;
            const Real beta_inp = 0;  // quadrature parameter (--)
            const int nghq = config.number_gauss_points_for_integration;

            gas_aer_uptkrates_1box1gas(
                l_condense_to_mode, temp, pmid, pstd, mw_h2so4, 1000 * mw_air,
                vol_molar_h2so4, vol_molar_air, accom_coef_h2so4, r_universal,
                r_pi, beta_inp, nghq, dgn_awet, alnsg_aer, uptkaer_ref);

            //========================================================================================
            // Assign uptake rate to each gas species and each mode using the
            // ref. value uptkaer_ref calculated above and the uptake rate
            // factor specified as constants at the beginning of the module
            //========================================================================================
            for (int igas = 0; igas < num_gas; ++igas)
              for (int imode = 0; imode < num_mode; ++imode)
                uptkaer(igas, imode) = 0.0;  // default is no uptake
					     
            //========================================================================================
            // Assign uptake rate to each gas species and each mode using the ref. value uptkaer_ref
            // calculated above and the uptake rate factor specified as constants at the
            // beginning of the module
            //========================================================================================

            for (int igas = 0; igas < num_gas; ++igas)
              for (int imode = 0; imode < num_mode; ++imode)
                if (l_gas_condense_to_mode(igas, imode))
                  uptkaer(igas, imode) =
                      uptkaer_ref[imode] * uptk_rate_factor(igas);

            // total uptake rate (sum of all aerosol modes) for h2so4.
            // Diagnosed for calling routine. Not used in this subroutne.
            diags.uptkrate_h2so4(k) = 0;
            for (int n = 0; n < num_mode; ++n)
              diags.uptkrate_h2so4(k) += uptkaer(h2so4, n);
          });
    }
  }

  KOKKOS_INLINE_FUNCTION
  void gas_aer_uptkrates_1box1gas(
      const Kokkos::Array<bool, num_mode> &l_condense_to_mode,
      const PackType &temp, const PackType &pmid, const Real pstd,
      const Real mw_gas, const Real mw_air, const Real vol_molar_gas,
      const Real vol_molar_air, const Real accom, const Real r_universal,
      const Real pi, const Real beta_inp, const int nghq,
      const Kokkos::Array<PackType, num_mode> &dgncur_awet,
      const Kokkos::Array<Real, num_mode> &lnsg,
      Kokkos::Array<PackType, num_mode> &uptkaer) const {
    //--------------------------------------------------------------------------------
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
    //--------------------------------------------------------------------------------
    //  using Gauss-Hermite quadrature of order nghq=2
    //
    //      D_p = particle diameter (cm)
    //      x = ln(D_p)
    //      dN/dx = log-normal particle number density distribution
    //--------------------------------------------------------------------------------
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
          "nghq integration option is not available: %d, valid are 20, 10, 4, "
          "and 2\n",
          nghq);
      Kokkos::abort("Invalid integration order requested.");
    }
    //---------------------------------------------------------------------------------------------

    // pressure (atmospheres)
    const PackType p_in_atm = pmid / pstd;
    // gas diffusivity (m2/s)
    const PackType gasdiffus = gas_diffusivity(temp, p_in_atm, mw_gas, mw_air,
                                               vol_molar_gas, vol_molar_air);
    // gas mean free path (m)
    const PackType gasfreepath =
        3.0 * gasdiffus / mean_molecular_speed(temp, mw_gas, r_universal, pi);

    const Real accomxp283 = accom * 0.283;
    const Real accomxp75 = accom * 0.75;

    // outermost loop over all modes
    for (int n = 0; n < num_mode; ++n) {
      const PackType lndpgn = ekat::log(dgncur_awet[n]);  // (m)

      // beta = dln(uptake_rate)/dln(D_p)
      //      = 2.0 in free molecular regime, 1.0 in continuum regime
      // if uptake_rate ~= a * (D_p**beta), then the 2 point quadrature is very
      // accurate
      PackType beta;
      if (abs(beta_inp - 1.5) > 0.5) {
        // D_p = dgncur_awet(n) * ekat::exp( 1.5*(lnsg[n]**2) )
        const PackType D_p = dgncur_awet[n];
        const PackType knudsen = two * gasfreepath / D_p;
        // tmpa = dln(fuchs_sutugin)/d(knudsen)
        const PackType tmpa =
            one / (one + knudsen) -
            (two * knudsen + one + accomxp283) /
                (knudsen * (knudsen + one + accomxp283) + accomxp75);
        beta = one - knudsen * tmpa;
        beta = max(one, min(two, beta));
      } else {
        beta = beta_inp;
      }

      const PackType constant =
          tworootpi *
          ekat::exp(beta * lndpgn + 0.5 * ekat::pow(beta * lnsg[n], 2.0));

      // sum over gauss-hermite quadrature points
      PackType sumghq = 0.0;
      for (int iq = 0; iq < nghq; ++iq) {
        const PackType lndp =
            lndpgn + beta * lnsg[n] * lnsg[n] + root2 * lnsg[n] * xghq[iq];
        const PackType D_p = ekat::exp(lndp);

        const PackType hh =
            fuchs_sutugin(D_p, gasfreepath, accomxp283, accomxp75);

        sumghq += wghq[iq] * D_p * hh / ekat::pow(D_p, beta);
      }
      const PackType uptkrate =
          constant * gasdiffus *
          sumghq;  // gas-to-aerosol mass transfer rates (1/s)
                   // for number concentration = 1 #/m3

      // -----------------------------------------------------------------------------------
      // Unit of uptkrate is for number = 1 #/m3.
      // -----------------------------------------------------------------------------------
      uptkaer[n] =
          l_condense_to_mode[n] ? uptkrate : 0.0;  // zero means no uptake
    }
  }

  //--------------------------------------------------------------------------------
  // gas_diffusivity       ! (m2/s)
  KOKKOS_INLINE_FUNCTION
  PackType gas_diffusivity(
      const PackType &T_in_K,    // temperature (K)
      const PackType &p_in_atm,  // pressure (atmospheres)
      const Real mw_gas,         // molec. weight of the condensing gas (g/mol)
      const Real mw_air,         // molec. weight of air (g/mol)
      const Real vd_gas,  // molec. diffusion volume of the condensing gas
      const Real vd_air) const {  // molec. diffusion volume of air

    const Real onethird = 1.0 / 3.0;

    const PackType gas_diffusivity =
        (1.0e-7 * ekat::pow(T_in_K, 1.75) *
         Kokkos::Experimental::sqrt(1.0 / mw_gas + 1.0 / mw_air)) /
        (p_in_atm * Kokkos::Experimental::pow(
                        Kokkos::Experimental::pow(vd_gas, onethird) +
                            Kokkos::Experimental::pow(vd_air, onethird),
                        2.0));

    return gas_diffusivity;
  }
  //--------------------------------------------------------------------------------
  //  mean_molecular_speed    ! (m/s)
  KOKKOS_INLINE_FUNCTION
  PackType mean_molecular_speed(
      const PackType &temp,    // temperature (K)
      const Real rmw,          // molec. weight (g/mol)
      const Real r_universal,  // universal gas constant
      const Real pi) const {
    const PackType mean_molecular_speed =
        ekat::sqrt(8.0 * r_universal * temp / (pi * rmw));

    return mean_molecular_speed;
  }
  //--------------------------------------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  PackType fuchs_sutugin(const PackType &D_p, const PackType &gasfreepath,
                         const Real accomxp283, const Real accomxp75) const {
    const PackType knudsen = 2.0 * gasfreepath / D_p;

    // fkn = ( 0.75*accomcoef*(1. + xkn) ) /
    //       ( xkn*xhn + xkn + 0.283*xkn*accomcoef + 0.75*accomcoef )

    const PackType fuchs_sutugin =
        (accomxp75 * (1.0 + knudsen)) /
        (knudsen * (knudsen + 1.0 + accomxp283) + accomxp75);

    return fuchs_sutugin;
  }
};

}  // namespace mam4

#endif
