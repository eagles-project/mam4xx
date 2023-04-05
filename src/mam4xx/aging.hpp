// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AGING_HPP
#define MAM4XX_AGING_HPP
#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>

namespace mam4 {

class Aging {

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

  // In E3SM this is read in from an input file and would be 8
  // In mam_refactor it is defined in phys_control.F90 as 3.
  static constexpr Real n_so4_monolayers_pcage = 3.0;
  static constexpr Real dr_so4_monolayers_pcage =
      n_so4_monolayers_pcage * 4.76e-10;

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

namespace aging {

//------------------------------------------------------------------------
// calculate fractions of aged pom/bc to be transferred to accum mode, aerosol
// change due to condenstion and coagulation
KOKKOS_INLINE_FUNCTION
void mam_pcarbon_aging_frac(
    Real dgn_a[AeroConfig::num_modes()], // dry geometric mean diameter of
                                         // number distribution [m]
    Real qaer_cur[AeroConfig::num_aerosol_ids()]
                 [AeroConfig::num_modes()], // aerosol mass mixing ratio
                                            // [mol/mol]
    Real qaer_del_cond[AeroConfig::num_aerosol_ids()]
                      [AeroConfig::num_modes()], // change of aerosol mass
                                                 // mixing ratio due to
                                                 // condensation [mol/mol]
    Real qaer_del_coag_in[AeroConfig::num_aerosol_ids()]
                         [AeroConfig::max_agepair()], // change of aerosol mass
                                                      // mixing ratio due to
                                                      // coagulation
    Real &xferfrac_pcage, // fraction of aged pom/bc transferred to accum
                          // [unitless]
    Real
        &frac_cond, // fraction of aerosol change due to condensation [unitless]
    Real &
        frac_coag) { // fraction of aerosol change due to coagulation [unitless]

  const int ipair = 0;
  const int iaer_so4 = static_cast<int>(AeroId::SO4);
  const int iaer_soa = static_cast<int>(AeroId::SOA);
  const int iaer_bc = static_cast<int>(AeroId::BC);
  const int iaer_pom = static_cast<int>(AeroId::POM);
  const int iaer_mom = static_cast<int>(AeroId::MOM);

  const int imom_pc = static_cast<int>(ModeIndex::PrimaryCarbon);

  const Real _molecular_weight_soa = 150 / 1000.0;
  // FIXME. MW for SO4 is not a standard MW. (BAD CONSTANT)
  const Real _molecular_weight_so4 = 115 / 1000.0;
  const Real _molecular_weight_pom = 150 / 1000.0;

  // Compute the aerosol volume per mole
  const Real so4_vol =
      _molecular_weight_so4 * 1000.0 / aero_species(iaer_so4).density;
  const Real soa_vol =
      _molecular_weight_soa * 1000.0 / aero_species(iaer_soa).density;
  const Real bc_vol = aero_species(iaer_bc).molecular_weight * 1000.0 /
                      aero_species(iaer_bc).density;
  const Real pom_vol =
      _molecular_weight_pom * 1000.0 / aero_species(iaer_pom).density;
  const Real mom_vol = aero_species(iaer_mom).molecular_weight * 1000.0 /
                       aero_species(iaer_mom).density;

  // (Bad Constants) for hygroscopicitiy
  constexpr Real hygro_soa = 0.14000000000000001;
  constexpr Real hygro_so4 = 0.50700000000000001;
  const Real fac_m2v_eqvhyg_aer = soa_vol * hygro_soa / hygro_so4;

  Real qaer_del_cond_tmp =
      qaer_del_cond[iaer_so4][imom_pc] * so4_vol +
      qaer_del_cond[iaer_soa][imom_pc] * fac_m2v_eqvhyg_aer;

  const Real qaer_del_coag_tmp =
      qaer_del_coag_in[iaer_so4][ipair] * so4_vol +
      qaer_del_coag_in[iaer_soa][ipair] * fac_m2v_eqvhyg_aer;

  qaer_del_cond_tmp = haero::max(qaer_del_cond_tmp, 1e-35);

  frac_cond = qaer_del_cond_tmp /
              (qaer_del_cond_tmp + haero::max(qaer_del_coag_tmp, 0.0));

  frac_coag = 1.0 - frac_cond;

  // for default MAM4 only so4 and soa contribute to aging
  const Real vol_shell = qaer_cur[iaer_so4][imom_pc] * so4_vol +
                         qaer_cur[iaer_soa][imom_pc] * fac_m2v_eqvhyg_aer;

  const int spec_modes[3] = {iaer_bc, iaer_pom, iaer_mom};
  const Real core_volumes[3] = {bc_vol, pom_vol, mom_vol};
  Real vol_core = 0.0;
  for (int mi = 0; mi < 3; ++mi) {
    const int ispec = spec_modes[mi];
    vol_core += qaer_cur[ispec][imom_pc] * core_volumes[mi];
  }

  const Real fac_volsfc = haero::exp(
      2.5 * haero::square(haero::log(mam4::modes(imom_pc).mean_std_dev)));

  const Real xferfrac_max =
      1.0 - 10.0 * std::numeric_limits<Real>::epsilon(); //  1-eps

  Real xferfrac_tmp1 = vol_shell * dgn_a[imom_pc] * fac_volsfc;
  Real xferfrac_tmp2 =
      haero::max(6.0 * Aging::dr_so4_monolayers_pcage * vol_core, 0.0);

  if (xferfrac_tmp1 >= xferfrac_tmp2) {
    xferfrac_pcage = xferfrac_max;
  } else {
    xferfrac_pcage = haero::min(xferfrac_tmp1 / xferfrac_tmp2, xferfrac_max);
  }
}

//------------------------------------------------------------------------
// transfer mass/number of aged pom and bc from pcarbon to accum mode
// adjust the change of aerosol mass/number due to condenations/coagulation
// in pcarbon anc accum mode
KOKKOS_INLINE_FUNCTION
void transfer_aged_pcarbon_to_accum(
    const int nsrc,            // pcarbon mode index [unitless]
    const int ndest,           // accum mode index [unitless]
    const Real xferfrac_pcage, // fraction of aged pom/bc transferred to accum
                               // [unitless]
    const Real frac_cond, // fraction of aerosol mass change due to condensation
                          // [unitless]
    const Real frac_coag, // fraction of aerosol mass change due to coagulation
                          // [unitless]
    Real qaer_cur[AeroConfig::num_modes()], // aerosol mass or number mixing
                                            // ratio [mol/mol]/[#/kmol]
    Real qaer_del_cond[AeroConfig::num_modes()],   // change of aerosol mass or
                                                   // number mixing ratio due to
                                                   // condensation
                                                   // [mol/mol]/[#/kmol]
    Real qaer_del_coag[AeroConfig::num_modes()]) { // change of aerosol mass or
                                                   // number mixing ratio due to
                                                   // coagulation
                                                   // [mol/mol]/[#/kmol]

  Real q_tmp = qaer_cur[nsrc] * xferfrac_pcage;

  qaer_cur[nsrc] -= q_tmp;
  qaer_cur[ndest] += q_tmp;

  qaer_del_cond[nsrc] -= q_tmp * frac_cond;
  qaer_del_cond[ndest] += q_tmp * frac_cond;

  qaer_del_coag[nsrc] -= q_tmp * frac_coag;
  qaer_del_coag[ndest] += q_tmp * frac_coag;
}

//------------------------------------------------------------------------
// transfer mass of aerosols contributing to aging (i.e., so4, soa)
//  from pcarbon to accum mode
//  adjust the change of aerosol mass/number due to condenations/coagulation
//  in pcarbon and accum mode

KOKKOS_INLINE_FUNCTION
void transfer_cond_coag_mass_to_accum(
    const int nsrc,                         // pcarbon mode index [unitless]
    const int ndest,                        // accum mode index [unitless]
    Real qaer_cur[AeroConfig::num_modes()], // aerosol mass mixing ratio
                                            // [mol/mol]
    Real qaer_del_cond[AeroConfig::num_modes()],   // change of aerosol mass
                                                   // mixing ratio due to
                                                   // condensation [mol/mol]
    Real qaer_del_coag[AeroConfig::num_modes()]) { // change of aerosol mass
                                                   // mixing ratio due to
                                                   // coagulation [mol/mol]

  qaer_cur[ndest] += qaer_cur[nsrc];
  qaer_del_cond[ndest] += qaer_del_cond[nsrc];
  qaer_del_coag[ndest] += qaer_del_coag[nsrc];

  qaer_cur[nsrc] = 0.0;
  qaer_del_cond[nsrc] = 0.0;
  qaer_del_coag[nsrc] = 0.0;
}

KOKKOS_INLINE_FUNCTION
void mam_pcarbon_aging_1subarea(
    Real dgn_a[AeroConfig::num_modes()],    // dry geometric mean diameter of
                                            // number distribution [m]
    Real qnum_cur[AeroConfig::num_modes()], // aerosol number mixing ratio
                                            // [#/kmol]
    Real qnum_del_cond[AeroConfig::num_modes()], // change of aerosol number
                                                 // mixing ratio due to
                                                 // condensation [#/kmol]
    Real qnum_del_coag[AeroConfig::num_modes()], // change of aerosol number
                                                 // mixing ratio due to
                                                 // coagulation [#/kmol]
    Real qaer_cur[AeroConfig::num_aerosol_ids()]
                 [AeroConfig::num_modes()], // aerosol mass mixing ratio
                                            // [mol/mol]
    Real qaer_del_cond[AeroConfig::num_aerosol_ids()]
                      [AeroConfig::num_modes()], // change of aerosol mass
                                                 // mixing ratio due to
                                                 // condensation [mol/mol]
    Real qaer_del_coag[AeroConfig::num_aerosol_ids()]
                      [AeroConfig::num_modes()], // change of aerosol mass
                                                 // mixing ratio due to
                                                 // coagulation [mol/mol]
    Real qaer_del_coag_in[AeroConfig::num_aerosol_ids()]
                         [AeroConfig::max_agepair()]) { //  change of aerosol
                                                        //  mass
                                                        //  mixing ratio due to
  //  coagulation from subrountine
  //  mam_coag_1subarea [mol/mol]

  Real xferfrac_pcage, frac_cond, frac_coag;

  const int nsrc = static_cast<int>(ModeIndex::PrimaryCarbon);
  const int ndest = static_cast<int>(ModeIndex::Accumulation);

  mam_pcarbon_aging_frac(dgn_a, qaer_cur, qaer_del_cond, qaer_del_coag_in,
                         xferfrac_pcage, frac_cond, frac_coag);
  // Note, there are probably optimizations to be done here, closely following
  // the Fortran code required extra unpacking of arrays.

  // MAM4 pcarbon mode only has pom, bc, mom, lmap only has index (>0) for these
  // species species is pom or bc transfer the aged fraction to accum mode
  //  include this transfer change in the cond and/or coag change (for mass
  //  budget)

  static constexpr int num_pcarbon_to_accum = 3;
  static constexpr int num_cond_coag_to_accum = 4;

  static constexpr int indx_aer_pcarbon_to_accum[num_pcarbon_to_accum] = {
      static_cast<int>(AeroId::POM), static_cast<int>(AeroId::BC),
      static_cast<int>(AeroId::MOM)};

  static constexpr int indx_aer_cond_coag_to_accum[num_cond_coag_to_accum] = {
      static_cast<int>(AeroId::SOA), static_cast<int>(AeroId::SO4),
      static_cast<int>(AeroId::NaCl), static_cast<int>(AeroId::DST)};

  Real qaer_cur_modes[AeroConfig::num_modes()];
  Real qaer_del_cond_modes[AeroConfig::num_modes()];
  Real qaer_del_coag_modes[AeroConfig::num_modes()];

  for (int a = 0; a < num_pcarbon_to_accum; ++a) {

    const int ispec = indx_aer_pcarbon_to_accum[a];

    // Pack mode information per aerosol
    for (int imode = 0; imode < AeroConfig::num_modes(); imode++) {
      qaer_cur_modes[imode] = qaer_cur[ispec][imode];
      qaer_del_cond_modes[imode] = qaer_del_cond[ispec][imode];
      qaer_del_coag_modes[imode] = qaer_del_coag[ispec][imode];
    }

    // species is pom or bc
    // transfer the aged fraction to accum mode
    // include this transfer change in the cond and/or coag change (for mass
    // budget)

    transfer_aged_pcarbon_to_accum(nsrc, ndest, xferfrac_pcage, frac_cond,
                                   frac_coag, qaer_cur_modes,
                                   qaer_del_cond_modes, qaer_del_coag_modes);

    // Unpack mode information per aerosol
    for (int imode = 0; imode < AeroConfig::num_modes(); imode++) {
      qaer_cur[ispec][imode] = qaer_cur_modes[imode];
      qaer_del_cond[ispec][imode] = qaer_del_cond_modes[imode];
      qaer_del_coag[ispec][imode] = qaer_del_coag_modes[imode];
    }
  }

  // species is soa, so4, or nh4 produced by condensation or coagulation
  // transfer all of it to accum mode
  // also transfer the condensation and coagulation changes
  // to accum mode (for mass budget)
  for (int a = 0; a < num_cond_coag_to_accum; ++a) {
    const int ispec = indx_aer_cond_coag_to_accum[a];

    // Pack mode information per aerosol
    for (int imode = 0; imode < AeroConfig::num_modes(); imode++) {
      qaer_cur_modes[imode] = qaer_cur[ispec][imode];
      qaer_del_cond_modes[imode] = qaer_del_cond[ispec][imode];
      qaer_del_coag_modes[imode] = qaer_del_coag[ispec][imode];
    }

    transfer_cond_coag_mass_to_accum(nsrc, ndest, qaer_cur_modes,
                                     qaer_del_cond_modes, qaer_del_coag_modes);

    // Unpack mode information per aerosol
    for (int imode = 0; imode < AeroConfig::num_modes(); imode++) {
      qaer_cur[ispec][imode] = qaer_cur_modes[imode];
      qaer_del_cond[ispec][imode] = qaer_del_cond_modes[imode];
      qaer_del_coag[ispec][imode] = qaer_del_coag_modes[imode];
    }
  }

  // number - transfer the aged fraction to accum mode
  // include this transfer change in the cond and/or coag change (for mass
  // budget)
  transfer_aged_pcarbon_to_accum(nsrc, ndest, xferfrac_pcage, frac_cond,
                                 frac_coag, qnum_cur, qnum_del_cond,
                                 qnum_del_coag);
}

KOKKOS_INLINE_FUNCTION
void aerosol_aging_rates_1box(const int k, const AeroConfig &aero_config,
                              const Real dt, const Atmosphere &atm,
                              const Prognostics &progs,
                              const Diagnostics &diags, const Tendencies &tends,
                              const Aging::Config &config) {

  const int num_aer = AeroConfig::num_aerosol_ids();
  const int num_mode = AeroConfig::num_modes();

  // These are variables that I don't know how to define at the moment for now
  // we will just allocate them here
  Real dgn_a[num_mode];
  // Real qnum_cur[AeroConfig::num_modes()],
  Real qnum_del_cond[num_mode];
  Real qnum_del_coag[num_aer];
  //  Real qaer_cur[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
  Real qaer_del_cond[num_aer][num_mode];
  Real qaer_del_coag[num_aer][num_mode];
  Real qaer_del_coag_in[num_aer][AeroConfig::max_agepair()];

  // Get prognostic fields
  // Aerosol mass
  Real qaer_cur[num_aer][num_mode];
  for (int imode = 0; imode < num_mode; ++imode)
    for (int ispec = 0; ispec < num_aer; ++ispec)
      qaer_cur[ispec][imode] = progs.q_aero_i[imode][ispec](k);

  // Aerosol number
  Real qnum_cur[num_mode];
  for (int imode = 0; imode < num_mode; ++imode) {
    qnum_cur[imode] = progs.n_mode_i[imode](k);
  }

  // primary carbon aging
  mam_pcarbon_aging_1subarea(dgn_a, qnum_cur, qnum_del_cond, qnum_del_coag,
                             qaer_cur, qaer_del_cond, qaer_del_coag,
                             qaer_del_coag_in);

  // compute the tendencies
  for (int imode = 0; imode < num_mode; ++imode) {
    for (int ispec = 0; ispec < num_aer; ++ispec) {
      tends.q_aero_i[imode][ispec](k) +=
          (qaer_cur[ispec][imode] - progs.q_aero_i[imode][ispec](k)) / dt;
    }
  }

  for (int imode = 0; imode < num_mode; ++imode) {
    tends.n_mode_i[imode](k) +=
        (qnum_cur[imode] - progs.n_mode_i[imode](k)) / dt;
  }

  // Update the prognostics
  for (int imode = 0; imode < num_mode; ++imode) {
    for (int ispec = 0; ispec < num_aer; ++ispec) {
      progs.q_aero_i[imode][ispec](k) = qaer_cur[ispec][imode];
    }

    for (int imode = 0; imode < num_mode; ++imode) {
      progs.n_mode_i[imode](k) = qnum_cur[imode];
    }
  }
}

} // namespace aging

// init -- initializes the implementation with MAM4's configuration
inline void Aging::init(const AeroConfig &aero_config,
                        const Config &process_config) {

  config_ = process_config;
};

// compute_tendencies -- computes tendencies and updates diagnostics
// NOTE: that both diags and tends are const below--this means their views
// NOTE: are fixed, but the data in those views is allowed to vary.
KOKKOS_INLINE_FUNCTION
void Aging::compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                               Real t, Real dt, const Atmosphere &atm,
                               const Prognostics &progs,
                               const Diagnostics &diags,
                               const Tendencies &tends) const {

  const int nk = atm.num_levels();

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, nk), KOKKOS_CLASS_LAMBDA(int k) {
        aging::aerosol_aging_rates_1box(k, config, dt, atm, progs, diags, tends,
                                        config_);
      });
}

} // namespace mam4
#endif
