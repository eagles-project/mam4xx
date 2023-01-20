#ifndef MAM4XX_AGING_HPP
#define MAM4XX_AGING_HPP
#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/merikanto2007.hpp>
#include <mam4xx/vehkamaki2002.hpp>
#include <mam4xx/wang2008.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

namespace mam4 {

class Aging {

public:
  static const int num_pcarbon_to_accum = 3;
  static const int num_cond_coag_to_accum = 2;

  // These are the aerosol indicies Primary Carbon mode aerosols that need
  // to be transferred to the accumulation mode.
  static constexpr int indx_aer_pcarbon_to_accum[num_pcarbon_to_accum] = {
      static_cast<int>(AeroId::POM), static_cast<int>(AeroId::BC),
      static_cast<int>(AeroId::MOM)};

  // These are the aerosol indicies of the Primary
  static constexpr int indx_aer_cond_coag_to_accum[num_cond_coag_to_accum] = {
      static_cast<int>(AeroId::SOA), static_cast<int>(AeroId::SO4)};

  Real aero_vol[AeroConfig::num_aerosol_ids()];

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

  // In E3SM this is read in from an input file
  static constexpr Real n_so4_monolayers_pcage = 8.0;
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

KOKKOS_INLINE_FUNCTION
void mam_pcarbon_aging_frac(
    const int nsrc, Real dgn_a[AeroConfig::num_modes()],
    Real qaer_cur[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer_del_cond[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer_del_coag_in[AeroConfig::num_aerosol_ids()]
                         [AeroConfig::num_modes()],
    Real &xferfrac_pcage, Real &frac_cond, Real &frac_coag) {

  const int ipair = 1;
  const int iaer_so4 =
      aerosol_index_for_mode(ModeIndex::PrimaryCarbon, AeroId::SO4);
  const int iaer_soa =
      aerosol_index_for_mode(ModeIndex::PrimaryCarbon, AeroId::SOA);
  const int iaer_bc =
      aerosol_index_for_mode(ModeIndex::PrimaryCarbon, AeroId::BC);
  const int iaer_pom =
      aerosol_index_for_mode(ModeIndex::PrimaryCarbon, AeroId::POM);
  const int iaer_mom =
      aerosol_index_for_mode(ModeIndex::PrimaryCarbon, AeroId::MOM);

  // Compute the aerosol volume per mole
  const Real so4_vol =
      aero_species(iaer_so4).molecular_weight / aero_species(iaer_so4).density;
  const Real soa_vol =
      aero_species(iaer_soa).molecular_weight / aero_species(iaer_soa).density;
  const Real bc_vol =
      aero_species(iaer_bc).molecular_weight / aero_species(iaer_bc).density;
  const Real pom_vol =
      aero_species(iaer_pom).molecular_weight / aero_species(iaer_pom).density;
  const Real mom_vol =
      aero_species(iaer_mom).molecular_weight / aero_species(iaer_mom).density;

  const Real fac_m2v_eqvhyg_aer =
      soa_vol * aero_species(iaer_soa).hygroscopicity;

  const Real vol_shell = qaer_cur[iaer_so4][nsrc] * so4_vol +
                         qaer_cur[iaer_soa][nsrc] * fac_m2v_eqvhyg_aer;

  const Real qaer_del_cond_tmp =
      haero::max(qaer_del_cond[iaer_so4][nsrc] * so4_vol +
                     qaer_del_cond[iaer_soa][nsrc] * fac_m2v_eqvhyg_aer,
                 std::numeric_limits<float>::epsilon());
  const Real qaer_del_coag_tmp =
      qaer_del_coag_in[iaer_so4][ipair] * so4_vol +
      qaer_del_coag_in[iaer_soa][ipair] * fac_m2v_eqvhyg_aer;

  frac_cond = 1.0 - qaer_del_cond_tmp / (qaer_del_cond_tmp +
                                         haero::max(qaer_del_coag_tmp, 0.0));
  frac_coag = 1.0 - frac_cond;

  const int core_modes[3] = {iaer_bc, iaer_pom, iaer_mom};
  const Real core_volumes[3] = {bc_vol, pom_vol, mom_vol};
  Real vol_core = 0.0;
  for (int mi = 0; mi < 3; ++mi) {
    const int m = core_modes[mi];
    vol_core += qaer_cur[m][nsrc] * core_volumes[m];
  }

  const Real fac_volsfc = haero::exp(
      2.5 * haero::log(haero::pow(mam4::modes(nsrc).mean_std_dev, 2.0)));
  const Real xferfrac_max =
      1.0 - 10.0 * std::numeric_limits<float>::epsilon(); //  1-eps

  Real xferfrac_tmp1 = vol_shell * dgn_a[nsrc] * fac_volsfc;
  Real xferfrac_tmp2 =
      haero::max(6.0 * Aging::dr_so4_monolayers_pcage * vol_core, 0.0);

  if (xferfrac_tmp1 >= xferfrac_tmp2) {
    xferfrac_pcage = xferfrac_max;
  } else {
    xferfrac_pcage = haero::min(xferfrac_tmp1 / xferfrac_tmp2, xferfrac_max);
  }
}

KOKKOS_INLINE_FUNCTION
void transfer_aged_pcarbon_to_accum(
    const int nsrc, const int ndest, const Real xferfrac_pcage,
    const Real frac_cond, const Real frac_coag,
    Real qaer_cur[AeroConfig::num_modes()],
    Real qaer_del_cond[AeroConfig::num_modes()],
    Real qaer_del_coag[AeroConfig::num_modes()]) {

  Real q_tmp = qaer_cur[nsrc] * xferfrac_pcage;

  qaer_cur[nsrc] -= q_tmp;
  qaer_cur[ndest] += q_tmp;

  qaer_del_cond[nsrc] -= q_tmp * frac_cond;
  qaer_del_cond[ndest] += q_tmp * frac_cond;

  qaer_del_coag[nsrc] -= q_tmp * frac_coag;
  qaer_del_coag[ndest] += q_tmp * frac_coag;
}

KOKKOS_INLINE_FUNCTION
void transfer_cond_coag_mass_to_accum(
    const int nsrc, const int ndest, Real qaer_cur[AeroConfig::num_modes()],
    Real qaer_del_cond[AeroConfig::num_modes()],
    Real qaer_del_coag[AeroConfig::num_modes()]) {

  qaer_cur[ndest] += qaer_cur[nsrc];
  qaer_del_cond[ndest] += qaer_del_cond[nsrc];
  qaer_del_coag[ndest] += qaer_del_coag[nsrc];

  qaer_cur[nsrc] = 0.0;
  qaer_del_cond[nsrc] = 0.0;
  qaer_del_coag[nsrc] = 0.0;
}

KOKKOS_INLINE_FUNCTION
void mam_pcarbon_aging_1subarea(
    Real dgn_a[AeroConfig::num_modes()], Real qnum_cur[AeroConfig::num_modes()],
    Real qnum_del_cond[AeroConfig::num_modes()],
    Real qnum_del_coag[AeroConfig::num_aerosol_ids()],
    Real qaer_cur[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer_del_cond[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer_del_coag[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real qaer_del_coag_in[AeroConfig::num_aerosol_ids()]
                         [AeroConfig::num_modes()]) {

  Real xferfrac_pcage, frac_cond, frac_coag;

  const int nsrc = static_cast<int>(ModeIndex::PrimaryCarbon);
  const int ndest = static_cast<int>(ModeIndex::Accumulation);

  mam_pcarbon_aging_frac(nsrc, dgn_a, qaer_cur, qaer_del_cond, qaer_del_coag_in,
                         xferfrac_pcage, frac_cond, frac_coag);
  // Note, there are probably optimizations to be done here, closely following
  // the Fortran code required extra unpacking of arrays.

  // MAM4 pcarbon mode only has pom, bc, mom, lmap only has index (>0) for these
  // species species is pom or bc transfer the aged fraction to accum mode
  //  include this transfer change in the cond and/or coag change (for mass
  //  budget)

  Real qaer_cur_modes[AeroConfig::num_modes()];
  Real qaer_del_cond_modes[AeroConfig::num_modes()];
  Real qaer_del_coag_modes[AeroConfig::num_modes()];

  for (int a = 0; a < Aging::num_pcarbon_to_accum; ++a) {

    const int ai = Aging::indx_aer_pcarbon_to_accum[a];

    // Pack mode information per aerosol
    for (int m = 0; m < AeroConfig::num_modes(); m++) {
      qaer_cur_modes[m] = qaer_cur[ai][m];
      qaer_del_cond_modes[m] = qaer_del_cond[ai][m];
      qaer_del_coag_modes[m] = qaer_del_coag[ai][m];
    }

    transfer_aged_pcarbon_to_accum(nsrc, ndest, xferfrac_pcage, frac_cond,
                                   frac_coag, qaer_cur_modes,
                                   qaer_del_cond_modes, qaer_del_coag_modes);

    // Unpack mode information per aerosol
    for (int m = 0; m < AeroConfig::num_modes(); m++) {
      qaer_cur[ai][m] = qaer_cur_modes[m];
      qaer_del_cond[ai][m] = qaer_del_cond_modes[m];
      qaer_del_coag[ai][m] = qaer_del_coag_modes[m];
    }
  }

  // species is soa, so4, or nh4 produced by condensation or coagulation
  // transfer all of it to accum mode
  // also transfer the condensation and coagulation changes
  // to accum mode (for mass budget)
  for (int a = 0; a < Aging::num_cond_coag_to_accum; ++a) {
    const int ai = Aging::indx_aer_cond_coag_to_accum[a];

    // Pack mode information per aerosol
    for (int m = 0; m < AeroConfig::num_modes(); m++) {
      qaer_cur_modes[m] = qaer_cur[ai][m];
      qaer_del_cond_modes[m] = qaer_del_cond[ai][m];
      qaer_del_coag_modes[m] = qaer_del_coag[ai][m];
    }

    transfer_cond_coag_mass_to_accum(nsrc, ndest, qaer_cur_modes,
                                     qaer_del_cond_modes, qaer_del_coag_modes);

    // Unpack mode information per aerosol
    for (int m = 0; m < AeroConfig::num_modes(); m++) {
      qaer_cur[ai][m] = qaer_cur_modes[m];
      qaer_del_cond[ai][m] = qaer_del_cond_modes[m];
      qaer_del_coag[ai][m] = qaer_del_coag_modes[m];
    }

    // number - transfer the aged fraction to accum mode
    // include this transfer change in the cond and/or coag change (for mass
    // budget)
  }

  transfer_cond_coag_mass_to_accum(nsrc, ndest, qnum_cur, qnum_del_cond,
                                   qnum_del_coag);
}

KOKKOS_INLINE_FUNCTION
void aerosol_aging_rates_1box(const int k, const AeroConfig &aero_config,
                              const Real dt, const Atmosphere &atm,
                              const Prognostics &progs,
                              const Diagnostics &diags,
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
  Real qaer_del_coag_in[num_aer][num_mode];

  // Get prognostic fields
  // Aerosol mass
  Real qaer_cur[num_aer][num_mode];
  for (int n = 0; n < num_mode; ++n)
    for (int g = 0; g < num_aer; ++g)
      qaer_cur[g][n] = progs.q_aero_i[n][g](k);

  // Aerosol number
  Real qnum_cur[num_mode];
  for (int i = 0; i < num_mode; ++i) {
    qnum_cur[i] = progs.n_mode_i[i](k);
  }

  mam_pcarbon_aging_1subarea(dgn_a, qnum_cur, qnum_del_cond, qnum_del_coag,
                             qaer_cur, qaer_del_cond, qaer_del_coag,
                             qaer_del_coag_in);
}

} // namespace aging

// init -- initializes the implementation with MAM4's configuration
inline void Aging::init(const AeroConfig &aero_config,
                        const Config &process_config) {

  config_ = process_config;

  for (int ai = 0; ai < AeroConfig::num_aerosol_ids(); ++ai) {
    const AeroSpecies as = aero_species(ai);

    aero_vol[ai] = as.molecular_weight / as.density;
  }
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
        aging::aerosol_aging_rates_1box(k, config, dt, atm, progs, diags,
                                        config_);
      });
}

} // namespace mam4
#endif