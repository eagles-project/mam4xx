#ifndef MAM4XX_AGING_HPP
#define MAM4XX_AGING_HPP
#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/merikanto2007.hpp>
#include <mam4xx/vehkamaki2002.hpp>
#include <mam4xx/wang2008.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>


namespace mam4{

class Aging{
public:
 struct Config{

    Config() {}; 

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
 };

private:
  static constexpr int num_mode = AeroConfig::num_modes();
  static constexpr int num_gas = AeroConfig::num_gas_ids();
  static constexpr int num_aer = AeroConfig::num_aerosol_ids();
  static constexpr int nait = static_cast<int>(ModeIndex::Aitken);
  static constexpr int npca = static_cast<int>(ModeIndex::PrimaryCarbon);
  static constexpr int iaer_so4 = static_cast<int>(AeroId::SO4);
  static constexpr int iaer_soag_bgn = static_cast<int>(AeroId::SOA);
  static constexpr int iaer_soag_end = static_cast<int>(AeroId::SOA);

  Config config_;


public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 aging"; }


  void init(const AeroConfig &aero_config,
            const Config &aging_config = Config()) {

        config_ = aging_config;

    };


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

namespace aging{

KOKKOS_INLINE_FUNCTION
void mam_pcarbon_aging_1subarea( int n_mode, 
                                Real *dgn_a[AeroConfig::num_modes()],
                                Real *qnum_cur[AeroConfig::num_modes()],
                                Real *qnum_del_cond[AeroConfig::num_modes()],
                                Real *qnum_del_coag[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
                                Real *qaer_cur[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
                                Real *qaer_del_cond[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
                                Real *qaer_del_coag_in[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()]
){



}


KOKKOS_INLINE_FUNCTION
void mam_pcarbon_aging_frac(
const int mode_pca,    // Called nsrc in original F90 code. 
Real *dgn_a, 
Real *qaer_cur[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],       // aerosol molar mixing ratio   [kmol/kmol-air]
Real *qaer_del_cond[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],  // 
Real *qaer_del_coag_in, 
Real *xferfrac_pcage,                                                         // fraction of aged pom/bc transferred to accum [unitless]
Real *frac_cond,                                                              // fraction of aerosol change due to condensation [unitless]
Real *frac_coag)                                                              // fraction of aerosol change due to condensation [unitless]
{                  




const int iaer_so4 = aerosol_index_for_mode(ModeIndex::Aitken, AeroId::SO4);
const int iaer_soa = aerosol_index_for_mode(ModeIndex::Aitken, AeroId::SOA); 

const Real so4_vol = aero_species(iaer_so4).molecular_weight/aero_species(int(AeroId::SO4)).density;
const Real soa_vol =  aero_species(int(AeroId::SOA)).molecular_weight/aero_species(int(AeroId::SOA)).density;
const Real fac_m2v_eqvhyg_aer = soa_vol * aero_species(int(AeroId::SO4)).hygroscopicity; 


(void) iaer_so4; 
(void) iaer_soa; 
(void) so4_vol;
(void) soa_vol;
(void) fac_m2v_eqvhyg_aer;
//(void) vold_shell;
//(void) vol_shell;

// const Real qaer_del_cond_tmp = qaer_del_cond[iaer_so4][nsrc]*mass_2_vol[iaer_so4] + qaer_del_cond[iaer_soa][nsrc]*fac_m2v_eqvhyg_aer[iaer_soa];

// const Real qaer_del_coag_tmp = qaer_del_coag_in[iaer_so4][ipair]*mass_2_vol[iaer_so4] + qaer_del_coag_in[iaer_soa][ipair]*fac_m2v_eqvhyg_aer[iaer_soa];


}

KOKKOS_INLINE_FUNCTION
void transfer_aged_pcarbon_to_accum(int nsrc, int ndest, Real xferfrac_pcage, 
Real  frac_cond, Real frac_coag, Real *q_cur, Real *q_del_cond, Real *q_del_coag){

const Real q_tmp = q_cur[nsrc]*xferfrac_pcage; 

q_cur[nsrc] -=  q_tmp;
q_cur[ndest] +=  q_tmp;

q_del_cond[nsrc] -= q_tmp*frac_cond; 
q_del_cond[ndest] += q_tmp*frac_cond;

q_del_coag[nsrc] -= q_tmp*frac_coag;
q_del_coag[nsrc] += q_tmp*frac_coag;

}

KOKKOS_INLINE_FUNCTION
void transfer_cond_coag_mass_to_accum(int nsrc, int ndest,
Real *qaer_cur, Real *qaer_del_cond, Real *qaer_del_coag){


qaer_cur[ndest] += qaer_cur[nsrc];
qaer_del_cond[ndest] += qaer_del_cond[nsrc];
qaer_del_coag[ndest] += qaer_del_cond[nsrc];

qaer_cur[nsrc] = 0.0;
qaer_del_cond[nsrc] = 0.0;
qaer_del_coag[nsrc] = 0.0;

}

}
}
#endif