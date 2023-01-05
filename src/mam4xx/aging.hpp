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

namespace aging{
KOKKOS_INLINE_FUNCTION
void mam_pcarbon_aging_frac(int nsrc, int ipar, Real *dgn_a, 
Real *qaer_cur, Real *qaer_del_cond, Real *qaer_del_coag_in, 
Real *xferfrac_pcage, Real *frac_cond, Real *frac_coag){

const int iaer_so4 = aerosol_index_for_mode(ModeIndex::Aitken, AeroId::SO4);
const int iaer_soa = aerosol_index_for_mode(ModeIndex::Aitken, AeroId::SOA); 

// for default MAM4 only so4 and soa contribute to aging, nsoa is for tagging and
// is set to 1 for default MAM4

const Real vol_shell = qaer_cur[iaer_so4][nsrc]*mass_2_vol[iaer_so4] + qaer_cur[iaer_soa][nsrc]*fac_m2v_eqvhyg_aer[iaer_soa];

(void) iaer_soa; 
(void) vol_shell;

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
Real xferfrac_pcage, Real frac_cond, Real frac_coag, 
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