#ifndef MAM4XX_MODALAER_OPT_HPP
#define MAM4XX_MODALAER_OPT_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>

namespace mam4 {
namespace modal_aer_opt {

// FIXME; update value
constexpr int ncoef=3;
// ! min, max aerosol surface mode radius treated [m]
constexpr Real rmmin = 0.01e-6;
constexpr Real rmmax = 25.e-6;
const Real xrmin = haero::log(rmmin);
const Real xrmax = haero::log(rmmax);
using View2D = DeviceType::view_2d<Real>;
// From radconstants
constexpr int nswbands = 14; 

KOKKOS_INLINE_FUNCTION
void modal_size_parameters(const Real sigma_logr_aer,
                           const Real dgnumwet,// in
                           Real &radsurf, Real &logradsurf,
                           Real cheb[ncoef],      
                           const bool ismethod2 )
{

   // ncol
   // sigma_logr_aer  ! geometric standard deviation of number distribution
   // dgnumwet(:,:)   ! aerosol wet number mode diameter [m]
   // radsurf(:,:)    ! aerosol surface mode radius [m]
   // logradsurf(:,:) ! log(aerosol surface mode radius)
   // cheb(:,:,:)     ! chebychev polynomial parameters

   // FORTRAN refactoring: ismethod is tempararily used to ensure BFB test
   /// bool ismethod2

   // integer  :: icol, kk, nc
   // real(r8) :: alnsg_amode      ! log(sigma)
   // real(r8) :: explnsigma
   // real(r8) :: xrad ! normalized aerosol radius
   // !-------------------------------------------------------------------------------
   constexpr Real half = 0.5;
   constexpr Real one = 1.0;
   constexpr Real two = 2.0;
  
   const Real alnsg_amode = haero::log(sigma_logr_aer);
   const Real explnsigma = haero::exp(two*alnsg_amode*alnsg_amode);
   // do kk = top_lev, pver
   // do icol = 1, ncol
   // ! convert from number mode diameter to surface area
    radsurf = half*dgnumwet*explnsigma;
    // ! --------------- FORTRAN refactoring -------------------
    // ! here two calculations are used to ensure passing BFB test
    // ! can be simplified (there is only round-off difference
    if (ismethod2)
    {
    	logradsurf = haero::log(half*dgnumwet) + two*alnsg_amode*alnsg_amode;
    } else {
    	logradsurf = haero::log(radsurf);

    } // ismethod2
     // ! --------------- FORTRAN refactoring -------------------

         // ! normalize size parameter
     Real xrad = mam4::utils::min_max_bound(xrmin, xrmax, logradsurf);
     xrad = (two*xrad-xrmax-xrmin)/(xrmax-xrmin);
     // chebyshev polynomials
     cheb[0] = one;
     cheb[1] = xrad;
     for (int nc = 2; nc < ncoef; ++nc)
     {
     	cheb[nc] = two*xrad*cheb[nc-1] - cheb[nc-2];
     }// nc

}// modal_size_parameters

KOKKOS_INLINE_FUNCTION
void calc_parameterized(const Real coef[ncoef],
                        const Real cheb_k[ncoef], 
                        Real & para)
{
 // calculate parameterized absorption, extinction or asymmetry factor
 // further calculations are needed. see modal_aero_sw and modal_aero_lw
 // ncol,ncoef
 // coef(pcols,ncoef)
  // cheb_k(ncoef,pcols)
  // para(pcols)
	constexpr Real half = 0.5;
	para = half*coef[0];
	for (int nc = 1; nc < ncoef; ++nc)
	{
	  para += cheb_k[nc]*coef[nc];
	}

}// calc_parameterized

KOKKOS_INLINE_FUNCTION
void update_aod_spec(const Real scath2o,
                     const Real absh2o, // in
                     const Real sumhygro, 
                     const Real sumscat,
                     const Real sumabs, // in
                     const Real hygro_s, 
                     const Real palb,
                     const Real dopaer,     // in
                     Real& scat_s,
                     Real& abs_s, 
                     Real& aod_s      )
{
	// scath2o, absh2o, sumscat, sumabs, sumhygro
// hygro_s, palb, dopaer	
  // scat_s, abs_s, aod_s  ! scatering, absorption and aod for a species	
  // update aerosol optical depth from scattering and absorption	
   constexpr Real one =1.0;	
   scat_s     = (scat_s + scath2o*hygro_s/sumhygro)/sumscat;
   abs_s      = (abs_s + absh2o*hygro_s/sumhygro)/sumabs;
   aod_s      += (abs_s*(one - palb) + palb*scat_s)*dopaer;

}// update_aod_spec

#if 0
KOKKOS_INLINE_FUNCTION
void calc_volc_ext()
{
 // calculate contributions from volcanic aerosol extinction
 // trop_level(pcols)!tropopause level for each column
 // state_zm(:,:) ! state%zm [m]
 // ext_cmip6_sw(pcols,pver) ! aerosol shortwave extinction [1/m]
 // extinct(pcols,pver) ! aerosol extinction [1/m]
 // tropopause_m(pcols) ! tropopause height [m]
	// kk_tropp = trop_level(icol)
	//

 // diagnose tropopause height
  tropopause_m = state_zm_kk_tropp;//!in meters
  //update tropopause layer first
   extinct_kk_tropp = half*( extinct_kk_tropp + ext_cmip6_sw_,kk_tropp)
  //extinction is assigned read in values only for visible band above tropopause
   // extinct(icol, 1:kk_tropp-1) = ext_cmip6_sw(icol, 1:kk_tropp-1)


} // calc_volc_ext
#endif

KOKKOS_INLINE_FUNCTION
void volcanic_cmip_sw(const ColumnView& zi,
                      const int ilev_tropp,
                      const View2D& ext_cmip6_sw_inv_m,
                      const View2D& ssa_cmip6_sw,
                      const View2D& af_cmip6_sw, 
                      const View2D& tau,
                      const View2D& tau_w,
                      const View2D& tau_w_g,
                      const View2D& tau_w_f)
{

	// !Intent-in
    // ncol       ! Number of columns
    // zi(:,:)    ! Height above surface at interfaces [m]
    // trop_level(pcols)  ! tropopause level index
    // ext_cmip6_sw_inv_m(pcols,pver,nswbands)  ! short wave extinction [m^{-1}]
    // ssa_cmip6_sw(:,:,:),af_cmip6_sw(:,:,:)

    // !Intent-inout
    // tau    (pcols,0:pver,nswbands) ! aerosol extinction optical depth
    // tau_w  (pcols,0:pver,nswbands) ! aerosol single scattering albedo * tau
    // tau_w_g(pcols,0:pver,nswbands) ! aerosol assymetry parameter * tau * w
    // tau_w_f(pcols,0:pver,nswbands) ! aerosol forward scattered fraction * tau * w

    // !Local variables
    // icol, ipver, ilev_tropp
    // lyr_thk ! thickness between level interfaces [m]
    // ext_unitless(nswbands), asym_unitless(nswbands)
    // ext_ssa(nswbands),ext_ssa_asym(nswbands)

    // !Logic:
    // !Update taus, tau_w, tau_w_g and tau_w_f with the read in volcanic
    // !aerosol extinction (1/km), single scattering albedo and asymmtry factors.

    // !Above the tropopause, the read in values from the file include both the stratospheric
    // !and volcanic aerosols. Therefore, we need to zero out taus above the tropopause
    // !and populate them exclusively from the read in values.

    // !If tropopause is found, update taus with 50% contributuions from the volcanic input
    // !file and 50% from the existing model computed values

    // !First handle the case of tropopause layer itself:
    // do icol = 1, ncol
       // ilev_tropp = trop_level(icol) !tropopause level
	// 
	constexpr Real half =0.5;

    const Real lyr_thk = zi(ilev_tropp) - zi(ilev_tropp+1);
       for (int i = 0; i < nswbands; ++i)
       {
       	const Real ext_unitless  = lyr_thk * ext_cmip6_sw_inv_m(ilev_tropp,i);
       	const Real asym_unitless = af_cmip6_sw(ilev_tropp,i);
       	const Real ext_ssa     = ext_unitless * ssa_cmip6_sw(ilev_tropp,i);
       	const Real ext_ssa_asym  = ext_ssa * asym_unitless;

       	tau(ilev_tropp,i) = half * ( tau(ilev_tropp,i) + ext_unitless);
        tau_w(ilev_tropp,i) = half * ( tau_w(ilev_tropp,i) + ext_ssa);
        tau_w_g(ilev_tropp,i) = half * ( tau_w_g(ilev_tropp,i) + ext_ssa_asym);
        tau_w_f(ilev_tropp,i) = half * ( tau_w_f(ilev_tropp,i) + ext_ssa_asym * asym_unitless);
       } // end i 

       // !As it will be more efficient for FORTRAN to loop over levels and then columns, the following loops
       // !are nested keeping that in mind
       // Note that in C++ ported code, the loop over levels is nested. Thus, the previous comment does not apply. 
       
       // ilev_tropp = trop_level(icol) !tropopause level
       for (int kk = 0; kk < ilev_tropp; ++kk)
       {
       	
        const Real lyr_thk = zi(kk) - zi(kk+1);
        for (int i = 0; i < nswbands; ++i)
       {
       	const Real ext_unitless  = lyr_thk * ext_cmip6_sw_inv_m(kk,i);
       	const Real asym_unitless = af_cmip6_sw(kk,i);
        const Real ext_ssa       = ext_unitless * ssa_cmip6_sw(kk,i);
        const Real ext_ssa_asym  = ext_ssa * asym_unitless;
        tau(kk,i) = ext_unitless;
        tau_w(kk,i) = ext_ssa;
        tau_w_g(kk,i) = ext_ssa_asym;
        tau_w_f(kk,i) = ext_ssa_asym * asym_unitless;

       } // end nswbands

       } // kk 

}//volcanic_cmip_sw



} // namespace modal_aer_opt

} // end namespace mam4

#endif	