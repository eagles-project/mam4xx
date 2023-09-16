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

} // namespace modal_aer_opt

} // end namespace mam4

#endif	