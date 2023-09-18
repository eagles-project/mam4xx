#ifndef MAM4XX_MODALAER_OPT_HPP
#define MAM4XX_MODALAER_OPT_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>
#include <Kokkos_Complex.hpp>

namespace mam4 {
namespace modal_aer_opt {


// ! min, max aerosol surface mode radius treated [m]
constexpr Real rmmin = 0.01e-6;
constexpr Real rmmax = 25.e-6;
const Real xrmin = haero::log(rmmin);
const Real xrmax = haero::log(rmmax);
using View2D = DeviceType::view_2d<Real>;
using ComplexView2D = DeviceType::view_2d<Kokkos::complex<Real>>;
// From radconstants
constexpr int nswbands = 14; 
constexpr int nlwbands = 16;

// Dimension sizes in coefficient arrays used to parameterize aerosol radiative properties
// in terms of refractive index and wet radius
constexpr int ncoef=5;
constexpr int prefr=7;
constexpr int prefi=10;

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
void calc_volc_ext(const int trop_level,
                   const ColumnView& state_zm,
                   const ColumnView& ext_cmip6_sw, 
                   const ColumnView& extinct,
                   Real &tropopause_m)
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
  tropopause_m = state_zm(trop_level);//!in meters
  //update tropopause layer first
  extinct(trop_level-1) = half*( extinct(trop_level-1) + ext_cmip6_sw(trop_level-1));
  //extinction is assigned read in values only for visible band above tropopause
  for (int kk = 0; kk < trop_level; ++kk)
  {
  	extinct(kk) = ext_cmip6_sw(kk)
  }
  

} // calc_volc_ext


KOKKOS_INLINE_FUNCTION
void calc_diag_spec(const Real specmmr_k,
                    const Real mass_k,
                    const Real vol, 
                    const Real specrefr, 
                    const Real specrefi, 
                    const Real hygro_aer,
                    Real& burden_s,
                    Real& scat_s,
                    Real& abs_s,
                    Real& hygro_s)
{
   // calculate some diagnostics for a species
   // specmmr_k(:)   mmr at level kk [kg/kg]
   // mass_k(:)         mass at layer kk [kg/m2]
   // vol(:)  volume concentration of aerosol species [m3/kg]
   // specrefr, specrefi   real and image part of specrefindex
   // hygro_aer        aerosol hygroscopicity [unitless]
   // burden_s(pcols)  aerosol burden of species [kg/m2]
   // scat_s(pcols)    scattering of species [unitless]
   // abs_s(pcols)     absorption of species [unit?]
   // hygro_s(pcols)   hygroscopicity of species [unitless]
	constexpr Real zero=0;

	burden_s = zero;
	burden_s += specmmr_k*mass_k;
    scat_s   = vol*specrefr;
    abs_s    = - vol*specrefi;
    hygro_s  = vol*hygro_aer;

}// calc_diag_spec


KOKKOS_INLINE_FUNCTION
void calc_refin_complex(const int lwsw, 
	                    const int ilwsw,
                        const Real qaerwat_kk, 
                        const Real *specvol,
                        const ComplexView2D& specrefindex, 
                        Real &dryvol, Real &wetvol, Real &watervol,
                        Kokkos::complex<Real> & crefin, Real&refr, Real&refi)
{
	/*-------------------------------------------------------------------
     calculate complex refractive index 
     also output wetvol and watervol
    -------------------------------------------------------------------*/

    
    // lwsw   ! indicator if this is lw or sw lw =0 and sw =1
    // ncol, ilwsw       
    // qaerwat_kk(:)   ! aerosol water at level kk [g/g]
    //specvol(:,:)    ! volume concentration of aerosol specie [m3/kg]
    // complex(r8), intent(in) :: specrefindex(:,:)     ! species refractive index

    // dryvol(pcols)    ! volume concentration of aerosol mode [m3/kg]
    // wetvol(pcols)    ! volume concentration of wet mode [m3/kg]
    // watervol(pcols)  ! volume concentration of water in each mode [m3/kg]
    // refr(pcols)      ! real part of refractive index
    // refi(pcols)      ! imaginary part of refractive index
    // complex(r8),intent(out) :: crefin(pcols) ! complex refractive index

    // FIXME 
    // if ((lwsw /= 'lw') .and. (lwsw /= 'sw')) then
    //     call endrun('calc_refin_complex is called with '// lwsw// ', it should be called with either lw or sw')
    // endif

    // crefin(:ncol) = (0._r8, 0._r8)
    const Real zero=0;
    dryvol = zero;

    crefin={};
    nspec

    for (int i = 0; i < nspec; ++i)
    {
     dryvol += specvol[i];
     crefin += specvol[i]*specrefindex(i,ilwsw);
    }

    watervol = qaerwat_kk/rhoh2o;
    wetvol   = watervol + dryvol;
    if (watervol < zero && lwsw==0 ) // lwsw=='lw'
    {
    	// BAD CONSTANT 
    	// FIXME 
    	// if (haero::abs(watervol > 1.e-1*wetvol))
    	// {
    	// 	write(iulog,*) 'watervol,wetvol,dryvol=',watervol(icol), &
        //                  wetvol(icol),dryvol(icol)
    	// }
    	watervol= zero;
        wetvol = dryvol; 
    } // end if watervol < zero && lwsw=='lw'

    // ! some different treatments for lw and sw
    if (lwsw==0) //lwsw=='lw'
    {
      crefin += watervol*crefwlw(ilwsw);
      // BAD CONSTANT 
      if (wetvol > small_value_40) 
      {
       crefin /= wetvol(icol);
      } // end if wetvol(icol) > small_value_40

    } else if ( lwsw==1) //  lwsw=='sw
    {

    	crefin +=  watervol*crefwsw(ilwsw);
    	// BAD CONTANT
         crefin /= haero::max(wetvol, small_value_60);

    }  // lwsw=='lw'
    //FIXME
    refr = crefin.real();
    refi = crefin.imag();



}// calc_refin_complex

KOKKOS_INLINE_FUNCTION
void compute_factors(const int prefri,
	                 const Real ref_ind, const Real* ref_table, 
                     int& ix, Real &tt)
{
  // Compute factors for the real or imaginary parts

  // prefri, ncol
  // ref_table(:) !refractive index table [unitless]
  // ref_ind(:)   !refractive index       [unitless]

  //FORTRAN refactor note: "do_print" is kept to maintain code consistenty with the
  //original code. THis can be removed from the C++ ported code
  // logical, intent(in), optional :: do_print ! to print log msg or not

  // !intent-inouts/outs
  // ix(:)
  // tt(:)
  constexpr Real threshold = 1.e-20;
  constexpr zero =0;
  ix = 1
  tt = zero;
  if(prefri > 1) {

  	for (int ii = 0; ii < prefri; ++ii)
  	{
  		if(ref_ind < ref_table[ii]) 
  		{break;}
  	} // ii
  	// FIXME: check Fortran to C++ indexing conversion  
  	ix  = haero::max(ii-1,0);
    const int ip1 = haero::min(ix+1,prefri-1);
    const Real dx  = ref_table[ip1]-ref_table[ix];
    if (haero::abs(dx) > threshold){
    	tt = (ref_ind-ref_table[ix])/dx;
        // FIXME: I did not port the following:
    	// if (present(do_print) .and. do_print) then
        //       if(tt(ic) < 0._r8 .or. tt(ic) > 1._r8) then
        //          write(iulog,*) 'tt,ref_ind,ix,ref_table,dx=',tt(ic),ref_ind(ic),ix(ic),ref_table(ix(ic)),dx
        //       endif
        //    endif

    } // (abs(dx) > threshold)

  } // if(prefri > 1)

}//compute_factors

KOKKOS_INLINE_FUNCTION
void binterp(const View3D& table,
	         const Real ref_real,
	         const Real ref_img, 
	         const Real ref_real_tab[prefr],
	         const Real ref_img_tab[prefi],
             int & itab, int & jtab,
             Real & ttab, Real & utab,
             Real coef[ncoef],
             const int itab_1)
{
  /*------------------------------------------------------------------------------
   Bilinear interpolation along the refractive index dimensions
   of the table to estimate Chebyshev coefficients at an
   intermediate refractive index.

   In short wave, the first call computes itab, jtab, ttab, utab and coef.
   The subsequent calls use itab, jtab, ttab and utab as inputs and compute coef

   In long wave, we have just one call to compute itab,jtab,ttab, utab and coef
  ------------------------------------------------------------------------------*/

  // intent-ins
  // ncol
  // table(ncoef,prefr,prefi)
  // ref_real(pcols), ref_img(pcols) !real and imganinary parts of refractive indices [unitless]
  // ref_real_tab(prefr), ref_img_tab(prefi) !real and imganinary table refractive indices [unitless]

  // intent-inouts/outs
  // itab(pcols), jtab(pcols)
  // ttab(pcols), utab(pcols)
  // coef(pcols,ncoef) !coefficient interpolated bilinearly
  // FIXME; maybe we need to loop over cols	
  if (itab_1 < -1)
  {
  	// compute factors for the real part
  	compute_factors(prefr,
	                ref_real, ref_real_tab, 
                    itab, ttab); 

  	// !compute factors for the imaginary part
  	compute_factors(prefi,
	                ref_img, ref_img_tab, 
                    jtab, utab); 
  } // itab_1 < -1

  tu   = ttab*utab;
  tuc  = ttab-tu;
  tcuc = one-tuc-utab;
  tcu = utab-tu;
  // FIXME: check Fortran to C++ indexing 
  jp1 = haero::min(jtab+1,prefi);
  ip1 = haero::min(itab+1,prefr);
  for (int icoef = 0; icoef < ncoef; ++icoef)
  {
  	coef[icoef] = tcuc * table(icoef,itab,jtab) + 
                  tuc  * table(icoef,ip1,jtab)  + 
                  tu   * table(icoef,ip1,jp1)  + 
                  tcu  * table(icoef,itab,jp1);
  } // icoef


}// binterp

KOKKOS_INLINE_FUNCTION
void modal_aero_sw(dt, lchnk, ncol, state_q, state_zm, temperature, pmid, pdel, pdeldry, & ! in
                   cldn, nnite, idxnite, is_cmip6_volc, ext_cmip6_sw, trop_level,  & ! in
                         qqcw, tauxar, wa, ga, fa) 
{
   // ! calculates aerosol sw radiative properties
   // dt               !timestep [s]
   // lchnk            ! chunk id
   // ncol             ! number of active columns in the chunk
   //  state_q(:,:,:)   ! water and tracers (state%q) in state [kg/kg] 
   //  state_zm(:,:)    ! mid-point height (state%zm) [m]
   //  temperature(:,:) ! temperature [K]
   //  pmid(:,:)        ! mid-point pressure [Pa]
   //  pdel(:,:)        ! pressure interval [Pa]
   //  pdeldry(:,:)     ! dry mass pressure interval [Pa]

   //  cldn(:,:)         ! layer cloud fraction [fraction]

   //  nnite          ! number of night columns
   //  idxnite(nnite) ! local column indices of night columns
   //  trop_level(pcols)!tropopause level for each column
   //  ext_cmip6_sw(pcols,pver) ! aerosol shortwave extinction [1/m]
   //  is_cmip6_volc

   //  qqcw(:)               ! Cloud borne aerosols mixing ratios [kg/kg or 1/kg]
   //  tauxar(pcols,0:pver,nswbands) ! layer extinction optical depth [1]
   //  wa(pcols,0:pver,nswbands)     ! layer single-scatter albedo [1]
   //  ga(pcols,0:pver,nswbands)     ! asymmetry factor [1]
   //  fa(pcols,0:pver,nswbands)     ! forward scattered fraction [1]

   // ! Local variables
   // integer :: icol, jj, isw, kk, ll, mm    ! indices
   // integer :: list_idx                 ! index of the climate or a diagnostic list                    
   // integer :: nspec
   // integer :: istat
   // integer :: itim_old           ! index

   // real(r8) :: mass(pcols,pver)        ! layer dry mass [kg/m2]
   // real(r8) :: air_density(pcols,pver) ! dry air density [kg/m3]

   // real(r8),    pointer :: specmmr(:,:)        ! species mass mixing ratio [kg/kg]
   // character*32         :: spectype            ! species type
   // real(r8)             :: hygro_aer           ! hygroscopicity [1]

   // real(r8) :: sigma_logr_aer         ! geometric standard deviation of number distribution
   // real(r8) :: radsurf(pcols,pver)    ! aerosol surface mode radius
   // real(r8) :: logradsurf(pcols,pver) ! log(aerosol surface mode radius)
   // real(r8) :: cheb(ncoef,pcols,pver) ! chebychev polynomial parameters

   // real(r8),allocatable :: specvol(:,:)        ! volume concentration of aerosol specie [m3/kg]
   // real(r8),allocatable :: specdens(:)         ! species density for all species [kg/m3]
   // complex(r8),allocatable :: specrefindex(:,:)     ! species refractive index

   // real(r8)    :: refr(pcols)     ! real part of refractive index
   // real(r8)    :: refi(pcols)     ! imaginary part of refractive index
   // complex(r8) :: crefin(pcols)   ! complex refractive index

   // real(r8) :: dryvol(pcols)   ! volume concentration of aerosol mode [m3/kg]
   // real(r8) :: watervol(pcols) ! volume concentration of water in each mode [m3/kg]
   // real(r8) :: wetvol(pcols)   ! volume concentration of wet mode [m3/kg]

   // integer  :: itab(pcols), jtab(pcols)  ! index for Bilinear interpolation
   // real(r8) :: ttab(pcols), utab(pcols)  ! coef for Bilinear interpolation
   // real(r8) :: cext(pcols,ncoef), cabs(pcols,ncoef), casm(pcols,ncoef)  ! coefficient for extinction, absoption, and asymmetry [unitless] 
   // real(r8) :: pext(pcols)     ! parameterized specific extinction [m2/kg]
   // real(r8) :: specpext(pcols) ! specific extinction [m2/kg]
   // real(r8) :: dopaer(pcols)   ! aerosol optical depth in layer [1]
   // real(r8) :: pabs(pcols)     ! parameterized specific absorption [m2/kg]
   // real(r8) :: pasm(pcols)     ! parameterized asymmetry factor [unitless?]
   // real(r8) :: palb(pcols)     ! parameterized single scattering albedo [unitless]

   // ! Diagnostics
   // real(r8) :: tropopause_m(pcols)         ! tropopause height [m]
   // real(r8) :: extinct(pcols,pver)         ! aerosol extinction [1/m]
   // real(r8) :: absorb(pcols,pver)          ! aerosol absorption [1/m]
   // real(r8) :: aodvis(pcols)               ! extinction optical depth
   // real(r8) :: aodall(pcols)               ! extinction optical depth
   // real(r8) :: aodabs(pcols)               ! absorption optical depth

   // real(r8) :: aodabsbc(pcols)             ! absorption optical depth of BC

   // real(r8) :: burdenmode(pcols)           ! aerosol burden for each mode [kg/m2]
   // real(r8) :: aodmode(pcols)              ! aerosol optical depth for each mode [1]

   // real(r8) :: dustaodmode(pcols)          ! dust aod in aerosol mode [1]
   // real(r8) :: dustvol(pcols)              ! volume concentration of dust in aerosol mode (m3/kg)

   // real(r8) :: burdendust(pcols), burdenso4(pcols), burdenbc(pcols), &
   //             burdenpom(pcols), burdensoa(pcols), burdenseasalt(pcols), &
   //             burdenmom(pcols)  ! burden for each aerosol species [kg/m2]
   // real(r8) :: scatdust(pcols), scatso4(pcols), scatbc(pcols), &
   //             scatpom(pcols), scatsoa(pcols), scatseasalt(pcols), &
   //             scatmom(pcols)  ! scattering albedo (?) for each aerosol species [unitless]
   // real(r8) :: absdust(pcols), absso4(pcols), absbc(pcols), &
   //             abspom(pcols), abssoa(pcols), absseasalt(pcols), &
   //             absmom(pcols)   ! absoprtion for each aerosol species [unit?] 
   // real(r8) :: hygrodust(pcols), hygroso4(pcols), hygrobc(pcols), &
   //             hygropom(pcols), hygrosoa(pcols), hygroseasalt(pcols), &
   //             hygromom(pcols)  ! hygroscopicity for each aerosol species [unitless]

   // real(r8) :: ssavis(pcols)    ! Aerosol singel-scatter albedo [unitless]
   // real(r8) :: specrefr, specrefi  ! real and imag parts of specref
   // real(r8) :: scath2o, absh2o   ! scattering and absorption of h2o
   // real(r8) :: sumscat, sumabs, sumhygro ! sum of scattering , absoprtion and hygroscopicity

   // ! total species AOD
   // real(r8) :: dustaod(pcols), so4aod(pcols), bcaod(pcols), &
   //             pomaod(pcols), soaaod(pcols), seasaltaod(pcols), momaod(pcols)

   // logical :: savaervis ! true if visible wavelength (0.55 micron)
   // logical :: savaernir ! true if near ir wavelength (~0.88 micron)
   // logical :: savaeruv  ! true if uv wavelength (~0.35 micron)

   // real(r8) :: aoduv(pcols)               ! extinction optical depth in uv
   // real(r8) :: aodnir(pcols)              ! extinction optical depth in nir

   // character(len=32) :: outname
	rga
	rair

   for (int kk = 0; kk < pver; ++kk)
   {
   	  mass(kk)        = pdeldry(kk)*rga;
      air_density(kk) = pmid(kk)/(rair*temperature(kk));
   }

   // Calculate aerosol size distribution parameters and aerosol water uptake
   //For prognostic aerosols
   modal_aero_calcsize_sub(ncol, lchnk, state_q, pdel, dt, qqcw, 
                list_idx_in=list_idx, update_mmr_in = .false., 
                dgnumdry_m=dgnumdry_m); // ! out

   modal_aero_wateruptake_dr(lchnk, ncol, state_q, temperature, pmid,  
                                  cldn, dgnumdry_m, 
                                  dgnumwet_m, qaerwat_m, 
                                  list_idx_in=list_idx   ); 
   //
      // ! loop over all aerosol modes
   nspec_amode[ntot_amode]
   sigmag_amode[ntot_amode]

   for (int mm = 0; mm < ntot_amode; ++mm)
   {
   	// ! diagnostics for visible band for each mode
   	burdenmode = zero;
   	aodmode = zero;
   	dustaodmode=zero;
   	// ! get mode info
    const int nspec = nspec_amode[mm];
    const Real sigma_logr_aer = sigmag_amode[mm];
    
    for (int kk = top_lev; kk < pver; ++kk)
    {

     auto cheb_kk =  Kokkos::subview(cheb, kk, Kokkos::ALL());	
     modal_size_parameters(sigma_logr_aer,
                           dgnumwet_m(kk,mm),// in
                           radsurf(kk), logradsurf(kk),
                           cheb_kk,      
                           false );
    } // kk

    for (int i = 0; i < nswbands; ++i)
    {
    const bool savaervis = isw == idx_sw_diag ? true : false;
    const bool savaeruv =  isw == idx_uv_diag ? true : false;
    const bool savaernir = isw == idx_nir_diag ? true : false;

    kk = top_lev, pver

    for (int kk = top_lev; kk < pver; ++kk)
    {
    	dustvol = zero;
        scatdust     = zero;
            absdust      = zero;
            hygrodust    = zero;
            scatso4      = zero;
            absso4       = zero;
            hygroso4     = zero;
            scatbc       = zero;
            absbc        = zero;
            hygrobc      = zero;
            scatpom      = zero;
            abspom       = zero;
            hygropom     = zero;
            scatsoa      = zero;
            abssoa       = zero;
            hygrosoa     = zero;
            scatseasalt  = zero;
            absseasalt   = zero;
            hygroseasalt = zero;
            scatmom      = zero;
            absmom       = zero;
            hygromom     = zero;

            // ! aerosol species loop
    for (int ll = 0; ll < nspec; ++ll)
    {
      // get aerosol properties and save for each species
      auto specmmr     = Kokkos::subview(state_q(Kokkos::ALL(),lmassptr_amode[ll][mm]));
      auto spectype  = specname_amode(lspectype_amode[ll][mm]);
      const Real hygro_aer        = spechygro(lspectype_amode[ll][mm]);
      specdens(ll)     = specdens_amode(lspectype_amode[ll][mm]);
      // FIXME is a complex number
      // specrefindex(ll,:) = specrefndxsw(:,lspectype_amode[ll][mm])
      // allocate(specvol(pcols,nspec),stat=istat)
      specvol(ll)    = specmmr(kk)/specdens(ll);

      // ! compute some diagnostics for visible band only
      if (savaervis) {
        /// FIXME complex
      	const Real specrefr = specrefindex(ll,isw).real(); 
        const Real specrefi = specrefindex(ll,isw).imag(); 

        burdenmode += specmmr(kk)*mass(kk);
        // FIXME, use enums
        if (spectype==AeroId::DST)
        {

          calc_diag_spec( specmmr(kk),
                          mass(kk),
                          specvol(ll), 
                          specrefr, 
                          specrefi, 
                          hygro_aer,
						  burdendust,
                    scatdust,
                    absdust,
                    hygrodust);	
          dustvol  = specvol(ll);
        }

        if (spectype == AeroId:SO4) {
        	calc_diag_spec( specmmr(kk),
                          mass(kk),
                          specvol(ll), 
                          specrefr, 
                          specrefi, 
                          hygro_aer,
						  burdenso4,
                    scatso4,
                    absso4,
                    hygroso4);
        }

        if (spectype == AeroId:BC) {
        	calc_diag_spec( specmmr(kk),
                          mass(kk),
                          specvol(ll), 
                          specrefr, 
                          specrefi, 
                          hygro_aer,
						  burdenbc,
                    scatbc,
                    absbc,
                    hygrobc);
        }

        if (spectype == AeroId:POM) {
        	calc_diag_spec( specmmr(kk),
                          mass(kk),
                          specvol(ll), 
                          specrefr, 
                          specrefi, 
                          hygro_aer,
						  burdenpom,
                    scatpom,
                    abspom,
                    hygropom);
        }

        if (spectype == AeroId:SOA) {
        	calc_diag_spec( specmmr(kk),
                          mass(kk),
                          specvol(ll), 
                          specrefr, 
                          specrefi, 
                          hygro_aer,
						  burdensoa,
                    scatsoa,
                    abssoa,
                    hygrosoa);
        }

        if (spectype == AeroId:NaCl) {
        	calc_diag_spec( specmmr(kk),
                          mass(kk),
                          specvol(ll), 
                          specrefr, 
                          specrefi, 
                          hygro_aer,
						  burdenseasalt,
                    scatseasalt,
                    absseasalt,
                    hygroseasalt);
        }

        if (spectype == AeroId:MOM) {
        	calc_diag_spec( specmmr(kk),
                          mass(kk),
                          specvol(ll), 
                          specrefr, 
                          specrefi, 
                          hygro_aer,
						  burdenmom,
                    scatmom,
                    absmom,
                    hygromom);
        }

      } //savaervis

    }// ll species loop ll

    // lw =0 and sw =1
    Real dryvol, wetvol, watervol= {};
    Kokkos::complex<Real> crefin = {};
    Real refr, refi = {};

    calc_refin_complex(1, 
	                   isw,
                       qaerwat_m(kk,mm), 
                       specvol,
                       specrefindex, 
                       dryvol, wetvol, watervol,
                       crefin, refr, refi);

    // interpolate coefficients linear in refractive index
    // first call calcs itab,jtab,ttab,utab
    const auto sub_extpsw = Kokkos::subview(extpsw, mm, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),isw); 
    const auto ref_real_tab = Kokkos::subview(refrtabsw, mm, Kokkos::ALL(),isw); 
    const auto ref_img_tab = Kokkos::subview(refitabsw, mm, Kokkos::ALL(),isw); 


    itab=zero;
    itab_1=zero;
    // FIXME: part of binterp only apply for first column 
    binterp(sub_extpsw,
	         refr
	         refi, 
	         ref_real_tab,
	         ref_img_tab,
             itab, jtab,
             ttab, utab,
             cext,
             itab_1);

    const auto sub_abspsw = Kokkos::subview(abspsw, mm, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),isw); 

    binterp(sub_abspsw,
	         refr
	         refi, 
	         ref_real_tab,
	         ref_img_tab,
             itab, jtab,
             ttab, utab,
             cabs,
             itab_1);


     const auto sub_asmpsw = Kokkos::subview(asmpsw, mm, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),isw); 

    binterp(sub_asmpsw,
	         refr
	         refi, 
	         ref_real_tab,
	         ref_img_tab,
             itab, jtab,
             ttab, utab,
             casm,
             itab_1);


    // parameterized optical properties
    const auto cheb_k = Kokkos::subview(cheb, Kokkos::ALL(), kk); 
    calc_parameterized(cext,
                       cheb_k, 
                       pext);

    calc_parameterized(cabs,
                       cheb_k, 
                       pabs);

    calc_parameterized(casm,
                       cheb_k, 
                       pasm);

    //  do icol=1,ncol

    if (logradsurf(kk) <= xrmax)
    {
      pext = haero::exp(pext);
    } else 
    {
      // BAD CONSTANT	
      pext(icol) = 1.5/(radsurf(kk)*rhoh2o);// ! geometric optics 	
    }// if logradsurf(kk) <= xrmax

    // convert from m2/kg water to m2/kg aerosol
    specpext = pext;
    pext *= wetvol*rhoh2o;
    pabs *= wetvol*rhoh2o;
    pabs = mam4::utils:min_max_bound(zero, pext, pabs);
    palb = one-pabs/haero::max(pext, small_value_40);
    dopaer = pext*mass(kk);
    if (savaeruv)
    {
     aoduv += dopaer;
    } // savaeruv
    if (savaernir)
    {
     aodnir += dopaer;
    } // savaernir
    // end cols

    // Save aerosol optical depth at longest visible wavelength
    // sum over layers
    if (savaervis){
    	// ! aerosol extinction (/m)
        // do icol = 1, ncol
      extinct(kk) += dopaer*air_density(kk)/mass(kk);
      absorb(kk)  += pabs*air_density(kk);
      aodvis    += dopaer;
      aodall += dopaer;
      aodabs += pabs*mass(kk);
      aodmode += dopaer;
      ssavis += dopaer*palb;

      if (wetvol > small_value_40)
      {
      	dustaodmode += dopaer*dustvol/wetvol;
      	// partition optical depth into contributions from each constituent
        // assume contribution is proportional to refractive index X volume

        scath2o  = watervol*crefwsw(isw).real();
        absh2o   = -watervol*crefwsw(isw).imag();
        sumscat  = scatso4 + scatpom + scatsoa + scatbc + 
                   scatdust + scatseasalt + scath2o + scatmom;
        sumabs   = absso4 + abspom + abssoa + absbc + 
                   absdust + absseasalt + absh2o + absmom;
        sumhygro = hygroso4 + hygropom + hygrosoa + hygrobc + 
                   hygrodust + hygroseasalt + hygromom;


        //
                   update_aod_spec(scath2o,
                     absh2o, // in
                     sumhygro, 
                     sumscat,
                     sumabs, // in
                     hygrodust, 
                     palb,
                     dopaer,     // in
                     scatdust,
                     absdust, 
                     dustaod );

 update_aod_spec(scath2o,
                     absh2o, // in
                     sumhygro, 
                     sumscat,
                     sumabs, // in
                     hygroso4, 
                     palb,
                     dopaer,     // in
                     scatso4,
                     absso4, 
                     so4aod );  

 update_aod_spec(scath2o,
                     absh2o, // in
                     sumhygro, 
                     sumscat,
                     sumabs, // in
                     hygropom, 
                     palb,
                     dopaer,     // in
                     scatpom,
                     abspom, 
                     pomaod );  
                                                             

 update_aod_spec(scath2o,
                     absh2o, // in
                     sumhygro, 
                     sumscat,
                     sumabs, // in
                     hygrosoa, 
                     palb,
                     dopaer,     // in
                     scatsoa,
                     abssoa, 
                     soaaod );  

 update_aod_spec(scath2o,
                     absh2o, // in
                     sumhygro, 
                     sumscat,
                     sumabs, // in
                     hygrobc, 
                     palb,
                     dopaer,     // in
                     scatbc,
                     absbc, 
                     bcaod );  

 update_aod_spec(scath2o,
                     absh2o, // in
                     sumhygro, 
                     sumscat,
                     sumabs, // in
                     hygroseasalt, 
                     palb,
                     dopaer,     // in
                     scatseasalt,
                     absseasalt, 
                     seasaltaod );  

 update_aod_spec(scath2o,
                     absh2o, // in
                     sumhygro, 
                     sumscat,
                     sumabs, // in
                     hygromom, 
                     palb,
                     dopaer,     // in
                     scatmom,
                     absmom, 
                     momaod );  

    aodabsbc(icol) += absbc*dopaer*(one-palb);


      } // if wetvol(icol) > small_value_40

    } // savaervis

     // do icol = 1, ncol
    //FIXME: I did not port the following:
     // call check_error_warning('sw', icol, kk, mm, isw, nspec,list_idx, & ! in
     //                    dopaer(icol), pabs(icol), dryvol, wetvol, watervol, crefin,cabs,& ! in
     //                    specdens, specrefindex, specvol, & ! in
     //                    nerr_dopaer, & ! inout
     //                    pext(icol), specpext(icol) ) ! optional in

     tauxar(kk,isw) += dopaer;
     wa(kk,isw)     += dopaer*palb;
     ga(kk,isw)     += dopaer*palb*pasm;
     fa(kk,isw)     += dopaer*palb*pasm*pasm;
    // enddo
    } // k
    }// isw




   } // mm

   if(is_cmip6_volc){
   	calc_volc_ext(trop_level,
                  state_zm,
                  ext_cmip6_sw, 
                  extinct,
                  tropopause_m);

   }




}
#endif

} // namespace modal_aer_opt

} // end namespace mam4

#endif	