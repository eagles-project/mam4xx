#ifndef MAM4XX_AER_RAD_PROPS_HPP
#define MAM4XX_AER_RAD_PROPS_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>

#include <mam4xx/modal_aer_opt.hpp>

namespace mam4 {

namespace aer_rad_props {

// From radconstants
constexpr int nswbands = modal_aer_opt::nswbands;
constexpr int nlwbands = modal_aer_opt::nlwbands;
using View2D = DeviceType::view_2d<Real>;

KOKKOS_INLINE_FUNCTION
void volcanic_cmip_sw(const ColumnView &zi, const int ilev_tropp,
                      const View2D &ext_cmip6_sw_inv_m,
                      const View2D &ssa_cmip6_sw, const View2D &af_cmip6_sw,
                      const View2D &tau, const View2D &tau_w,
                      const View2D &tau_w_g, const View2D &tau_w_f) {

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
  // tau_w_f(pcols,0:pver,nswbands) ! aerosol forward scattered fraction * tau *
  // w

  // !Local variables
  // icol, ipver, ilev_tropp
  // lyr_thk ! thickness between level interfaces [m]
  // ext_unitless(nswbands), asym_unitless(nswbands)
  // ext_ssa(nswbands),ext_ssa_asym(nswbands)

  // !Logic:
  // !Update taus, tau_w, tau_w_g and tau_w_f with the read in volcanic
  // !aerosol extinction (1/km), single scattering albedo and asymmtry factors.

  // !Above the tropopause, the read in values from the file include both the
  // stratospheric !and volcanic aerosols. Therefore, we need to zero out taus
  // above the tropopause !and populate them exclusively from the read in
  // values.

  // !If tropopause is found, update taus with 50% contributuions from the
  // volcanic input !file and 50% from the existing model computed values

  // !First handle the case of tropopause layer itself:
  // do icol = 1, ncol
  // ilev_tropp = trop_level(icol) !tropopause level
  //
  constexpr Real half = 0.5;

  const Real lyr_thk = zi(ilev_tropp) - zi(ilev_tropp + 1);
  for (int i = 0; i < nswbands; ++i) {
    const Real ext_unitless = lyr_thk * ext_cmip6_sw_inv_m(ilev_tropp, i);
    const Real asym_unitless = af_cmip6_sw(ilev_tropp, i);
    const Real ext_ssa = ext_unitless * ssa_cmip6_sw(ilev_tropp, i);
    const Real ext_ssa_asym = ext_ssa * asym_unitless;

    tau(ilev_tropp, i) = half * (tau(ilev_tropp, i) + ext_unitless);
    tau_w(ilev_tropp, i) = half * (tau_w(ilev_tropp, i) + ext_ssa);
    tau_w_g(ilev_tropp, i) = half * (tau_w_g(ilev_tropp, i) + ext_ssa_asym);
    tau_w_f(ilev_tropp, i) =
        half * (tau_w_f(ilev_tropp, i) + ext_ssa_asym * asym_unitless);
  } // end i

  // !As it will be more efficient for FORTRAN to loop over levels and then
  // columns, the following loops !are nested keeping that in mind Note that in
  // C++ ported code, the loop over levels is nested. Thus, the previous comment
  // does not apply.

  // ilev_tropp = trop_level(icol) !tropopause level
  for (int kk = 0; kk < ilev_tropp; ++kk) {

    const Real lyr_thk = zi(kk) - zi(kk + 1);
    for (int i = 0; i < nswbands; ++i) {
      const Real ext_unitless = lyr_thk * ext_cmip6_sw_inv_m(kk, i);
      const Real asym_unitless = af_cmip6_sw(kk, i);
      const Real ext_ssa = ext_unitless * ssa_cmip6_sw(kk, i);
      const Real ext_ssa_asym = ext_ssa * asym_unitless;
      tau(kk, i) = ext_unitless;
      tau_w(kk, i) = ext_ssa;
      tau_w_g(kk, i) = ext_ssa_asym;
      tau_w_f(kk, i) = ext_ssa_asym * asym_unitless;

    } // end nswbands

  } // kk

} // volcanic_cmip_sw
// FIXME; to move compute_odap_volcanic_at_troplayer_lw to a new file,
// aer_rad_props
KOKKOS_INLINE_FUNCTION
void compute_odap_volcanic_at_troplayer_lw(const int ilev_tropp,
                                           const ColumnView &zi,
                                           const View2D &ext_cmip6_lw_inv_m,
                                           const View2D &odap_aer) {
  // Update odap_aer with a combination read in volcanic aerosol extinction
  // [1/m] (50%) and module computed values (50%).

  // intent-ins
  //  ncol
  // trop_level(:)
  // zi(:,:) !geopotential height above surface at interfaces [m]
  // ext_cmip6_lw_inv_m(:,:,:) !long wave extinction in the units of [1/m]

  //! intent-inouts
  //  odap_aer(:,:,:)  ! [fraction] absorption optical depth, per layer
  //  [unitless]

  // !local
  // integer :: icol, ilev_tropp
  // real(r8) :: lyr_thk !layer thickness [m]
  // do icol = 1, ncol
  // ilev_tropp = trop_level(icol) !tropopause level
  const Real lyr_thk =
      zi(ilev_tropp) - zi(ilev_tropp + 1); //! compute layer thickness in meters
  constexpr Real half = 0.5;
  //! update taus with 50% contributuions from the volcanic input file
  //! and 50% from the existing model computed values at the tropopause layer
  for (int i = 0; i < nlwbands; ++i) {
    odap_aer(ilev_tropp, i) =
        half * (odap_aer(ilev_tropp, i) +
                (lyr_thk * ext_cmip6_lw_inv_m(ilev_tropp, i)));
  }

} // compute_odap_volcanic_at_troplayer_lw

KOKKOS_INLINE_FUNCTION
void compute_odap_volcanic_above_troplayer_lw(const int ilev_tropp,
                                              const ColumnView &zi,
                                              const View2D &ext_cmip6_lw_inv_m,
                                              const View2D &odap_aer) {

  //     !Above the tropopause, the read in values from the file include both
  //     the stratospheric
  // !and volcanic aerosols. Therefore, we need to zero out odap_aer above the
  // tropopause !and populate it exclusively from the read in values.

  // !intent-ins
  // integer, intent(in) :: pver, ncol
  // integer, intent(in) :: trop_level(:)

  // real(r8), intent(in) :: zi(:,:) !geopotential height above surface at
  // interfaces [m] real(r8), intent(in) :: ext_cmip6_lw_inv_m(:,:,:) !long wave
  // extinction in the units of [1/m]

  // !intent-inouts
  // real(r8), intent(inout) :: odap_aer(:,:,:) ! [fraction] absorption optical
  // depth, per layer [unitless]

  // !local
  // integer :: ipver, icol, ilev_tropp
  // real(r8) :: lyr_thk !layer thickness [m]

  // !As it will be more efficient for FORTRAN to loop over levels and then
  // columns, the following loops !are nested keeping that in mind

  for (int kk = 0; kk < ilev_tropp; ++kk) {
    const Real lyr_thk =
        zi(kk) - zi(kk + 1); // ! compute layer thickness in meters
    for (int i = 0; i < nlwbands; ++i) {
      odap_aer(kk, i) = lyr_thk * ext_cmip6_lw_inv_m(kk, i);
    } // end i

  } // end kk

} // compute_odap_volcanic_above_troplayer_lw

} // namespace aer_rad_props

} // end namespace mam4

#endif