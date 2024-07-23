#ifndef MAM4XX_AERO_MODEL_EMISSIONS_HPP
#define MAM4XX_AERO_MODEL_EMISSIONS_HPP

#include <haero/atmosphere.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/gas_chem_mechanism.hpp>
#include <mam4xx/mam4_types.hpp>

namespace mam4::aero_model_emissions {

constexpr int n_online_emiss = 9;

// FIXME: keeping this here, just in case
// struct OnlineEmissionsDataFields {
//   std::vector<std::string> SO2_data_fields = {"BB", "ENE_ELEV", "IND_ELEV", "contvolc"};
//   std::vector<std::string> SOAG_data_fields = {"SOAbb_src", "SOAbg_src", "SOAff_src"};
//   std::vector<std::string> bc_a4_data_fields = {"BB"};
//   std::vector<std::string> num_a1_data_fields = {"num_a1_SO4_ELEV_BB", "num_a1_SO4_ELEV_ENE", "num_a1_SO4_ELEV_IND", "num_a1_SO4_ELEV_contvolc"};
//   std::vector<std::string> num_a2_data_fields = {"num_a2_SO4_ELEV_contvolc"};
//   std::vector<std::string> num_a4_data_fields = {"num_a1_BC_ELEV_BB", "num_a1_POM_ELEV_BB"};
//   std::vector<std::string> pom_a4_data_fields = {"BB"};
//   std::vector<std::string> so4_a1_data_fields = {"BB", "ENE_ELEV", "IND_ELEV", "contvolc"};
//   std::vector<std::string> so4_a2_data_fields = {"contvolc"};

//   OnlineEmissionsDataFields() = default;
// };

std::map<std::string, std::vector<std::string>> const online_emimssions_data_fields{
  {"SO2", {"BB", "ENE_ELEV", "IND_ELEV", "contvolc"}},
  {"SOAG", {"SOAbb_src", "SOAbg_src", "SOAff_src"}},
  {"bc_a4", {"BB"}},
  {"num_a1", {"num_a1_SO4_ELEV_BB", "num_a1_SO4_ELEV_ENE", "num_a1_SO4_ELEV_IND", "num_a1_SO4_ELEV_contvolc"}},
  {"num_a2", {"num_a2_SO4_ELEV_contvolc"}},
  {"num_a4", {"num_a1_BC_ELEV_BB", "num_a1_POM_ELEV_BB"}},
  {"pom_a4", {"BB"}},
  {"so4_a1", {"BB", "ENE_ELEV", "IND_ELEV", "contvolc"}},
  {"so4_a2", {"contvolc"}}
};

// nbin corresponds to dst_a1, dst_a3
const int dust_nbin = 2;
// nnum corresponds to num_a1, num_a3
const int dust_nnum = 2;
// number of entries in the dust flux input array
const int dust_nflux_in = 4;
// Kok11: fractions of bin (0.1-1) and bin (1-10) in size 0.1-10
const Real dust_emis_sclfctr[dust_nbin] = {0.011, 0.989};

const Real dust_emis_fact = -1.0e36;

const int salt_nsection = 31;
// FIXME: Unnecessary indices
const int sec1_beg = 1;
const int sec1_end = 9;
const int sec2_beg = 10;
const int sec2_end = 13;
const int sec3_beg = 14;
const int sec3_end = 21;
const int sec4_beg = 22;
const int sec4_end = salt_nsection;
// FIXME: BAD CONSTANTS--so many...
// only use up to ~20um
const Real Dg[salt_nsection] = {
    0.0020e-5, 0.0025e-5, 0.0032e-5, 0.0040e-5, 0.0051e-5, 0.0065e-5,
    0.0082e-5, 0.0104e-5, 0.0132e-5, 0.0167e-5, 0.0211e-5, 0.0267e-5,
    0.0338e-5, 0.0428e-5, 0.0541e-5, 0.0685e-5, 0.0867e-5, 0.1098e-5,
    0.1389e-5, 0.1759e-5, 0.2226e-5, 0.2818e-5, 0.3571e-5, 0.4526e-5,
    0.5735e-5, 0.7267e-5, 0.9208e-5, 1.1668e-5, 1.4786e-5, 1.8736e-5,
    2.3742e-5
    };

namespace {
// use Ekmam's ss
// radius [m]
Real rdry[salt_nsection];
// multiply rm with 1.814 because it should be RH=80% and not dry particles
// for the parameterization
// [um]
Real rm[salt_nsection];
// use in [Manahan]
Real bm[salt_nsection];

// constants for calculating emission polynomial
Real consta[salt_nsection], constb[salt_nsection];
}

KOKKOS_INLINE_FUNCTION
void seasalt_emis() {} // end seasalt_emis()
// seasalt_emis(lchnk, ncol, fi, cam_in%ocnfrac, seasalt_emis_scale, &  // in
KOKKOS_INLINE_FUNCTION
void marine_organic_emis() {} // end marine_organic_emis()
// marine_organic_emis(lchnk, ncol, fi, cam_in%ocnfrac, seasalt_emis_scale, & // in

KOKKOS_INLINE_FUNCTION
void dust_emis(Real dust_density, Real *dust_dmt_vwr, Real soil_erodibility, Real dust_flux_in[dust_nflux_in], Real cflux[2 * dust_nbin]) {
              //  Real dust_flux_in, Real cflx, Real soil_erod) {
// subroutine dust_emis( ncol, lchnk, dust_flux_in, &  // in
//                       cflx, &                       // inout
//                       soil_erod )                   // out
//   use soil_erod_mod, only : soil_erod_fact
//   use soil_erod_mod, only : soil_erodibility
//   use mo_constants,  only : dust_density
//   use physconst,     only : pi

// args
//   integer,  intent(in)    :: ncol, lchnk
//   real(r8), intent(in)    :: dust_flux_in(:,:) // dust emission fluxes in four bins from land [kg/m^2/s]
//   real(r8), intent(inout) :: cflx(:,:)         // emission fluxes in MAM modes [kg/m^2/s or #/m^2/s]
//   real(r8), intent(out)   :: soil_erod(:)      // soil erodibility factor [unitless]

// local vars
//   integer :: icol, ibin, idx_dst, inum
//   real(r8) :: dst_mass_to_num(dust_nbin)

  // FIXME: BAD CONSTANT
  Real soil_erod_threshold = 0.1;
  // HACK: made this up
  int dust_indices[dust_nbin] = {3};

  // dust_density[dust_nbin] -- Mass weighted diameter resolved [m]
  // HACK: made this up
  Real dust_mass_to_num[dust_nbin] = {0.0};
  // HACK: made this up
  Real dust_emis_scalefactor[dust_nbin] = {1.1};
  // HACK: made this up
  Real soil_erod_fact = 0.42;

  for (int ibin = 0; ibin < dust_nbin; ++ibin) {
    // FIXME: BAD CONSTANT
    dust_mass_to_num[ibin] = 6.0 / (haero::Constants::pi * dust_density * haero::pow(dust_dmt_vwr[ibin], 3));
  }

  // local version (necessary?) that's zero if less than the threshold
  Real soil_erod = soil_erodibility < soil_erod_threshold ? soil_erodibility : 0.0;

  Real dust_flux_neg_sum;
  for (int ibin = 0; ibin < dust_nflux_in; ++ibin) {
    dust_flux_neg_sum -= dust_flux_in[ibin];
  }

  // rebin and adjust dust emissions..

  // Correct the dust input flux calculated by CLM, which uses size distribution in Zender03
  // to calculate fraction of bin (0.1-10um) in range (0.1-20um) = 0.87
  // based on Kok11, that fraction is 0.73
  // FIXME: BAD CONSTANTS
  Real frac_ratio = 0.73 / 0.87;
  for (int ibin = 0; ibin < dust_nbin; ++ibin) {
    int idx_dust = dust_indices[ibin];
    // FIXME: BAD CONSTANT
    cflux[idx_dust] = dust_flux_neg_sum * frac_ratio * dust_emis_scalefactor[ibin] * soil_erod / soil_erod_fact * 1.15;
    int inum = dust_indices[ibin + dust_nbin];
    // TODO: maybe make this a separate array, since the first half is input and
    // also work array(?), and the second half is output?
    cflux[inum] = cflux[idx_dust] * dust_mass_to_num[ibin];
  }
} // end dust_emis()

KOKKOS_INLINE_FUNCTION
void init_salt_constants() {
  // use Ekman's ss
  // multiply rm with 1.814 because it should be RH=80% and not dry particles
  // for the parameterization

  for (int i = 0; i < salt_nsection; ++i) {
    rdry[i] = Dg[i] / 2.0;
    rm[i] = 1.814 * rdry[i] * 1.0e6;
    bm[i] = (0.380 - haero::log10(rm[i])) / 0.65;
  }

  // FIXME: BAD CONSTANTS BAD CONSTANTS
  // calculate constants form emission polynomials
  // NOTE: I simplified some very ugly math here, so if things go wonky,
  // check that first
  for (int isec = sec1_beg; isec < sec1_end; ++isec) {
    consta[isec] =  -2.576e35 * haero::pow(Dg[isec], 4)
                   + 5.932e28 * haero::pow(Dg[isec], 3)
                   - 2.867e21 * haero::pow(Dg[isec], 2)
                   - 3.003e13 * Dg[isec]
                   - 2.881e6;
    constb[isec] =   7.188e37 * haero::pow(Dg[isec], 4)
                   - 1.616e31 * haero::pow(Dg[isec], 3)
                   + 6.791e23 * haero::pow(Dg[isec], 2)
                   + 1.829e16 * Dg[isec]
                   + 0.609e8;
  }
  for (int isec = sec2_beg; isec < sec2_end; ++isec) {
    consta[isec] =  -2.452e33 * haero::pow(Dg[isec], 4)
                   + 2.404e27 * haero::pow(Dg[isec], 3)
                   - 8.148e20 * haero::pow(Dg[isec], 2)
                   + 1.183e14 * Dg[isec]
                   - 6.743e6;
    constb[isec] =   7.368e35 * haero::pow(Dg[isec], 4)
                   - 7.310e29 * haero::pow(Dg[isec], 3)
                   + 2.528e23 * haero::pow(Dg[isec], 2)
                   - 3.787e16 * Dg[isec]
                   + 0.279e9;
  }
  for (int isec = sec3_beg; isec < sec3_end; ++isec) {
    consta[isec] =   1.085e29 * haero::pow(Dg[isec], 4)
                   - 9.841e23 * haero::pow(Dg[isec], 3)
                   + 3.132e18 * haero::pow(Dg[isec], 2)
                   - 4.165e12 * Dg[isec]
                   + 0.181e6;
    constb[isec] =  -2.859e31 * haero::pow(Dg[isec], 4)
                   + 2.601e26 * haero::pow(Dg[isec], 3)
                   - 8.297e20 * haero::pow(Dg[isec], 2)
                   + 1.105e15 * Dg[isec]
                   - 5.800e8;
  }
  for (int isec = sec4_beg; isec < sec4_end; ++isec) {
    // use monahan
    consta[isec] = (
                     1.373 * haero::pow(rm[isec], -3)
                     * (1 + 0.057 * haero::pow(rm[isec], 1.05))
                     * haero::pow(10,
                                  1.19 * haero::exp(haero::square(bm[isec])))
                    )
                   * (rm[isec] - rm[isec - 1]);
  }
}  // end init_salt_constants

KOKKOS_INLINE_FUNCTION
void calc_seasalt_fluxes(Real surface_temp, Real u10cubed,
                         Real fluxes[salt_nsection]) {
                        //  TODO: does this get updated?
  // Calculations of source strength and size distribution
  // NB the 0.1 is the dlogDp we have to multiply with to get the flux, but the
  // value depends of course on what dlogDp you have.
  // You will also have to change the sections of Dg if you use a different number
  // of size bins with different intervals.

  // whitecap area
  // FIXME: BAD CONSTANTS
  const Real wc_area = 3.84e-6 * u10cubed * 0.1;

  // calculate number flux fluxes (# / m2 / s)
  for (int isec = sec1_beg; isec < sec1_end; ++isec) {
    fluxes[isec] = wc_area * (surface_temp * consta[isec] + constb[isec]);
  }
  for (int isec = sec2_beg; isec < sec2_end; ++isec) {
    fluxes[isec] = wc_area * (surface_temp * consta[isec] + constb[isec]);
  }
  for (int isec = sec3_beg; isec < sec3_end; ++isec) {
    fluxes[isec] = wc_area * (surface_temp * consta[isec] + constb[isec]);
  }
  for (int isec = sec4_beg; isec < sec4_end; ++isec) {
    fluxes[isec] = consta[isec] * u10cubed;
  }
} // end calc_seasalt_fluxes()

KOKKOS_INLINE_FUNCTION
void calculate_seasalt_numflux_in_bins(Real surface_temp, Real u_bottom,
                                       Real v_bottom, Real z_bottom,
                                       Real salt_numflux[salt_nsection]) {

  // surface_temp: sea surface temperature [K]
  // fi[nsections]: sea salt number fluxes in each size bin [# / m2 / s]

  // m roughness length over oceans--from ocean model
  // FIXME: BAD CONSTANT
  const Real z0 = 0.0001;

  // Needed in Gantt et al. calculation of organic mass fraction
  Real u10 = haero::sqrt(haero::square(u_bottom) + haero::square(v_bottom));

  // move the winds to 10m high from the midpoint of the gridbox:
  // follows Tie and Seinfeld and Pandis, p.859 with math.
  // u10cubed is defined as the 3.41 power of 10m wind
  // FIXME: terrible variable name
  // FIXME: BAD CONSTANT
  Real u10cubed = u10 * haero::log(10.0 / z0) / haero::log(z_bottom / z0);

  // we need them to the 3.41 power, according to Gong et al., 1997
  // FIXME: BAD CONSTANT
  u10cubed = haero::pow(u10cubed, 3.41);

  calc_seasalt_fluxes(surface_temp, u10cubed, salt_numflux);
} // end calculate_seasalt_numflux_in_bins()

KOKKOS_INLINE_FUNCTION
void aero_model_emissions() {


} // end aero_model_emissions()
    // ( state, & // in
//       cam_in ) // inout
//   use seasalt_model, only: seasalt_emis, marine_organic_emis, seasalt_names, seasalt_indices, seasalt_active,seasalt_nbin, &
//                            nslt_om
//   use sslt_sections, only: nsections
//   use dust_model,    only: dust_emis, dust_names, dust_indices, dust_active,dust_nbin, dust_nnum

//   // Arguments:

//   type(physics_state),    intent(in)    :: state   // Physics state variables
//   type(cam_in_t),         intent(inout) :: cam_in  // import state

//   // local vars

//   integer :: lchnk, ncol
//   integer :: ispec, spec_idx
//   real(r8) :: soil_erod_tmp(pcols)
//   real(r8) :: sflx(pcols)   // accumulate over all bins for output
//   real(r8) :: srf_temp(pcols)   // sea surface temperature [K]
//   real(r8) :: fi(pcols, nsections)        // sea salt number fluxes in each size bin [#/m2/s]


//   lchnk = state%lchnk
//   ncol = state%ncol
//   srf_temp = cam_in%sst

//   call dust_emis( ncol, lchnk, cam_in%dstflx, & // in
//                   cam_in%cflx, &                // inout
//                   soil_erod_tmp )               // out

//   // some dust emis diagnostics ...
//   sflx(:)=0.
//   do ispec = 1, dust_nbin+dust_nnum
//      spec_idx = dust_indices(ispec)
//      if (ispec<=dust_nbin) sflx(:ncol)=sflx(:ncol)+cam_in%cflx(:ncol,spec_idx)
//      call outfld(trim(dust_names(ispec))//'SF',cam_in%cflx(:,spec_idx),pcols, lchnk)
//   enddo
//   call outfld('DSTSFMBL',sflx(:),pcols,lchnk)
//   call outfld('LND_MBL',soil_erod_tmp(:),pcols, lchnk )

//   call calculate_seasalt_numflux_in_bins(ncol, cam_in%sst, state%u(:ncol,pver), state%v(:ncol,pver),  state%zm(:ncol,pver), & // in
//                                          fi) // out

//   sflx(:)=0.

//   call seasalt_emis(lchnk, ncol, fi, cam_in%ocnfrac, seasalt_emis_scale, &  // in
//                     cam_in%cflx)                                                              // inout

//   call marine_organic_emis(lchnk, ncol, fi, cam_in%ocnfrac, seasalt_emis_scale, & // in
//                            cam_in%cflx)                                                             // inout

//   // Write out salt mass fluxes to history files
//   do ispec = 1, seasalt_nbin-nslt_om
//      spec_idx = seasalt_indices(ispec)
//      sflx(:ncol)=sflx(:ncol)+cam_in%cflx(:ncol,spec_idx)
//      call outfld(trim(seasalt_names(ispec))//'SF',cam_in%cflx(:,spec_idx),pcols,lchnk)
//   enddo
//   // accumulated flux
//   call outfld('SSTSFMBL',sflx(:),pcols,lchnk)

//   // Write out marine organic mass fluxes to history files
//   sflx(:)=0.
//   do ispec = seasalt_nbin-nslt_om+1, seasalt_nbin
//      spec_idx = seasalt_indices(ispec)
//      sflx(:ncol)=sflx(:ncol)+cam_in%cflx(:ncol,spec_idx)
//      call outfld(trim(seasalt_names(ispec))//'SF',cam_in%cflx(:,spec_idx),pcols,lchnk)
//   enddo

//   // accumulated flux
//   call outfld('SSTSFMBL_OM',sflx(:),pcols,lchnk)

// end subroutine aero_model_emissions

} // namespace mam4::aero_model_emissions
#endif
