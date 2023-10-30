#ifndef MAM4XX_MO_SETEXT_HPP
#define MAM4XX_MO_SETEXT_HPP

#include <haero/math.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace mo_setext {
  using View1D = DeviceType::view_1d<Real>;
using View2D = DeviceType::view_2d<Real>;
using View3D = DeviceType::view_3d<Real>;
constexpr int pver = mam4::nlev;
// FIXME get this constant
constexpr int extfrc_cnt = 9;
// FIXME check if this constant is defined somewhere else.
constexpr int extcnt = 9; //, & ! number of species with external forcing

KOKKOS_INLINE_FUNCTION
void extfrc_set(
                const int *forcings_frc_ndx,
                const int *forcings_nsectors,
                const bool *forcings_file_alt_data,
                const View1D forcings_fields_data[extfrc_cnt][4], const View2D &frcing) {
  /*--------------------------------------------------------
  ! ... form the external forcing
  !--------------------------------------------------------*/

  // ncol                  ! columns in chunk
  // lchnk                 ! chunk index
  // zint(ncol, pverp)     ! interface geopot above surface [km]
  // frcing(ncol,pver,extcnt)   ! insitu forcings [molec/cm^3/s]
  // Note: we do not need zint to compute frcing
  // const ColumnView &zint, 

  constexpr Real zero = 0.0;

  // frcing(:,:,:) = zero;
  if (extfrc_cnt < 1 || extcnt < 1) {
    return;
  }

  /*--------------------------------------------------------
  ! ... set non-zero forcings
  !--------------------------------------------------------*/

  for (int mm = 0; mm < extfrc_cnt; ++mm) {
    // Fortran to C++ indexing
    const int nn = forcings_frc_ndx[mm]-1;
    for (int kk = 0; kk < pver; ++kk) {
      frcing(kk, nn) = zero;
    } // k

    for (int isec = 0; isec < forcings_nsectors[mm]; ++isec) {
      if (forcings_file_alt_data[mm]) {
        for (int kk = 0; kk < pver; ++kk) {  
          // frcing(:ncol,:,nn) = frcing(:ncol,:,nn) + &
                                // forcings(mm)%fields(isec)%data(:ncol,pver:1:-1,lchnk)
          frcing(kk, nn) += forcings_fields_data[mm][isec](pver-1-kk);
        } // kk
      } else {
        // // forcings(mm)%fields(isec)%data(:ncol,:,lchnk)
        for (int kk = 0; kk < pver; ++kk) {
          frcing(kk, nn) += forcings_fields_data[mm][isec](kk);
        }
      }
    } // isec

    //  xfcname = trim(forcings(mm)%species)//'_XFRC'
    // call outfld( xfcname, frcing(:ncol,:,nn), ncol, lchnk )

    // frcing_col(:ncol) = 0._r8
    // do kk = 1,pver
    //    frcing_col(:ncol) = frcing_col(:ncol) + &
    //                  frcing(:ncol,kk,nn)*(zint(:ncol,kk)-zint(:ncol,kk+1))*km_to_cm
    // enddo
    // xfcname = trim(forcings(mm)%species)//'_CLXF'
    // call outfld( xfcname, frcing_col(:ncol), ncol, lchnk )

  } // end mm
} // extfrc_set

KOKKOS_INLINE_FUNCTION
void setext(const int *forcings_frc_ndx,
            const int *forcings_nsectors, const bool *forcings_file_alt_data,
            const View1D forcings_fields_data[extfrc_cnt][4],
            const View2D &extfrc) // ! out
{

  /*--------------------------------------------------------
  !     ... for this latitude slice:
  !         - form the production from datasets
  !         - form the nox (xnox) production from lighing
  !         - form the nox (xnox) production from airplanes
  !         - form the co production from airplanes
  !--------------------------------------------------------*/

  // @param[in]   zint(ncol,pver+1)           ! interface geopot height [km]
  // @param[out]  extfrc(ncol,pver,extcnt)    ! the "extraneous" forcing

  /*--------------------------------------------------------
    !     ... local variables
    !--------------------------------------------------------
    ! variables for output. in current MAM4 they are not calculated and are
    assigned zero real(r8), dimension(ncol,pver) :: no_lgt, no_air, co_air */

  /*--------------------------------------------------------
  !     ... set frcing from datasets
  !--------------------------------------------------------*/
  extfrc_set(
             forcings_frc_ndx, forcings_nsectors, forcings_file_alt_data,
             forcings_fields_data,
             extfrc); // out

  /*--------------------------------------------------------
  !     ... set nox production from lighting
  !         note: from ground to cloud top production is c shaped
  !
  ! FORTRAN refactor: nox is not included in current MAM4
  ! the related code are removed but outfld is kept for BFB testing
  !--------------------------------------------------------*/
  // no_lgt(:,:) = 0._r8
  // call outfld( 'NO_Lightning', no_lgt(:ncol,:), ncol, lchnk )

  // ! FORTRAN refactor: in the subroutine airpl_set, has_airpl_src is false,
  // ! the subroutine only has two outfld calls that output zero
  // ! remove the subroutine call and move out the zero outfld
  // no_air(:,:) = 0._r8
  // co_air(:,:) = 0._r8
  // call outfld( 'NO_Aircraft',  no_air(:ncol,:), ncol, lchnk )
  // call outfld( 'CO_Aircraft',  co_air(:ncol,:), ncol, lchnk )

} // setext

} // namespace mo_setext

} // end namespace mam4

#endif