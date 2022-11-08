#ifndef MAM4XX_GASAEREXCH_SOAEXCH_HPP
#define MAM4XX_GASAEREXCH_SOAEXCH_HPP

#include "mam4xx/aero_config.hpp"

#include <haero/haero.hpp>
#include <haero/math.hpp>

namespace mam4 {
namespace gasaerexch {

using Real = haero::Real;

// ==============================================================================
// Calculate SOA species's eiquilibrium vapor mixing ratio under the ambient
// condition, ignoring the solute effect.
//
// History:
// - MAM box model by R. C. Easter, PNNL.
// - Subroutine cretated duing code refactoring by Hui Wan, PNNL, 2022.
// ==============================================================================
KOKKOS_INLINE_FUNCTION
void soa_equilib_mixing_ratio_no_solute(const int ntot_soaspec,
                                        const Real &T_in_K, 
					const Real &p_in_Pa,
                                        const Real &pstd_in_Pa,
                                        const Real r_universal,
                                        const Real &g0_soa) {

  // ntot_soaspec     number of SOA species
  // T_in_K           temperature in Kelvin
  // p_in_Pa          air pressure in Pascal
  // pstd_in_Pa       standard air pressure in Pascal
  // r_universal      universal gas constant in J/K/mol
#if 0
  real(wp), intent(out) :: g0_soa(ntot_soaspec)   ! ambient soa gas equilib mixing ratio (mol/mol at actual mw)

  ! Local variables

  real(wp) :: p0_soa(ntot_soaspec)             ! soa gas equilib vapor presssure (atm)
  integer  :: ll                               ! loop index for SOA species

  ! Parameters: should they be changed to intent(in)? - Not ncessary for now, but perhaps
  ! need to change to intent in when we get multiple SOA species.

  real(wp) :: p0_soa_at_298K(ntot_soaspec)     ! p0_soa_298 = soa gas equilib vapor presssure (atm) at 298 K
  real(wp) :: delh_vap_soa(ntot_soaspec)       ! delh_vap_soa = heat of vaporization for gas soa (J/mol)

  !---------------------
  ! Set parameter values
  !---------------------
    delh_vap_soa(:) = 156.0e3_wp
  p0_soa_at_298K(:) = 1.0e-10_wp

  !----------------------------------------------------------------
  ! Calculate equilibrium mixing ratio under the ambient condition
  !----------------------------------------------------------------
  do ll = 1, ntot_soaspec

     ! The equilibrium vapor pressure

     p0_soa(ll) = p0_soa_at_298K(ll) * &
                  exp( -( delh_vap_soa(ll)/(r_universal/1.e3_wp) )* &
                        ( (1._wp/T_in_K)-(1._wp/298._wp) )          )

     ! Convert to mixing ratio in mol/mol

     g0_soa(ll) = pstd_in_Pa*p0_soa(ll)/p_in_Pa

  end do

end subroutine soa_equilib_mixing_ratio_no_solute
#endif
}

KOKKOS_INLINE_FUNCTION
void mam_soaexch_1subarea(
    const int num_soamode, const int npca, const int npoa, const int soag,
    const Real dt, const Real dtsub_soa_fixed, const Real pstd,
    const Real r_universal, const Real &temp, const Real &pmid,
    const Real uptkaer[AeroConfig::num_gas_ids()][AeroConfig::num_modes()],
    const Real qaer_poa[1][AeroConfig::num_modes()],
    Real qgas_cur[AeroConfig::num_gas_ids()],
    Real qgas_avg[AeroConfig::num_gas_ids()],
    Real qaer_cur[AeroConfig::num_aerosol_ids()][AeroConfig::num_modes()],
    Real &soa_out) {

  // static constexpr int num_mode = AeroConfig::num_modes();
  // static constexpr int num_gas = AeroConfig::num_gas_ids();
  // static constexpr int num_aer = AeroConfig::num_aerosol_ids();

  // ----------------------------------------------------------------------
  // SOA species's equilibrium vapor mixing ratio at the ambient T and p,
  // assuming no solute effect. (This equilibrium mixing ratio depends only on
  // T, p and the parameters used inside the subroutine. There is no dependence
  // on SOA gas or aerosol mixing ratios, so the calculation does not need to
  // repeat during temporal sub-cycling.)
  // -----------------------------------------------------------------------
  const int ntot_soaspec = 1;

  soa_equilib_mixing_ratio_no_solute(ntot_soaspec, temp, pmid, pstd,
                                     r_universal, soa_out);
}
} // namespace gasaerexch
} // namespace mam4
#endif
