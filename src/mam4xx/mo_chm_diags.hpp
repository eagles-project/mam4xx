#ifndef MAM4XX_MO_CHM_DIAGS_HPP
#define MAM4XX_MO_CHM_DIAGS_HPP

#include <haero/math.hpp>
#include <mam4xx/gas_chem.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

namespace mo_chm_diags {

using Real = haero::Real;
using View1D = DeviceType::view_1d<Real>;

// FIXME: bad constants
constexpr Real S_molwgt = 32.066;
constexpr Real rearth = 6.37122e6;
constexpr Real rgrav =
    1.0 / 9.80616; // reciprocal of acceleration of gravity ~ m/s^2
constexpr Real avogadro = haero::Constants::avogadro;
constexpr const int gas_pcnst = gas_chemistry::gas_pcnst;
constexpr const int pver = mam4::nlev; // number of vertical levels

KOKKOS_INLINE_FUNCTION
void het_diags(
    const ThreadTeam &team,
    const ColumnView het_rates[gas_pcnst], //[pver][gas_pcnst], //in
    const ColumnView mmr[gas_pcnst],       //[pver][gas_pcnst],
    const ColumnView &pdel,                //[pver],
    const Real &wght,
    View1D wrk_wd, //[gas_pcnst], //output
    // Real noy_wk, //output //this isn't actually used in this function?
    View1D sox_wk, // output
    // Real nhx_wk, //output //this isn't actually used in this function?
    const Real adv_mass[gas_pcnst], // constant from elsewhere
    const Real sox_species[3]) {
  // change to pass values for a single col
  //===========
  // output integrated wet deposition field
  //===========

  sox_wk[0] = 0;
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, gas_pcnst), KOKKOS_LAMBDA(int mm) {
        //
        // compute vertical integral
        //
        wrk_wd(mm) = 0;

        for (int kk = 0; kk < pver; kk++) {
          wrk_wd(mm) += het_rates[mm](kk) * mmr[mm](kk) *
                        pdel(kk); // parallel_reduce in the future?
        }

        wrk_wd(mm) *= rgrav * wght * haero::square(rearth);
      });

  for (int mm = 0; mm < gas_pcnst; mm++) {
    for (int i = 0; i < 3; i++) { // FIXME: bad constant (len of sox species)
      if (sox_species[i] == mm)
        sox_wk[0] += wrk_wd[mm] * S_molwgt / adv_mass[mm];
    }
  }
} // het_diags

} // namespace mo_chm_diags
} // namespace mam4
#endif