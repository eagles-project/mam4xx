#ifndef MAM4XX_MICROPHYSICS_HPP
#define MAM4XX_MICROPHYSICS_HPP

#include <mam4xx/mo_photo.hpp>

namespace mam4 {

namespace microphysics {

using View1D = DeviceType::view_1d<Real>;
using ConstView1D = DeviceType::view_1d<const Real>;

// FIXME: check if we have ported these function in mam4xx. If no, let's move
// them there.
KOKKOS_INLINE_FUNCTION
void mmr2vmr(const Real q[mam4::gas_chemistry::gas_pcnst],  // in
             const Real mw[mam4::gas_chemistry::gas_pcnst], // in
             Real vmr[mam4::gas_chemistry::gas_pcnst])      // out
{
  for (int i = 0; i < mam4::gas_chemistry::gas_pcnst; ++i) {
    vmr[i] = mam4::conversions::vmr_from_mmr(q[i], mw[i]);
  }
} // mmr2vmr
// FIXME: check if we have ported these function in mam4xx. If no, let's move
// there.
KOKKOS_INLINE_FUNCTION
void vmr2mmr(const Real vmr[mam4::gas_chemistry::gas_pcnst], // in
             const Real mw[mam4::gas_chemistry::gas_pcnst],  // in
             Real q[mam4::gas_chemistry::gas_pcnst])         // out
{
  for (int i = 0; i < mam4::gas_chemistry::gas_pcnst; ++i) {
    q[i] = mam4::conversions::mmr_from_vmr(vmr[i], mw[i]);
  }
} // vmr2mmr

using View2D = DeviceType::view_2d<Real>;
KOKKOS_INLINE_FUNCTION
void compute_o3_column_density(
    const ThreadTeam &team, const haero::Atmosphere &atm,
    const mam4::Prognostics &progs,
    // const View2D &invariants,
    const Real adv_mass_kg_per_moles[mam4::gas_chemistry::gas_pcnst],
    ColumnView o3_col_dens) {
  constexpr int gas_pcnst =
      mam4::gas_chemistry::gas_pcnst; // number of gas phase species
  constexpr int offset_aerosol = mam4::utils::gasses_start_ind();
  constexpr int o3_idx = mam4::gas_chemistry::o3_idx;

  Real o3_col_deltas[mam4::nlev + 1] =
      {}; // o3 column density above model [1/cm^2]
  // NOTE: if we need o2 column densities, set_ub_col and setcol must be changed
  const int nlev = mam4::nlev;
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](const int k) {
    const Real pdel = atm.hydrostatic_dp(k);
    // extract aerosol state variables into "working arrays" (mass
    // mixing ratios) (in EAM, this is done in the gas_phase_chemdr
    // subroutine defined within
    //  mozart/mo_gas_phase_chemdr.F90)
    Real q[gas_pcnst] = {};
    Real state_q[pcnst] = {};
    mam4::utils::extract_stateq_from_prognostics(progs, atm, state_q, k);

    for (int i = offset_aerosol; i < pcnst; ++i) {
      q[i - offset_aerosol] = state_q[i];
    }

    // convert mass mixing ratios to volume mixing ratios (VMR),
    // equivalent to tracer mixing ratios (TMR))
    Real vmr[gas_pcnst] = {};
    mmr2vmr(q, adv_mass_kg_per_moles, vmr);

    // compute the change in o3 density for this column above its neighbor
    o3_col_deltas[k + 1] = mam4::mo_photo::set_ub_col(vmr[o3_idx], pdel);
  });
  team.team_barrier();
  // sum the o3 column deltas to densities
  mam4::mo_photo::setcol(team, o3_col_deltas, // in
                         o3_col_dens);        // out
}
KOKKOS_INLINE_FUNCTION
void compute_o3_column_density(const ThreadTeam &team, const ConstView1D &pdel,
                               const View1D &mmr_o3, const Real o3_col_deltas_0,
                               const Real mw_o3, const View1D &o3_col_dens) {
  constexpr Real xfactor = 2.8704e21 / (9.80616 * 1.38044); // BAD_CONSTANT!
  constexpr int nlev = mam4::nlev;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int kk) {
    Real suma = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(team, kk),
        [&](int i, Real &lsum) {
          const Real vmr_o3_i =
              mam4::conversions::vmr_from_mmr(mmr_o3(i), mw_o3);
          lsum += xfactor * pdel(i) * vmr_o3_i;
        },
        suma);
    const Real vmr_o3_kk = mam4::conversions::vmr_from_mmr(mmr_o3(kk), mw_o3);
    o3_col_dens(kk) =
        o3_col_deltas_0 + suma + 0.5 * xfactor * pdel(kk) * vmr_o3_kk;
  });
}
} // namespace microphysics
} // namespace mam4
#endif
