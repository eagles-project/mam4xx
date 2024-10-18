#ifndef MAM4XX_MICROPHYSICS_GAS_PHASE_CHEM_DR_HPP
#define MAM4XX_MICROPHYSICS_GAS_PHASE_CHEM_DR_HPP

#include <mam4xx/mo_photo.hpp>
#include <mam4xx/mo_setext.hpp>
#include <mam4xx/mo_setinv.hpp>
#include <mam4xx/mo_setsox.hpp>

namespace mam4 {

namespace microphysics {

/**
 * Performs a series of atmospheric chemistry and microphysics calculations
 * for a given atmospheric column. This includes setting external forcings,
 * computing ozone column density, setting up photolysis work arrays, and
 * performing aerosol microphysics calculations among other steps.
 *
 * @param cnst_offline_icol View into tracer constants.
 * @param forcings_in External forcings input array.
 * @param atm Atmospheric state.
 * @param progs Prognostic variables involved in the calculations.
 * @param adv_mass_kg_per_moles Conversion factor or relevant constant for mass
 * to moles conversion.
 * @param photo_table Photolysis rate tables or related data.
 * @param dt Time step for the simulation.
 * @param rlats Real latitude value, possibly for regional calculations.
 * @param chlorine_loading Chlorine loading parameter for chemistry
 * calculations.
 * @param config Configuration parameters.
 * @param o3_col_dens_i Output view for ozone column density.
 * @param photo_rates Photolysis rates for the column.
 * @param extfrc_icol External forcings for the column.
 * @param invariants_icol Invariant atmospheric parameters.
 * @param work_photo_table_icol Work arrays for photolysis calculations.
 * @param offset_aerosol Offset index for aerosol constituents.
 * @param o3_sfc, o3_tau, o3_lbl Parameters for ozone surface chemistry.
 * @param dry_diameter_icol, wet_diameter_icol, wetdens_icol Views for aerosol
 * properties.
 */

// number of species with external forcing
using mam4::gas_chemistry::extcnt;
using mam4::mo_photo::PhotoTableData;
using mam4::mo_setext::Forcing;
using mam4::mo_setinv::num_tracer_cnst;

using View2D = DeviceType::view_2d<Real>;
using View1D = DeviceType::view_1d<Real>;
KOKKOS_INLINE_FUNCTION
void perform_atmospheric_chemistry_and_microphysics(
    const ThreadTeam &team, const Real dt, const Real rlats,
    const View1D cnst_offline_icol[num_tracer_cnst], const Forcing *forcings_in,
    const haero::Atmosphere &atm, mam4::Prognostics &progs,
    const PhotoTableData &photo_table, const Real chlorine_loading,
    const mam4::mo_setsox::Config &config_setsox,
    const mam4::microphysics::AmicPhysConfig &config_amicphys,
    const Real linoz_psc_T, const Real zenith_angle_icol,
    const Real d_sfc_alb_dir_vis_icol, const View1D &o3_col_dens_i,
    const View2D &photo_rates_icol, const View2D &extfrc_icol,
    const View2D &invariants_icol, const View1D &work_photo_table_icol,
    const View1D &linoz_o3_clim_icol, const View1D &linoz_t_clim_icol,
    const View1D &linoz_o3col_clim_icol, const View1D &linoz_PmL_clim_icol,
    const View1D &linoz_dPmL_dO3_icol, const View1D &linoz_dPmL_dT_icol,
    const View1D &linoz_dPmL_dO3col_icol,
    const View1D &linoz_cariolle_pscs_icol, const Real eccf,
    const Real adv_mass_kg_per_moles[gas_pcnst],
    const int (&clsmap_4)[gas_pcnst], const int (&permute_4)[gas_pcnst],
    const int offset_aerosol, const Real o3_sfc, const Real o3_tau,
    const int o3_lbl, const View2D &dry_diameter_icol,
    const View2D &wet_diameter_icol, const View2D &wetdens_icol) {

  mam4::mo_setext::extfrc_set(forcings_in, extfrc_icol);

  mam4::mo_setinv::setinv(team,                                    // in
                          invariants_icol,                         // out
                          atm.temperature, atm.vapor_mixing_ratio, // in
                          cnst_offline_icol, atm.pressure);        // in

  mam4::microphysics::compute_o3_column_density(team, atm, progs,      // in
                                                invariants_icol,       // in
                                                adv_mass_kg_per_moles, // in
                                                o3_col_dens_i);        // out

  // set up photolysis work arrays for this column.
  mam4::mo_photo::PhotoTableWorkArrays photo_work_arrays_icol;

  //  set work view using 1D photo_work_arrays_icol
  // Note: We are not allocating views here.
  mam4::mo_photo::set_photo_table_work_arrays(photo_table,
                                              work_photo_table_icol,   // in
                                              photo_work_arrays_icol); // out

  mam4::mo_photo::table_photo(photo_rates_icol,                          // out
                              atm.pressure, atm.hydrostatic_dp,          // in
                              atm.temperature, o3_col_dens_i,            // in
                              zenith_angle_icol, d_sfc_alb_dir_vis_icol, // in
                              atm.liquid_mixing_ratio, atm.cloud_fraction, // in
                              eccf, photo_table,                           // in
                              photo_work_arrays_icol); // out

  // compute aerosol microphysics on each vertical level within this
  // column
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](const int kk) {
    // extract atm state variables (input)
    Real temp = atm.temperature(kk);
    Real pmid = atm.pressure(kk);
    Real pdel = atm.hydrostatic_dp(kk);
    Real zm = atm.height(kk);
    Real pblh = atm.planetary_boundary_layer_height;
    Real qv = atm.vapor_mixing_ratio(kk);
    Real cldfrac = atm.cloud_fraction(kk);
    Real lwc = atm.liquid_mixing_ratio(kk);
    Real cldnum = atm.cloud_liquid_number_mixing_ratio(kk);

    // extract aerosol state variables into "working arrays" (mass
    // mixing ratios) (in EAM, this is done in the gas_phase_chemdr
    // subroutine defined within
    //  mozart/mo_gas_phase_chemdr.F90)
    Real state_q[pcnst] = {};
    Real qqcw_pcnst[pcnst] = {};
    // output (state_q)
    mam4::utils::extract_stateq_from_prognostics(progs, atm, state_q, kk);
    // output (qqcw_pcnst)
    mam4::utils::extract_qqcw_from_prognostics(progs, qqcw_pcnst, kk);

    Real qq[gas_pcnst] = {};
    Real qqcw[gas_pcnst] = {};
    for (int i = offset_aerosol; i < pcnst; ++i) {
      qq[i - offset_aerosol] = state_q[i];
      qqcw[i - offset_aerosol] = qqcw_pcnst[i];
    }
    // convert mass mixing ratios to volume mixing ratios (VMR),
    // equivalent to tracer mixing ratios (TMR))
    Real vmr[gas_pcnst], vmrcw[gas_pcnst];
    // output (vmr)
    mam4::microphysics::mmr2vmr(qq, adv_mass_kg_per_moles, vmr);
    // output (vmrcw)
    mam4::microphysics::mmr2vmr(qqcw, adv_mass_kg_per_moles, vmrcw);

    //---------------------
    // Gas Phase Chemistry
    //---------------------
    //
    const auto &extfrc_k = ekat::subview(extfrc_icol, kk);
    const auto &invariants_k = ekat::subview(invariants_icol, kk);
    const auto &photo_rates_k = ekat::subview(photo_rates_icol, kk);

    // vmr0 stores mixing ratios before chemistry changes the mixing
    // ratios
    Real vmr0[gas_pcnst] = {};
    mam4::microphysics::gas_phase_chemistry(
        // in
        temp, dt, photo_rates_k.data(), extfrc_k.data(), invariants_k.data(),
        clsmap_4, permute_4,
        // out
        vmr, vmr0);

    // create work array copies to retain "pre-chemistry (aqueous)"
    // values
    Real vmr_pregas[gas_pcnst] = {};
    Real vmr_precld[gas_pcnst] = {};
    for (int i = 0; i < gas_pcnst; ++i) {
      vmr_pregas[i] = vmr[i];
      vmr_precld[i] = vmrcw[i];
    }

    //----------------------
    // Aerosol microphysics
    //----------------------
    // the logic below is taken from the aero_model_gasaerexch
    // subroutine in eam/src/chemistry/modal_aero/aero_model.F90

    // aqueous chemistry ...
    const Real mbar = haero::Constants::molec_weight_dry_air;
    constexpr int indexm = mam4::gas_chemistry::indexm;
    mam4::mo_setsox::setsox_single_level(
        // in
        offset_aerosol, dt, pmid, pdel, temp, mbar, lwc, cldfrac, cldnum,
        invariants_k[indexm], config_setsox,
        // out
        vmrcw, vmr);

    // calculate aerosol water content using water uptake treatment
    // * dry and wet diameters [m]
    // * wet densities [kg/m3]
    // * aerosol water mass mixing ratio [kg/kg]
    Real dgncur_a_kk[num_modes] = {};
    Real dgncur_awet_kk[num_modes] = {};
    Real wetdens_kk[num_modes] = {};
    Real qaerwat_kk[num_modes] = {};

    for (int imode = 0; imode < num_modes; imode++) {
      dgncur_awet_kk[imode] = wet_diameter_icol(imode, kk);
      dgncur_a_kk[imode] = dry_diameter_icol(imode, kk);
      wetdens_kk[imode] = wetdens_icol(imode, kk);
    }
    // do aerosol microphysics (gas-aerosol exchange, nucleation,
    // coagulation)
    mam4::microphysics::modal_aero_amicphys_intr(
        // in
        config_amicphys, dt, temp, pmid, pdel, zm, pblh, qv, cldfrac,
        // out
        vmr, vmrcw,
        // in
        vmr0, vmr_pregas, vmr_precld, dgncur_a_kk, dgncur_awet_kk, wetdens_kk);

    mam4::microphysics::vmr2mmr(vmrcw, adv_mass_kg_per_moles, qqcw);

    //-----------------
    // LINOZ chemistry
    //-----------------

    // the following things are diagnostics, which we're not
    // including in the first rev
    Real do3_linoz, do3_linoz_psc, ss_o3, o3col_du_diag, o3clim_linoz_diag,
        zenith_angle_degrees;

    // index of "O3" in solsym array (in EAM)
    constexpr int o3_ndx = static_cast<int>(mam4::GasId::O3);
    mam4::lin_strat_chem::lin_strat_chem_solve_kk(
        // in
        o3_col_dens_i(kk), temp, zenith_angle_icol, pmid, dt, rlats,
        linoz_o3_clim_icol(kk), linoz_t_clim_icol(kk),
        linoz_o3col_clim_icol(kk), linoz_PmL_clim_icol(kk),
        linoz_dPmL_dO3_icol(kk), linoz_dPmL_dT_icol(kk),
        linoz_dPmL_dO3col_icol(kk), linoz_cariolle_pscs_icol(kk),
        chlorine_loading, linoz_psc_T,
        // out
        vmr[o3_ndx],
        // outputs that are not used
        do3_linoz, do3_linoz_psc, ss_o3, o3col_du_diag, o3clim_linoz_diag,
        zenith_angle_degrees);

    // Update source terms above the ozone decay threshold
    if (kk >= nlev - o3_lbl) {
      Real o3l_vmr, do3mass;
      // initial O3 vmr
      o3l_vmr = vmr[o3_ndx];
      // get new value of O3 vmr
      o3l_vmr = mam4::lin_strat_chem::lin_strat_sfcsink_kk(dt, pdel, // in
                                                           o3l_vmr,  // out
                                                           o3_sfc,   // in
                                                           o3_tau,   // in
                                                           do3mass); // out
      // Update the mixing ratio (vmr) for O3
      vmr[o3_ndx] = o3l_vmr;
    }
    // Check for negative values and reset to zero
    for (int i = 0; i < gas_pcnst; ++i) {
      if (vmr[i] < 0.0)
        vmr[i] = 0.0;
    }

    //----------------------
    // Dry deposition (gas)
    //----------------------

    // FIXME: drydep integration in progress!
    // mam4::drydep::drydep_xactive(...);

    mam4::microphysics::vmr2mmr(vmr, adv_mass_kg_per_moles, qq);

    for (int i = offset_aerosol; i < pcnst; ++i) {
      state_q[i] = qq[i - offset_aerosol];
      qqcw_pcnst[i] = qqcw[i - offset_aerosol];
    }
    mam4::utils::inject_stateq_to_prognostics(state_q, progs, kk);
    mam4::utils::inject_qqcw_to_prognostics(qqcw_pcnst, progs, kk);
  }); // parallel_for for vertical levels
}

} // namespace microphysics
} // namespace mam4
#endif
