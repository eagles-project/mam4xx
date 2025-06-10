#ifndef MAM4XX_MICROPHYSICS_GAS_PHASE_CHEM_DR_HPP
#define MAM4XX_MICROPHYSICS_GAS_PHASE_CHEM_DR_HPP

#include <mam4xx/aero_model.hpp>
#include <mam4xx/compute_o3_column_density.hpp>
#include <mam4xx/gas_phase_chemistry.hpp>
#include <mam4xx/lin_strat_chem.hpp>
#include <mam4xx/mam4_amicphys.hpp>
#include <mam4xx/mo_drydep.hpp>
#include <mam4xx/mo_photo.hpp>
#include <mam4xx/mo_setext.hpp>
#include <mam4xx/mo_sethet.hpp>
#include <mam4xx/mo_setinv.hpp>
#include <mam4xx/mo_setsox.hpp>
#include <mam4xx/seq_drydep.hpp>

namespace mam4 {

namespace microphysics {
// number of species with external forcing
using mam4::gas_chemistry::extcnt;
using mam4::mo_photo::PhotoTableData;
using mam4::mo_setext::Forcing;
using mam4::mo_setinv::num_tracer_cnst;

using View2D = DeviceType::view_2d<Real>;
using ConstView2D = DeviceType::view_2d<const Real>;
using View1D = DeviceType::view_1d<Real>;
using ConstView1D = DeviceType::view_1d<const Real>;

KOKKOS_INLINE_FUNCTION
void mmr2vmr_col(const ThreadTeam &team, const haero::Atmosphere &atm,
                 const mam4::Prognostics &progs,
                 const Real adv_mass_kg_per_moles[gas_pcnst],
                 const int offset_aerosol,
                 const ColumnView vmr_col[gas_pcnst]) {
  // Make a local copy of nlev to avoid the identifier "mam4::nlev" being
  // undefined in device code.
  constexpr int nlev_local = nlev;
  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, nlev_local), [&](const int kk) {
        Real state_q[pcnst] = {};
        mam4::utils::extract_stateq_from_prognostics(progs, atm, state_q, kk);
        Real qq[gas_pcnst] = {};
        for (int i = offset_aerosol; i < pcnst; ++i) {
          qq[i - offset_aerosol] = state_q[i];
        }
        Real vmr[gas_pcnst];
        // output (vmr)
        mam4::microphysics::mmr2vmr(qq, adv_mass_kg_per_moles, vmr);
        for (int i = 0; i < gas_pcnst; ++i) {
          vmr_col[i](kk) = vmr[i];
        }
      });
}
/**
 * Performs a series of atmospheric chemistry and microphysics calculations
 * for a given atmospheric column. This includes setting external forcings,
 * computing ozone column density, setting up photolysis work arrays, and
 * performing aerosol microphysics calculations among other steps.
 *
 *
 * @param [in] team Kokkos team type for the Kokkos::TeamThreadRange
 * @param [in] dt Time step
 * @param [in] rlats Column latitudes
 * @param [in] month
 * @param [in] sfc_temp      -- surface temperature [K]
 * @param [in] pressure_sfc  -- surface pressure [Pa]
 * @param [in] wind_speed    -- 10-meter wind spped [m/s]
 * @param [in] rain          -- rain content [kg/m2/s]
 * @param [in] solar_flux    -- direct shortwave surface radiation [W/m^2]
 * @param [in] cnst_offline_icol Invariant tracer
 * @param [in] forcings_in Struct for external forcing or vertical emissions
 * @param [in] atm  Atmosphere state variables
 * @param [in] [out] prognostics A collection of aerosol prognostic variables
 * @param [in] photo_table  photolysis table data
 * @param [in] chlorine_loading
 * @param [in] config_setsox conf struc for setsox
 * @param [in] config_amicphys conf struc for amicphys
 * @param [in] linoz_psc_T PSC ozone loss T (K) threshold
 * @param [in] zenith_angle_icol
 * @param [in] d_sfc_alb_dir_vis_icol
 * @param [out] o3_col_dens_i
 * @param [in] photo_rates_icol
 * @param [in] extfrc_icol
 * @param [in] invariants_icol
 * @param [in] work_photo_table_icol
 * @param [in] linoz_o3_clim_icol
 * @param [in] linoz_t_clim_icol
 * @param [in] linoz_o3col_clim_icol
 * @param [in] linoz_PmL_clim_icol
 * @param [in] linoz_dPmL_dO3_icol
 * @param [in] linoz_dPmL_dT_icol
 * @param [in] linoz_dPmL_dO3col_icol
 * @param [in] linoz_cariolle_pscs_icol
 * @param [in] eccf
 * @param [in] adv_mass_kg_per_moles
 * @param [in] mmr[gas_pcnst]  -- constituent MMRs [kg/kg]
 * @param [in] fraction_landuse
 * @param [in] col_index_season
 * @param [in] clsmap_4
 * @param [in] permute_4
 * @param [in] offset_aerosol
 * @param [in] o3_sfc
 * @param [in] o3_tau
 * @param [in] o3_lbl
 * @param [in] dry_diameter_icol
 * @param [in] wet_diameter_icol
 * @param [in] wetdens_icol
 * @param [in] seq_drydep::Data drydep_data = seq_drydep::set_gas_drydep_data()
 * @param [out] aqso4_flx[num_modes] So4 flux in kg/m2/s
 * @param [out] qh2so4_flx[num_modes] H2So4 flux in kg/m2/s
 * @param [out] dvel[gas_pcnst] -- deposition velocity [1/cm/s]
 * @param [out] dflx[gas_pcnst] -- deposition flux [1/cm^2/s]
 * @param [out] progs           -- prognostics: stateq, qqcw updated
 **/

KOKKOS_INLINE_FUNCTION
void perform_atmospheric_chemistry_and_microphysics(
    const ThreadTeam &team, const Real dt, const Real rlats,
    const Real sfc_temp, const Real pressure_sfc, const Real wind_speed,
    const Real rain, const Real solar_flux,
    const View1D cnst_offline_icol[num_tracer_cnst], const Forcing *forcings_in,
    const haero::Atmosphere &atm, const PhotoTableData &photo_table,
    const Real chlorine_loading, const mam4::mo_setsox::Config &config_setsox,
    const AmicPhysConfig &config_amicphys, const Real linoz_psc_T,
    const Real zenith_angle_icol, const Real d_sfc_alb_dir_vis_icol,
    const View1D &o3_col_dens_i, const View2D &photo_rates_icol,
    const View2D &extfrc_icol, const View2D &invariants_icol,
    const View1D &work_photo_table_icol, const View1D &linoz_o3_clim_icol,
    const View1D &linoz_t_clim_icol, const View1D &linoz_o3col_clim_icol,
    const View1D &linoz_PmL_clim_icol, const View1D &linoz_dPmL_dO3_icol,
    const View1D &linoz_dPmL_dT_icol, const View1D &linoz_dPmL_dO3col_icol,
    const View1D &linoz_cariolle_pscs_icol, const Real eccf,
    const Real adv_mass_kg_per_moles[gas_pcnst],
    const Real fraction_landuse[mam4::mo_drydep::n_land_type],
    const int index_season[mam4::mo_drydep::n_land_type],
    const int (&clsmap_4)[gas_pcnst], const int (&permute_4)[gas_pcnst],
    const int offset_aerosol, const Real o3_sfc, const Real o3_tau,
    const int o3_lbl, const ConstView2D dry_diameter_icol,
    const ConstView2D wet_diameter_icol, const ConstView2D wetdens_icol,
    const Real phis,      // surf geopotential //in
    const View1D &cmfdqr, // dq/dt for convection [kg/kg/s] //in ndx_cmfdqr =
                          // pbuf_get_index('RPRDTOT') // from convect shallow
    const ConstView1D
        &prain, // stratoform precip [kg/kg/s] //in precip_total_tend
    const ConstView1D &nevapr, // nevapr evaporation [kg/kg/s] //in
    const View1D &work_set_het, const seq_drydep::Data &drydep_data,
    const View1D &aqso4_flx, const View1D &aqh2so4_flx,
    Real dvel[gas_pcnst], // deposition velocity [cm/s]
    Real dflx[gas_pcnst], mam4::Prognostics &progs) {

  const int nlev = mam4::nlev;
  auto work_set_het_ptr = (Real *)work_set_het.data();
  const auto het_rates = View2D(work_set_het_ptr, nlev, gas_pcnst);
  work_set_het_ptr += nlev * gas_pcnst;
  const auto dqdt_aqso4 = View2D(work_set_het_ptr, gas_pcnst, nlev);
  work_set_het_ptr += nlev * gas_pcnst;
  const auto dqdt_aqh2so4 = View2D(work_set_het_ptr, gas_pcnst, nlev);
  work_set_het_ptr += nlev * gas_pcnst;

  // vmr0 stores mixing ratios before chemistry changes the mixing
  ColumnView vmr_col[gas_pcnst];
  for (int i = 0; i < gas_pcnst; ++i) {
    vmr_col[i] = ColumnView(work_set_het_ptr, nlev);
    work_set_het_ptr += nlev;
  }
  const int sethet_work_len = mam4::mo_sethet::get_work_len_sethet();
  const auto work_sethet_call = View1D(work_set_het_ptr, sethet_work_len);
  work_set_het_ptr += sethet_work_len;
  EKAT_KERNEL_ASSERT_MSG(
      mam4::mo_sethet::get_total_work_len_sethet() ==
          work_set_het_ptr - work_set_het.data(),
      "Error: Work buffer memory allocation exceeds expected value.");
  //
  const int surface_lev = nlev - 1; // Surface level
  // specific humidity [kg/kg]
  const Real spec_hum = atm.vapor_mixing_ratio(surface_lev);
  // surface air temperature [K]
  const Real air_temp = atm.temperature(surface_lev);
  // potential temperature [K] *(temp*(1+vapor_mixing_ratio))
  //(FIXME: We followed Fortran, compare it with MAM4xx's potential temp
  // func)
  const Real tv = air_temp * (1.0 + spec_hum);
  // 10-meter pressure [Pa]
  // Surface pressure at 10m (Followed the fortran code)
  const Real pressure_10m = atm.pressure(surface_lev);

  mam4::mo_setext::extfrc_set(team, forcings_in, extfrc_icol);

  mam4::mo_setinv::setinv(team,                                    // in
                          invariants_icol,                         // out
                          atm.temperature, atm.vapor_mixing_ratio, // in
                          cnst_offline_icol, atm.pressure);        // in
  team.team_barrier();
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

  team.team_barrier();
  mam4::mo_photo::table_photo(team, photo_rates_icol,                    // out
                              atm.pressure, atm.hydrostatic_dp,          // in
                              atm.temperature, o3_col_dens_i,            // in
                              zenith_angle_icol, d_sfc_alb_dir_vis_icol, // in
                              atm.liquid_mixing_ratio, atm.cloud_fraction, // in
                              eccf, photo_table,                           // in
                              photo_work_arrays_icol); // out
  team.team_barrier();
  // work array.
  // het_rates_icol work array.
  mmr2vmr_col(team, atm, progs, adv_mass_kg_per_moles, offset_aerosol, vmr_col);
  team.team_barrier();

  mam4::mo_sethet::sethet(team, atm, het_rates, rlats, phis, cmfdqr, prain,
                          nevapr, dt, invariants_icol, vmr_col,
                          work_sethet_call);
  team.team_barrier();
  {
    //----------------------
    // Dry deposition (gas)
    //----------------------
    Real state_q[pcnst] = {};
    mam4::utils::extract_stateq_from_prognostics(progs, atm, state_q, nlev - 1);
    Real qq[gas_pcnst] = {};
    for (int i = offset_aerosol; i < pcnst; ++i)
      qq[i - offset_aerosol] = state_q[i];

    team.team_barrier();
    mam4::mo_drydep::drydep_xactive(
        drydep_data,
        fraction_landuse, // fraction of land use for column by land type
        index_season,     // column-specific mapping of month indices to
                          // seasonal land-type indices [-]
        sfc_temp,         // surface temperature [K]
        air_temp,         // surface air temperature [K]
        tv,               // potential temperature [K]
        pressure_sfc,     // surface pressure [Pa]
        pressure_10m,     // 10-meter pressure [Pa]
        spec_hum,         // specific humidity [kg/kg]
        wind_speed,       // 10-meter wind spped [m/s]
        rain,             // rain content [??]
        solar_flux,       // direct shortwave surface radiation [W/m^2]
        qq,               // constituent MMRs [kg/kg]
        dvel,             // deposition velocity [1/cm/s]
        dflx              // deposition flux [1/cm^2/s]
    );
  }
  team.team_barrier();
  // compute aerosol microphysics on each vertical level within this
  // column
  const int o3_ndx = static_cast<int>(mam4::GasId::O3);
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
    // equivalent to tracer mixing ratios (TMR)
    Real vmr[gas_pcnst] = {}, vmrcw[gas_pcnst] = {};
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
    const auto &het_rates_k = ekat::subview(het_rates, kk);

    // vmr0 stores mixing ratios before chemistry changes the mixing
    // ratios
    Real vmr0[gas_pcnst] = {};
    mam4::microphysics::gas_phase_chemistry(
        // in
        temp, dt, photo_rates_k.data(), extfrc_k.data(), invariants_k.data(),
        clsmap_4, permute_4, het_rates_k.data(),
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
    Real dqdt_aqso4_t[gas_pcnst] = {};
    Real dqdt_aqh2so4_t[gas_pcnst] = {};
    mam4::mo_setsox::setsox_single_level(
        // in
        offset_aerosol, dt, pmid, pdel, temp, mbar, lwc, cldfrac, cldnum,
        invariants_k[indexm], config_setsox,
        // out
        dqdt_aqso4_t, dqdt_aqh2so4_t, vmrcw, vmr);
    team.team_barrier();
    for (int i = 0; i < gas_pcnst; ++i) {
      dqdt_aqso4(i, kk) = dqdt_aqso4_t[i];
      dqdt_aqh2so4(i, kk) = dqdt_aqh2so4_t[i];
    }
    // calculate aerosol water content using water uptake treatment
    // * dry and wet diameters [m]
    // * wet densities [kg/m3]
    // * aerosol water mass mixing ratio [kg/kg]
    Real dgncur_a_kk[num_modes] = {};
    Real dgncur_awet_kk[num_modes] = {};
    Real wetdens_kk[num_modes] = {};

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
    Real do3_linoz = 0, do3_linoz_psc = 0, ss_o3 = 0, o3col_du_diag = 0,
         o3clim_linoz_diag = 0, zenith_angle_degrees = 0;

    // index of "O3" in solsym array (in EAM)
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
      const Real o3l_vmr_old = vmr[o3_ndx];
      Real do3mass = 0;
      const Real o3l_vmr_new =
          mam4::lin_strat_chem::lin_strat_sfcsink_kk(dt, pdel,    // in
                                                     o3l_vmr_old, // in
                                                     o3_sfc,      // in
                                                     o3_tau,      // in
                                                     do3mass);    // out
      // Update the mixing ratio (vmr) for O3
      vmr[o3_ndx] = o3l_vmr_new;
    }
    // Check for negative values and reset to zero
    for (int i = 0; i < gas_pcnst; ++i) {
      if (vmr[i] < 0.0)
        vmr[i] = 0.0;
    }

    mam4::microphysics::vmr2mmr(vmr, adv_mass_kg_per_moles, qq);

    for (int i = offset_aerosol; i < pcnst; ++i) {
      state_q[i] = qq[i - offset_aerosol];
      qqcw_pcnst[i] = qqcw[i - offset_aerosol];
    }
    mam4::utils::inject_stateq_to_prognostics(state_q, progs, kk);
    mam4::utils::inject_qqcw_to_prognostics(qqcw_pcnst, progs, kk);
  }); // parallel_for for vertical levels
  team.team_barrier();
  // Diagnose the column-integrated flux (kg/m2/s) using
  // volume mixing ratios ( // kmol/kmol(air) )
  haero::ConstColumnView pdel = atm.hydrostatic_dp; // layer thickness (Pa)
  for (int m = 0; m < num_modes; ++m) {
    const int ll = config_setsox.lptr_so4_cw_amode[m] - offset_aerosol;
    if (0 <= ll) {
      const auto aqso4 = ekat::subview(dqdt_aqso4, ll);
      const auto aqh2so4 = ekat::subview(dqdt_aqh2so4, ll);
      const Real vmr_so4 = aero_model::calc_sfc_flux(team, aqso4, pdel, nlev);
      const Real vmr_h2s = aero_model::calc_sfc_flux(team, aqh2so4, pdel, nlev);
      aqso4_flx[m] =
          conversions::mmr_from_vmr(vmr_so4, adv_mass_kg_per_moles[ll]);
      aqh2so4_flx[m] =
          conversions::mmr_from_vmr(vmr_h2s, adv_mass_kg_per_moles[ll]);
    }
  }
}

} // namespace microphysics
} // namespace mam4
#endif
