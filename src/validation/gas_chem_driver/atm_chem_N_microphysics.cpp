// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <mam4xx/aero_config.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
void atm_chem_N_microphysics(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Ensemble parameters
    // Declare array of strings for input names
    std::string input_arrays[] = {"asdir",
                                  "calday",
                                  "cflx",
                                  "cldfr",
                                  "cldw",
                                  "cmfdqr",
                                  "delt",
                                  "dgnum",
                                  "dgnumwet",
                                  "drydepflx",
                                  "eccen",
                                  "extcnt",
                                  "fieldname_len",
                                  "fsds",
                                  "gas_pcnst",
                                  "imozart",
                                  "indexm",
                                  "inv_ndx_cnst_o3",
                                  "lambm0",
                                  "lchnk",
                                  "linoz_PmL_clim",
                                  "linoz_cariolle_psc",
                                  "linoz_dPmL_dO3",
                                  "linoz_dPmL_dO3col",
                                  "linoz_dPmL_dT",
                                  "linoz_o3_clim",
                                  "linoz_o3col_clim",
                                  "linoz_t_clim",
                                  "map2chm",
                                  "mvelpp",
                                  "nabscol",
                                  "ncldwtr",
                                  "ncol",
                                  "ndx_h2so4",
                                  "nevapr",
                                  "nfs",
                                  "o3_ndx",
                                  "obliqr",
                                  "pblh",
                                  "pcnst",
                                  "pcols",
                                  "pdel",
                                  "pdeldry",
                                  "phis",
                                  "phtcnt",
                                  "pi",
                                  "pint",
                                  "pmid",
                                  "prain",
                                  "precc",
                                  "precl",
                                  "ps",
                                  "pver",
                                  "qqcw",
                                  "qtend",
                                  "rga",
                                  "rxntot",
                                  "rxt_tag_cnt",
                                  "rxt_tag_map",
                                  "snowhland",
                                  "state_q",
                                  "synoz_ndx",
                                  "tfld",
                                  "troplev",
                                  "ts",
                                  "ufld",
                                  "vfld",
                                  "wetdens",
                                  "zi",
                                  "zm"};

    // Iterate over input_arrays and error if not in input
    for (std::string name : input_arrays) {
      if (!input.has_array(name.c_str())) {
        std::cerr << "Required name for array: " << name << std::endl;
        exit(1);
      }
    }

    using mam4::gas_chemistry::extcnt;
    using mam4::mo_photo::PhotoTableData;
    using mam4::mo_setext::Forcing;
    using mam4::mo_setinv::num_tracer_cnst;

    using View2D = DeviceType::view_2d<Real>;
    using ConstView2D = DeviceType::view_2d<const Real>;
    using View1D = DeviceType::view_1d<Real>;

    const Real dt = input.get_array("delt")[0];
    // const Real rlats;
    // const View1D cnst_offline_icol[num_tracer_cnst];
    // const Forcing *forcings_in;
    // struct Forcing {
    //   // This index is in Fortran format. i.e. starts in 1
    //   int frc_ndx;
    //   bool file_alt_data;
    //   View1D fields_data[MAX_NUM_SECTIONS];
    //   int nsectors;
    // };

    // maybe correct--variables look similar or correspond to similar in fortran
    // =====================================
    // TODO: verify these are the proper shape
    const auto linoz_o3_clim_icol_ = input.get_array("linoz_o3_clim")[0];
    const auto linoz_t_clim_icol_ = input.get_array("linoz_t_clim")[0];
    const auto linoz_o3col_clim_icol_ = input.get_array("linoz_o3col_clim")[0];
    const auto linoz_PmL_clim_icol_ = input.get_array("linoz_PmL_clim")[0];
    const auto linoz_dPmL_dO3_icol_ = input.get_array("linoz_dPmL_dO3")[0];
    const auto linoz_dPmL_dT_icol_ = input.get_array("linoz_dPmL_dT")[0];
    const auto linoz_dPmL_dO3col_icol_ =
        input.get_array("linoz_dPmL_dO3col")[0];
    const auto linoz_cariolle_pscs_icol_ =
        input.get_array("linoz_cariolle_psc")[0];
    // from gas_phase_chemdr.F90:428
    const Real eccf = 1.0;
    const Real pblh = input.get_array("pblh")[0];
    const auto dry_diameter_icol_ = input.get_array("dgnum")[0];
    const auto wet_diameter_icol_ = input.get_array("dgnumwet")[0];
    const auto wetdens_icol_ = input.get_array("wetdens")[0];
    // from lin_strat_chem.F90:75
    const Real linoz_psc_T = 193.0;

    const Atmosphere atm = validation::create_atmosphere(nlev, pblh);
    mam4::Prognostics progs = validation::create_prognostics(nlev);
    const mam4::mo_setsox::Config config_setsox;
    mam4::microphysics::AmicPhysConfig config_amicphys;

    // Have:
    // =====================================
    // {"asdir", "calday", "cflx", "cldfr", "cldw", "cmfdqr", "delt", "dgnum",
    //  "dgnumwet", "drydepflx", "eccen", "extcnt", "fieldname_len", "fsds",
    //  "gas_pcnst", "imozart", "indexm", "inv_ndx_cnst_o3", "lambm0", "lchnk",
    //  "linoz_PmL_clim", "linoz_cariolle_psc", "linoz_dPmL_dO3",
    //  "linoz_dPmL_dO3col", "linoz_dPmL_dT", "linoz_o3_clim",
    //  "linoz_o3col_clim", "linoz_t_clim", "map2chm", "mvelpp", "nabscol",
    //  "ncldwtr", "ncol", "ndx_h2so4", "nevapr", "nfs", "o3_ndx", "obliqr",
    //  "pblh", "pcnst", "pcols", "pdel", "pdeldry", "phis", "phtcnt", "pi",
    //  "pint", "pmid", "prain", "precc", "precl", "ps", "pver", "qqcw",
    //  "qtend", "rga", "rxntot", "rxt_tag_cnt", "rxt_tag_map", "snowhland",
    //  "state_q", "synoz_ndx", "tfld", "troplev", "ts", "ufld", "vfld",
    //  "wetdens", "zi", "zm"}

    // Missing:
    // =====================================
    // zenith_angle_icol -> orbit::zenith
    // d_sfc_alb_dir_vis_icol -> asdir?
    // o3_col_dens_i -> linoz_o3col_clim?
    // photo_rates_icol ->
    // extfrc_icol ->
    // invariants_icol ->
    // work_photo_table_icol ->
    // adv_mass_kg_per_moles ->
    // clsmap_4 ->
    // permute_4 ->
    // offset_aerosol ->
    // o3_sfc ->
    // o3_tau ->
    // o3_lbl ->
    // chlorine_loading ->
    // =======================================
    // const Real zenith_angle_icol;
    // const Real d_sfc_alb_dir_vis_icol;
    // const View1D o3_col_dens_i;
    // const View2D photo_rates_icol;
    // const View2D extfrc_icol;
    // const View2D invariants_icol;
    // const View1D work_photo_table_icol;
    // const Real adv_mass_kg_per_moles[gas_pcnst];
    // const int (clsmap_4)[gas_pcnst];
    // const int (permute_4)[gas_pcnst];
    // const int offset_aerosol;
    // const Real o3_sfc;
    // const Real o3_tau;
    // const int o3_lbl;

    // Appears to come from file/other module:
    // =======================================
    // const Real chlorine_loading;

    const View1D linoz_o3_clim_icol;
    const View1D linoz_t_clim_icol;
    const View1D linoz_o3col_clim_icol;
    const View1D linoz_PmL_clim_icol;
    const View1D linoz_dPmL_dO3_icol;
    const View1D linoz_dPmL_dT_icol;
    const View1D linoz_dPmL_dO3col_icol;
    const View1D linoz_cariolle_pscs_icol;

    const ConstView2D dry_diameter_icol;
    const ConstView2D wet_diameter_icol;
    const ConstView2D wetdens_icol;

    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy,
        KOKKOS_LAMBDA(const ThreadTeam &team){
            // mam4::perform_atmospheric_chemistry_and_microphysics(
            //     team, dt, rlats, cnst_offline_icol, forcings_in, atm, progs,
            //     photo_table, chlorine_loading, config_setsox,
            //     config_amicphys,
            //     linoz_psc_T, zenith_angle_icol, d_sfc_alb_dir_vis_icol,
            //     o3_col_dens_i, photo_rates_icol, extfrc_icol,
            //     invariants_icol,
            //     work_photo_table_icol, linoz_o3_clim_icol, linoz_t_clim_icol,
            //     linoz_o3col_clim_icol, linoz_PmL_clim_icol,
            //     linoz_dPmL_dO3_icol,
            //     linoz_dPmL_dT_icol, linoz_dPmL_dO3col_icol,
            //     linoz_cariolle_pscs_icol,
            //     eccf, adv_mass_kg_per_moles, clsmap_4, permute_4,
            //     offset_aerosol,
            //     o3_sfc, o3_tau, o3_lbl, dry_diameter_icol, wet_diameter_icol,
            //     wetdens_icol);
        });

    std::vector<Real> cflx_out;
    // 2D view
    std::vector<Real> dgnum_out;
    // 2D view
    std::vector<Real> dgnumwet_out;
    std::vector<Real> drydepflx_out;
    std::vector<Real> qqcw_out;
    std::vector<Real> qtend_out;
    // 2D view
    std::vector<Real> wetdens_out;

    output.set("cflx", cflx_out);
    output.set("dgnum", dgnum_out);
    output.set("dgnumwet", dgnumwet_out);
    output.set("drydepflx", drydepflx_out);
    output.set("qqcw", qqcw_out);
    output.set("qtend", qtend_out);
    output.set("wetdens", wetdens_out);

    // NOTE: these are provided as outputs in the python module but do not
    // appear to be outputs
    // Real asdir_out;
    // Real fsds_out;
    // Real pblh_out;
    // Real precc_out;
    // Real precl_out;
    // Real snowhland_out;
    // Real ts_out;
    // output.set("asdir", asdir_out);
    // output.set("fsds", fsds_out);
    // output.set("pblh", pblh_out);
    // output.set("precc", precc_out);
    // output.set("precl", precl_out);
    // output.set("snowhland", snowhland_out);
    // output.set("ts", ts_out);
  });
}
