// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "Kokkos_Core.hpp"
#include <mam4xx/mam4.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;

void hetfrz_rates_1box(Ensemble *ensemble) {

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    if (!input.has("dt")) {
      std::cerr << "Required name: "
                << "dt" << std::endl;
      exit(1);
    }

    if (!input.has_array("ncnst")) {
      std::cerr << "Required name: "
                << "ncnst" << std::endl;
      exit(1);
    }

    if (!input.has_array("pi")) {
      std::cerr << "Required name: "
                << "pi" << std::endl;
      exit(1);
    }

    if (!input.has_array("rhoh2o")) {
      std::cerr << "Required name: "
                << "rhoh2o" << std::endl;
      exit(1);
    }

    if (!input.has_array("deltatin")) {
      std::cerr << "Required name: "
                << "deltatin" << std::endl;
      exit(1);
    }

    if (!input.has_array("rair")) {
      std::cerr << "Required name: "
                << "rair" << std::endl;
      exit(1);
    }

    if (!input.has_array("mincld")) {
      std::cerr << "Required name: "
                << "mincld" << std::endl;
      exit(1);
    }

    if (!input.has_array("temperature")) {
      std::cerr << "Required name: "
                << "temperature" << std::endl;
      exit(1);
    }

    if (!input.has_array("pmid")) {
      std::cerr << "Required name: "
                << "pmid" << std::endl;
      exit(1);
    }

    if (!input.has_array("ast")) {
      std::cerr << "Required name: "
                << "ast" << std::endl;
      exit(1);
    }

    if (!input.has_array("qc")) {
      std::cerr << "Required name: "
                << "qc" << std::endl;
      exit(1);
    }

    if (!input.has_array("nc")) {
      std::cerr << "Required name: "
                << "nc" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_bc_accum")) {
      std::cerr << "Required name: "
                << "state_q_bc_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_pom_accum")) {
      std::cerr << "Required name: "
                << "state_q_pom_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_soa_accum")) {
      std::cerr << "Required name: "
                << "state_q_soa_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_dust_accum")) {
      std::cerr << "Required name: "
                << "state_q_dust_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_nacl_accum")) {
      std::cerr << "Required name: "
                << "state_q_nacl_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_mom_accum")) {
      std::cerr << "Required name: "
                << "state_q_mom_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_num_accum")) {
      std::cerr << "Required name: "
                << "state_q_num_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_dust_coarse")) {
      std::cerr << "Required name: "
                << "state_q_dust_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_nacl_coarse")) {
      std::cerr << "Required name: "
                << "state_q_nacl_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_so4_coarse")) {
      std::cerr << "Required name: "
                << "state_q_so4_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_bc_coarse")) {
      std::cerr << "Required name: "
                << "state_q_bc_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_pom_coarse")) {
      std::cerr << "Required name: "
                << "state_q_pom_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_soa_coarse")) {
      std::cerr << "Required name: "
                << "state_q_soa_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_mom_coarse")) {
      std::cerr << "Required name: "
                << "state_q_mom_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_num_coarse")) {
      std::cerr << "Required name: "
                << "state_q_num_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_bc_pcarbon")) {
      std::cerr << "Required name: "
                << "state_q_bc_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_pom_pcarbon")) {
      std::cerr << "Required name: "
                << "state_q_pom_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_mom_pcarbon")) {
      std::cerr << "Required name: "
                << "state_q_mom_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("state_q_num_pcarbon")) {
      std::cerr << "Required name: "
                << "state_q_num_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_so4_accum")) {
      std::cerr << "Required name: "
                << "aer_cb_so4_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_bc_accum")) {
      std::cerr << "Required name: "
                << "aer_cb_bc_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_pom_accum")) {
      std::cerr << "Required name: "
                << "aer_cb_pom_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_soa_accum")) {
      std::cerr << "Required name: "
                << "aer_cb_soa_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_dst_accum")) {
      std::cerr << "Required name: "
                << "aer_cb_dst_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_ncl_accum")) {
      std::cerr << "Required name: "
                << "aer_cb_ncl_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_mom_accum")) {
      std::cerr << "Required name: "
                << "aer_cb_mom_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_num_accum")) {
      std::cerr << "Required name: "
                << "aer_cb_num_accum" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_dst_coarse")) {
      std::cerr << "Required name: "
                << "aer_cb_dst_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_ncl_coarse")) {
      std::cerr << "Required name: "
                << "aer_cb_ncl_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_so4_coarse")) {
      std::cerr << "Required name: "
                << "aer_cb_so4_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_bc_coarse")) {
      std::cerr << "Required name: "
                << "aer_cb_bc_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_pom_coarse")) {
      std::cerr << "Required name: "
                << "aer_cb_pom_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_soa_coarse")) {
      std::cerr << "Required name: "
                << "aer_cb_soa_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_mom_coarse")) {
      std::cerr << "Required name: "
                << "aer_cb_mom_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_num_coarse")) {
      std::cerr << "Required name: "
                << "aer_cb_num_coarse" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_bc_pcarbon")) {
      std::cerr << "Required name: "
                << "aer_cb_bc_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_pom_pcarbon")) {
      std::cerr << "Required name: "
                << "aer_cb_pom_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_mom_pcarbon")) {
      std::cerr << "Required name: "
                << "aer_cb_mom_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("aer_cb_num_pcarbon")) {
      std::cerr << "Required name: "
                << "aer_cb_num_pcarbon" << std::endl;
      exit(1);
    }

    if (!input.has_array("factnum")) {
      std::cerr << "Required name: "
                << "factnum" << std::endl;
      exit(1);
    }

    // auto dt = input.get("deltatin");
    auto ncnst = input.get_array("ncnst");
    auto pi = input.get_array("pi");
    auto rhoh2o = input.get_array("rhoh2o");
    auto deltatin = input.get_array("deltatin");
    auto dt = deltatin[0];
    auto rair = input.get_array("rair");
    auto mincld = input.get_array("mincld");
    auto temperature = input.get_array("temperature");
    auto pmid = input.get_array("pmid");
    auto ast = input.get_array("ast");
    auto qc = input.get_array("qc");
    auto nc = input.get_array("nc");
    auto state_q_bc_accum = input.get_array("state_q_bc_accum");
    auto state_q_pom_accum = input.get_array("state_q_pom_accum");
    auto state_q_soa_accum = input.get_array("state_q_soa_accum");
    auto state_q_dust_accum = input.get_array("state_q_dust_accum");
    auto state_q_nacl_accum = input.get_array("state_q_nacl_accum");
    auto state_q_mom_accum = input.get_array("state_q_mom_accum");
    auto state_q_num_accum = input.get_array("state_q_num_accum");
    auto state_q_dust_coarse = input.get_array("state_q_dust_coarse");
    auto state_q_nacl_coarse = input.get_array("state_q_nacl_coarse");
    auto state_q_so4_coarse = input.get_array("state_q_so4_coarse");
    auto state_q_bc_coarse = input.get_array("state_q_bc_coarse");
    auto state_q_pom_coarse = input.get_array("state_q_pom_coarse");
    auto state_q_soa_coarse = input.get_array("state_q_soa_coarse");
    auto state_q_mom_coarse = input.get_array("state_q_mom_coarse");
    auto state_q_num_coarse = input.get_array("state_q_num_coarse");
    auto state_q_bc_pcarbon = input.get_array("state_q_bc_pcarbon");
    auto state_q_pom_pcarbon = input.get_array("state_q_pom_pcarbon");
    auto state_q_mom_pcarbon = input.get_array("state_q_mom_pcarbon");
    auto state_q_num_pcarbon = input.get_array("state_q_num_pcarbon");
    auto aer_cb_so4_accum = input.get_array("aer_cb_so4_accum");
    auto aer_cb_bc_accum = input.get_array("aer_cb_bc_accum");
    auto aer_cb_pom_accum = input.get_array("aer_cb_pom_accum");
    auto aer_cb_soa_accum = input.get_array("aer_cb_soa_accum");
    auto aer_cb_dst_accum = input.get_array("aer_cb_dst_accum");
    auto aer_cb_ncl_accum = input.get_array("aer_cb_ncl_accum");
    auto aer_cb_mom_accum = input.get_array("aer_cb_mom_accum");
    auto aer_cb_num_accum = input.get_array("aer_cb_num_accum");
    auto aer_cb_dst_coarse = input.get_array("aer_cb_dst_coarse");
    auto aer_cb_ncl_coarse = input.get_array("aer_cb_ncl_coarse");
    auto aer_cb_so4_coarse = input.get_array("aer_cb_so4_coarse");
    auto aer_cb_bc_coarse = input.get_array("aer_cb_bc_coarse");
    auto aer_cb_pom_coarse = input.get_array("aer_cb_pom_coarse");
    auto aer_cb_soa_coarse = input.get_array("aer_cb_soa_coarse");
    auto aer_cb_mom_coarse = input.get_array("aer_cb_mom_coarse");
    auto aer_cb_num_coarse = input.get_array("aer_cb_num_coarse");
    auto aer_cb_bc_pcarbon = input.get_array("aer_cb_bc_pcarbon");
    auto aer_cb_pom_pcarbon = input.get_array("aer_cb_pom_pcarbon");
    auto aer_cb_mom_pcarbon = input.get_array("aer_cb_mom_pcarbon");
    auto aer_cb_num_pcarbon = input.get_array("aer_cb_num_pcarbon");
    auto factnum = input.get_array("factnum");

    const int nlev = qc.size();
    Real pblh = 1000;
    mam4::Prognostics progs = validation::create_prognostics(nlev);
    mam4::Diagnostics diags = validation::create_diagnostics(nlev);
    mam4::Tendencies tends = validation::create_tendencies(nlev);

    auto host_column = Kokkos::create_mirror_view(
        progs.n_mode_i[int(ModeIndex::Accumulation)]);

    // Copy data from input arrays to Kokkos views for aerosol number
    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_num_accum[k];
    }
    Kokkos::deep_copy(progs.n_mode_i[int(ModeIndex::Accumulation)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_num_coarse[k];
    }
    Kokkos::deep_copy(progs.n_mode_i[int(ModeIndex::Coarse)], host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_num_pcarbon[k];
    }
    Kokkos::deep_copy(progs.n_mode_i[int(ModeIndex::PrimaryCarbon)],
                      host_column);

    // Copy data from input arrays to Kokkos views for aerosol mass
    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_bc_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_i[int(ModeIndex::Accumulation)][int(AeroId::BC)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_pom_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_i[int(ModeIndex::Accumulation)][int(AeroId::POM)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_soa_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_i[int(ModeIndex::Accumulation)][int(AeroId::SOA)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_dust_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_i[int(ModeIndex::Accumulation)][int(AeroId::DST)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_nacl_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_i[int(ModeIndex::Accumulation)][int(AeroId::NaCl)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_mom_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_i[int(ModeIndex::Accumulation)][int(AeroId::MOM)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_bc_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_i[int(ModeIndex::Coarse)][int(AeroId::BC)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_pom_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_i[int(ModeIndex::Coarse)][int(AeroId::POM)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_so4_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_i[int(ModeIndex::Coarse)][int(AeroId::SO4)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_dust_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_i[int(ModeIndex::Coarse)][int(AeroId::DST)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_nacl_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_i[int(ModeIndex::Coarse)][int(AeroId::NaCl)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_mom_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_i[int(ModeIndex::Coarse)][int(AeroId::MOM)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_bc_pcarbon[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_i[int(ModeIndex::PrimaryCarbon)][int(AeroId::BC)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_pom_pcarbon[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_i[int(ModeIndex::PrimaryCarbon)][int(AeroId::POM)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = state_q_mom_pcarbon[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_i[int(ModeIndex::PrimaryCarbon)][int(AeroId::MOM)],
        host_column);

    // Copy cloudborne aerosol numbers into host_column and copy to Kokkos view
    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_num_accum[k];
    }
    Kokkos::deep_copy(progs.n_mode_c[int(ModeIndex::Accumulation)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_num_coarse[k];
    }
    Kokkos::deep_copy(progs.n_mode_c[int(ModeIndex::Coarse)], host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_num_pcarbon[k];
    }
    Kokkos::deep_copy(progs.n_mode_c[int(ModeIndex::PrimaryCarbon)],
                      host_column);

    // Copy cloudborne aerosol mass into host_column and copy to Kokkos view
    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_so4_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_c[int(ModeIndex::Accumulation)][int(AeroId::SO4)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_dst_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_c[int(ModeIndex::Accumulation)][int(AeroId::DST)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_ncl_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_c[int(ModeIndex::Accumulation)][int(AeroId::NaCl)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_bc_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_c[int(ModeIndex::Accumulation)][int(AeroId::BC)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_pom_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_c[int(ModeIndex::Accumulation)][int(AeroId::POM)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_mom_accum[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_c[int(ModeIndex::Accumulation)][int(AeroId::MOM)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_so4_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_c[int(ModeIndex::Coarse)][int(AeroId::SO4)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_dst_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_c[int(ModeIndex::Coarse)][int(AeroId::DST)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_ncl_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_c[int(ModeIndex::Coarse)][int(AeroId::NaCl)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_bc_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_c[int(ModeIndex::Coarse)][int(AeroId::BC)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_pom_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_c[int(ModeIndex::Coarse)][int(AeroId::POM)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_mom_coarse[k];
    }
    Kokkos::deep_copy(progs.q_aero_c[int(ModeIndex::Coarse)][int(AeroId::MOM)],
                      host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_bc_pcarbon[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_c[int(ModeIndex::PrimaryCarbon)][int(AeroId::BC)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_pom_pcarbon[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_c[int(ModeIndex::PrimaryCarbon)][int(AeroId::POM)],
        host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = aer_cb_mom_pcarbon[k];
    }
    Kokkos::deep_copy(
        progs.q_aero_c[int(ModeIndex::PrimaryCarbon)][int(AeroId::MOM)],
        host_column);

    // Copy data from input arrays to Kokkos views for temperature, pressure,
    // mixing-ratios
    for (int k = 0; k < nlev; ++k) {
      host_column(k) = temperature[k];
    }
    auto d_temperature = validation::create_column_view(nlev);
    Kokkos::deep_copy(d_temperature, host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = pmid[k];
    }
    auto d_pressure = validation::create_column_view(nlev);
    Kokkos::deep_copy(d_pressure, host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = qc[k];
    }
    auto d_liquid_mixing_ratio = validation::create_column_view(nlev);
    Kokkos::deep_copy(d_liquid_mixing_ratio, host_column);

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = nc[k];
    }
    auto d_cloud_liquid_number_mixing_ratio =
        validation::create_column_view(nlev);
    Kokkos::deep_copy(d_cloud_liquid_number_mixing_ratio, host_column);

    // Because the Atmosphere type uses ConstColumnViews, we have to construct
    // an atm object with views for all its data or set them piecemeal.
    auto d_vapor_mixing_ratio = validation::create_column_view(nlev);
    auto d_ice_mixing_ratio = validation::create_column_view(nlev);
    auto d_cloud_ice_number_mixing_ratio = validation::create_column_view(nlev);
    auto d_height = validation::create_column_view(nlev);
    auto d_hydrostatic_dp = validation::create_column_view(nlev);
    auto d_cloud_fraction = validation::create_column_view(nlev);
    auto d_updraft_vel_ice_nucleation = validation::create_column_view(nlev);
    Atmosphere atm(nlev, d_temperature, d_pressure, d_vapor_mixing_ratio,
                   d_liquid_mixing_ratio, d_cloud_liquid_number_mixing_ratio,
                   d_ice_mixing_ratio, d_cloud_ice_number_mixing_ratio,
                   d_height, d_hydrostatic_dp, d_cloud_fraction,
                   d_updraft_vel_ice_nucleation, pblh);
    Surface sfc = mam4::testing::create_surface();

    for (int k = 0; k < nlev; ++k) {
      host_column(k) = ast[k];
    }
    Kokkos::deep_copy(diags.stratiform_cloud_fraction, host_column);

    // Now need to unpack factnum and copy to device
    const int num_modes = AeroConfig::num_modes();
    for (int imode = 0; imode < num_modes; ++imode) {
      for (int k = 0; k < nlev; ++k) {
        ;
        host_column(k) = factnum[k + imode * nlev];
      }
      Kokkos::deep_copy(diags.activation_fraction[imode], host_column);
    }

    mam4::AeroConfig mam4_config;
    mam4::HetfrzProcess process(mam4_config);
    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Real t = 0.0;
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          process.compute_tendencies(team, t, dt, atm, sfc, progs, diags,
                                     tends);
        });

    // Copy tendencies hetfrz_immersion_nucleation_tend to host and write to
    // output
    auto host_tend_column =
        Kokkos::create_mirror_view(diags.hetfrz_immersion_nucleation_tend);
    Kokkos::deep_copy(host_tend_column, diags.hetfrz_immersion_nucleation_tend);
    output.set("frzimm", std::vector<Real>(host_tend_column.data(),
                                           host_tend_column.data() +
                                               host_tend_column.extent_int(0)));

    // Copy tendencies hetfrz_contact_nucleation_tend to host and write to
    // output
    Kokkos::deep_copy(host_tend_column, diags.hetfrz_contact_nucleation_tend);
    output.set("frzcnt", std::vector<Real>(host_tend_column.data(),
                                           host_tend_column.data() +
                                               host_tend_column.extent_int(0)));

    // Copy tendencies hetfrz_deposition_sublimation_tend to host and write to
    // output
    Kokkos::deep_copy(host_tend_column, diags.hetfrz_depostion_nucleation_tend);
    output.set("frzdep", std::vector<Real>(host_tend_column.data(),
                                           host_tend_column.data() +
                                               host_tend_column.extent_int(0)));
  });
}
