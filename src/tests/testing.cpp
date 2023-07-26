// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"

#include <haero/testing.hpp>

// the testing namespace contains functions that are useful only within tests,
// not to be used in production code
namespace mam4::testing {

Prognostics create_prognostics(int num_levels) {
  Prognostics p(num_levels);
  for (int mode = 0; mode < AeroConfig::num_modes(); ++mode) {
    p.n_mode_i[mode] = create_column_view(num_levels);
    p.n_mode_c[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(p.n_mode_i[mode], 0.0);
    Kokkos::deep_copy(p.n_mode_c[mode], 0.0);
    for (int spec = 0; spec < AeroConfig::num_aerosol_ids(); ++spec) {
      p.q_aero_i[mode][spec] = create_column_view(num_levels);
      p.q_aero_c[mode][spec] = create_column_view(num_levels);
      Kokkos::deep_copy(p.q_aero_i[mode][spec], 0.0);
      Kokkos::deep_copy(p.q_aero_c[mode][spec], 0.0);
    }
  }
  for (int gas = 0; gas < AeroConfig::num_gas_ids(); ++gas) {
    p.q_gas[gas] = create_column_view(num_levels);
    p.q_gas_avg[gas] = create_column_view(num_levels);
    Kokkos::deep_copy(p.q_gas[gas], 0.0);
    Kokkos::deep_copy(p.q_gas_avg[gas], 0.0);
    for (int mode = 0; mode < AeroConfig::num_modes(); ++mode) {
      p.uptkaer[gas][mode] = create_column_view(num_levels);
      Kokkos::deep_copy(p.uptkaer[gas][mode], 0.0);
    }
  }
  return p;
}

Diagnostics create_diagnostics(int num_levels) {
  Diagnostics d(num_levels);
  for (int mode = 0; mode < AeroConfig::num_modes(); ++mode) {
    d.hygroscopicity[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(d.hygroscopicity[mode], 0.0);
    d.dry_geometric_mean_diameter_i[mode] = create_column_view(num_levels);
    d.dry_geometric_mean_diameter_c[mode] = create_column_view(num_levels);
    d.dry_geometric_mean_diameter_total[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(d.dry_geometric_mean_diameter_i[mode], 0.0);
    Kokkos::deep_copy(d.dry_geometric_mean_diameter_c[mode], 0.0);
    Kokkos::deep_copy(d.dry_geometric_mean_diameter_total[mode], 0.0);
    d.wet_geometric_mean_diameter_i[mode] = create_column_view(num_levels);
    d.wet_geometric_mean_diameter_c[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(d.wet_geometric_mean_diameter_i[mode], 0.0);
    Kokkos::deep_copy(d.wet_geometric_mean_diameter_c[mode], 0.0);

    d.wet_density[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(d.wet_density[mode], 0.0);

    d.uptkrate_h2so4 = create_column_view(num_levels);
    Kokkos::deep_copy(d.uptkrate_h2so4, 0.0);

    d.g0_soa_out = create_column_view(num_levels);
    Kokkos::deep_copy(d.g0_soa_out, 0.0);

    d.is_cloudy =
        haero::DeviceType::view_1d<bool>("is_cloudy_bool", num_levels);
    Kokkos::deep_copy(d.is_cloudy, false);
    d.num_substeps =
        haero::DeviceType::view_1d<int>("num_substeps", num_levels);
    Kokkos::deep_copy(d.num_substeps, 0);

    d.icenuc_num_hetfrz = create_column_view(num_levels);
    d.icenuc_num_immfrz = create_column_view(num_levels);
    d.icenuc_num_depnuc = create_column_view(num_levels);
    d.icenuc_num_meydep = create_column_view(num_levels);
    Kokkos::deep_copy(d.icenuc_num_hetfrz, 0.0);
    Kokkos::deep_copy(d.icenuc_num_immfrz, 0.0);
    Kokkos::deep_copy(d.icenuc_num_depnuc, 0.0);
    Kokkos::deep_copy(d.icenuc_num_meydep, 0.0);

    d.num_act_aerosol_ice_nucle_hom = create_column_view(num_levels);
    d.num_act_aerosol_ice_nucle = create_column_view(num_levels);
    Kokkos::deep_copy(d.num_act_aerosol_ice_nucle_hom, 0.0);
    Kokkos::deep_copy(d.num_act_aerosol_ice_nucle, 0.0);

    d.stratiform_cloud_fraction = create_column_view(num_levels);
    Kokkos::deep_copy(d.stratiform_cloud_fraction, 0.0);
    d.activation_fraction[mode] = create_column_view(num_levels);
    Kokkos::deep_copy(d.activation_fraction[mode], 0.0);

    d.hetfrz_immersion_nucleation_tend = create_column_view(num_levels);
    Kokkos::deep_copy(d.hetfrz_immersion_nucleation_tend, 0.0);
    d.hetfrz_contact_nucleation_tend = create_column_view(num_levels);
    Kokkos::deep_copy(d.hetfrz_contact_nucleation_tend, 0.0);
    d.hetfrz_depostion_nucleation_tend = create_column_view(num_levels);
    Kokkos::deep_copy(d.hetfrz_depostion_nucleation_tend, 0.0);

    d.bc_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.bc_num, 0.0);
    d.dst1_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.dst1_num, 0.0);
    d.dst3_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.dst3_num, 0.0);
    d.bcc_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.bcc_num, 0.0);
    d.dst1c_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.dst1c_num, 0.0);
    d.dst3c_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.dst3c_num, 0.0);
    d.bcuc_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.bcuc_num, 0.0);
    d.dst1uc_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.dst1uc_num, 0.0);
    d.dst3uc_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.dst3uc_num, 0.0);
    d.bc_a1_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.bc_a1_num, 0.0);
    d.dst_a1_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.dst_a1_num, 0.0);
    d.dst_a3_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.dst_a3_num, 0.0);
    d.bc_c1_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.bc_c1_num, 0.0);
    d.dst_c1_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.dst_c1_num, 0.0);
    d.dst_c3_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.dst_c3_num, 0.0);
    d.fn_bc_c1_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.fn_bc_c1_num, 0.0);
    d.fn_dst_c1_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.fn_dst_c1_num, 0.0);
    d.fn_dst_c3_num = create_column_view(num_levels);
    Kokkos::deep_copy(d.fn_dst_c3_num, 0.0);
    d.na500 = create_column_view(num_levels);
    Kokkos::deep_copy(d.na500, 0.0);
    d.totna500 = create_column_view(num_levels);
    Kokkos::deep_copy(d.totna500, 0.0);
    d.freqimm = create_column_view(num_levels);
    Kokkos::deep_copy(d.freqimm, 0.0);
    d.freqcnt = create_column_view(num_levels);
    Kokkos::deep_copy(d.freqcnt, 0.0);
    d.freqdep = create_column_view(num_levels);
    Kokkos::deep_copy(d.freqdep, 0.0);
    d.freqmix = create_column_view(num_levels);
    Kokkos::deep_copy(d.freqmix, 0.0);
    d.dstfrezimm = create_column_view(num_levels);
    Kokkos::deep_copy(d.dstfrezimm, 0.0);
    d.dstfrezcnt = create_column_view(num_levels);
    Kokkos::deep_copy(d.dstfrezcnt, 0.0);
    d.dstfrezdep = create_column_view(num_levels);
    Kokkos::deep_copy(d.dstfrezdep, 0.0);
    d.bcfrezimm = create_column_view(num_levels);
    Kokkos::deep_copy(d.bcfrezimm, 0.0);
    d.bcfrezcnt = create_column_view(num_levels);
    Kokkos::deep_copy(d.bcfrezcnt, 0.0);
    d.bcfrezdep = create_column_view(num_levels);
    Kokkos::deep_copy(d.bcfrezdep, 0.0);
    d.nimix_imm = create_column_view(num_levels);
    Kokkos::deep_copy(d.nimix_imm, 0.0);
    d.nimix_cnt = create_column_view(num_levels);
    Kokkos::deep_copy(d.nimix_cnt, 0.0);
    d.nimix_dep = create_column_view(num_levels);
    Kokkos::deep_copy(d.nimix_dep, 0.0);
    d.dstnidep = create_column_view(num_levels);
    Kokkos::deep_copy(d.dstnidep, 0.0);
    d.dstnicnt = create_column_view(num_levels);
    Kokkos::deep_copy(d.dstnicnt, 0.0);
    d.dstniimm = create_column_view(num_levels);
    Kokkos::deep_copy(d.dstniimm, 0.0);
    d.bcnidep = create_column_view(num_levels);
    Kokkos::deep_copy(d.bcnidep, 0.0);
    d.bcnicnt = create_column_view(num_levels);
    Kokkos::deep_copy(d.bcnicnt, 0.0);
    d.bcniimm = create_column_view(num_levels);
    Kokkos::deep_copy(d.bcniimm, 0.0);
    d.numice10s = create_column_view(num_levels);
    Kokkos::deep_copy(d.numice10s, 0.0);
    d.numimm10sdst = create_column_view(num_levels);
    Kokkos::deep_copy(d.numimm10sdst, 0.0);
    d.numimm10sbc = create_column_view(num_levels);
    Kokkos::deep_copy(d.numimm10sbc, 0.0);
  }
  return d;
}

Tendencies create_tendencies(int num_levels) {
  return create_prognostics(num_levels);
}

} // namespace mam4::testing
