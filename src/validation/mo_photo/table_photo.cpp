// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;
using namespace haero;
using namespace mo_photo;

void table_photo(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename HostType::view_1d<Real>;
    using View1D = typename DeviceType::view_1d<Real>;
    using View2D = typename DeviceType::view_2d<Real>;
    using View3D = typename DeviceType::view_3d<Real>;

    const int ncol = 4;

    const auto sza_db = input.get_array("sza");
    const auto del_sza_db = input.get_array("del_sza");
    const auto alb_db = input.get_array("alb");
    const auto press_db = input.get_array("press");
    const auto del_p_db = input.get_array("del_p");
    const auto colo3_db = input.get_array("colo3");
    const auto o3rat_db = input.get_array("o3rat");
    const auto del_alb_db = input.get_array("del_alb");
    const auto del_o3rat_db = input.get_array("del_o3rat");
    const auto etfphot_db = input.get_array("etfphot");

    auto shape_rsf_tab = input.get_array("shape_rsf_tab");
    auto synthetic_values = input.get_array("synthetic_values_rsf_tab");
    auto shape_xsqy = input.get_array("shape_xsqy");

    int nw = int(shape_rsf_tab[0]);
    int nt = int(shape_xsqy[2]);
    int np_xs = int(shape_xsqy[3]);
    int numj = int(shape_xsqy[0]);
    int nump = int(shape_rsf_tab[1]);
    int numsza = int(shape_rsf_tab[2]);
    int numcolo3 = int(shape_rsf_tab[3]);
    int numalb = int(shape_rsf_tab[4]);

    PhotoTableData table_data = create_photo_table_data(
        nw, nt, np_xs, numj, nump, numsza, numcolo3, numalb);

    auto synthetic_values_xsqy = input.get_array("synthetic_values_xsqy");
    mam4::validation::create_synthetic_rsf_tab(
        table_data.rsf_tab, table_data.nw, table_data.nump, table_data.numsza,
        table_data.numcolo3, table_data.numalb, synthetic_values.data());

    auto sza_host = View1DHost((Real *)sza_db.data(), table_data.numsza);
    Kokkos::deep_copy(table_data.sza, sza_host);

    auto del_sza_host =
        View1DHost((Real *)del_sza_db.data(), table_data.numsza - 1);
    Kokkos::deep_copy(table_data.del_sza, del_sza_host);

    auto alb_host = View1DHost((Real *)alb_db.data(), table_data.numalb);
    Kokkos::deep_copy(table_data.alb, alb_host);

    auto press_host = View1DHost((Real *)press_db.data(), table_data.nump);
    Kokkos::deep_copy(table_data.press, press_host);

    auto del_p_host = View1DHost((Real *)del_p_db.data(), table_data.nump - 1);
    Kokkos::deep_copy(table_data.del_p, del_p_host);

    auto colo3_host = View1DHost((Real *)colo3_db.data(), table_data.nump);
    Kokkos::deep_copy(table_data.colo3, colo3_host);

    auto o3rat_host = View1DHost((Real *)o3rat_db.data(), table_data.numcolo3);
    Kokkos::deep_copy(table_data.o3rat, o3rat_host);

    auto del_alb_host =
        View1DHost((Real *)del_alb_db.data(), table_data.numalb - 1);
    Kokkos::deep_copy(table_data.del_alb, del_alb_host);

    auto del_o3rat_host =
        View1DHost((Real *)del_o3rat_db.data(), table_data.numcolo3 - 1);
    Kokkos::deep_copy(table_data.del_o3rat, del_o3rat_host);

    auto etfphot_host = View1DHost((Real *)etfphot_db.data(), table_data.nw);
    Kokkos::deep_copy(table_data.etfphot, etfphot_host);

    const auto prs_db = input.get_array("prs");
    const auto dprs_db = input.get_array("dprs");

    auto prs_host = View1DHost((Real *)prs_db.data(), table_data.np_xs);
    Kokkos::deep_copy(table_data.prs, prs_host);

    auto dprs_host = View1DHost((Real *)dprs_db.data(), table_data.np_xs - 1);
    Kokkos::deep_copy(table_data.dprs, dprs_host);

    View3D rsf("rsf", ncol, table_data.nw, pver);
    View3D xswk("xswk", ncol, table_data.numj, table_data.nw);

    const Real values_xsqy = synthetic_values_xsqy[0];
    Kokkos::deep_copy(table_data.xsqy, values_xsqy);

    View3D j_long("j_long", ncol, table_data.numj, pver);

    auto psum_l = View2D("psum_l", ncol, table_data.nw);
    auto psum_u = View2D("psum_u", ncol, table_data.nw);

    auto pht_alias_mult_1_db = input.get_array("pht_alias_mult");
    auto pht_alias_mult_1_host =
        View1DHost((Real *)pht_alias_mult_1_db.data(), 2);
    Kokkos::deep_copy(table_data.pht_alias_mult_1, pht_alias_mult_1_host);

    auto lng_indexer_db = input.get_array("lng_indexer");
    Kokkos::deep_copy(table_data.lng_indexer, lng_indexer_db[0] - 1);

    View3D photo("photo", ncol, pver, 1);

    const auto pmid_db = input.get_array("pmid");
    const auto pdel_db = input.get_array("pdel");
    const auto temper_db = input.get_array("temper");
    const auto colo3_in_db = input.get_array("col_dens_1");
    const auto lwc_db = input.get_array("lwc");
    const auto clouds_db = input.get_array("clouds");
    const auto srf_alb_db = input.get_array("srf_alb");
    const Real esfact = input.get_array("esfact")[0];
    const auto zen_angle_db = input.get_array("zen_angle");

    View2D pmid("pmid", ncol, pver);
    View2D pdel("pdel", ncol, pver);
    View2D temper("temper", ncol, pver);
    View2D colo3_in("colo3_in", ncol, pver);
    View2D lwc("lwc", ncol, pver);
    View2D clouds("clouds", ncol, pver);

    mam4::validation::convert_1d_vector_to_2d_view_device(pmid_db, pmid);
    mam4::validation::convert_1d_vector_to_2d_view_device(pdel_db, pdel);
    mam4::validation::convert_1d_vector_to_2d_view_device(temper_db, temper);
    mam4::validation::convert_1d_vector_to_2d_view_device(colo3_in_db,
                                                          colo3_in);
    mam4::validation::convert_1d_vector_to_2d_view_device(lwc_db, lwc);
    mam4::validation::convert_1d_vector_to_2d_view_device(clouds_db, clouds);

    auto srf_alb_host = View1DHost((Real *)srf_alb_db.data(), ncol);
    auto zen_angle_host = View1DHost((Real *)zen_angle_db.data(), ncol);

    View1D srf_alb("srf_alb", ncol);
    View1D zen_angle("zen_angle", ncol);
    Kokkos::deep_copy(srf_alb, srf_alb_host);
    Kokkos::deep_copy(zen_angle, zen_angle_host);

    auto team_policy = ThreadTeamPolicy(ncol, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          const int i = team.league_rank();
          auto photo_icol =
              Kokkos::subview(photo, i, Kokkos::ALL(), Kokkos::ALL());
          auto pmid_icol = Kokkos::subview(pmid, i, Kokkos::ALL());
          auto pdel_icol = Kokkos::subview(pdel, i, Kokkos::ALL());
          auto temper_icol = Kokkos::subview(temper, i, Kokkos::ALL());
          auto colo3_in_icol = Kokkos::subview(colo3_in, i, Kokkos::ALL());
          auto lwc_icol = Kokkos::subview(lwc, i, Kokkos::ALL());
          auto clouds_icol = Kokkos::subview(clouds, i, Kokkos::ALL());

          // set column-specific work arrays
          PhotoTableWorkArrays work_arrays{};
          work_arrays.lng_prates =
              Kokkos::subview(j_long, i, Kokkos::ALL(), Kokkos::ALL());
          work_arrays.rsf =
              Kokkos::subview(rsf, i, Kokkos::ALL(), Kokkos::ALL());
          work_arrays.xswk =
              Kokkos::subview(xswk, i, Kokkos::ALL(), Kokkos::ALL());
          work_arrays.psum_l = Kokkos::subview(psum_l, i, Kokkos::ALL());
          work_arrays.psum_u = Kokkos::subview(psum_u, i, Kokkos::ALL());

          table_photo(photo_icol, // out
                      pmid_icol, pdel_icol,
                      temper_icol, // in
                      colo3_in_icol, zen_angle(i), srf_alb(i), lwc_icol,
                      clouds_icol, // in
                      esfact, table_data, work_arrays);
        });

    auto photo_out_device =
        Kokkos::subview(photo, Kokkos::ALL(), Kokkos::ALL(), 0);
    const Real zero = 0;
    std::vector<Real> photo_out(pver * ncol, zero);
    mam4::validation::convert_2d_view_device_to_1d_vector(photo_out_device,
                                                          photo_out);
    output.set("photo", photo_out);
  });
}
