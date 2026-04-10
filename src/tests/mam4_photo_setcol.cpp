// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "atmosphere_utils.hpp"
#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <ekat_logger.hpp>

// Precision-dependent tolerance traits
template <typename T> struct PrecisionTolerance;

template <> struct PrecisionTolerance<float> {
  static constexpr float tol = 1e-5f; // Single precision tolerance
};

template <> struct PrecisionTolerance<double> {
  static constexpr double tol = 1e-12; // Double precision tolerance
};

using mam4::Real;

// Test compute_o3_column_density: serial reference vs parallel implementation
TEST_CASE("compute_o3_column_density", "mo_photo") {

  ekat::Comm comm;

  ekat::logger::Logger<> logger("compute_o3_column_density tests",
                                ekat::logger::LogLevel::debug, comm);
  using View1D = mam4::DeviceType::view_1d<Real>;
  using View1DHost = typename mam4::HostType::view_1d<Real>;

  // Initialize random number generator for reproducibility
  std::default_random_engine generator(98765);
  std::uniform_real_distribution<double> unif_dist(1e-8, 1e-5);

  constexpr int pver = mam4::nlev;
  constexpr Real tol = PrecisionTolerance<Real>::tol;

  // Test inputs
  const Real o3_col_deltas_0 =
      1.6e15;                    // Top-of-atmosphere column density [1/cm²]
  const Real mw_o3 = 48.0 / 1e3; // Ozone molecular weight [kg/mol]
  constexpr Real xfactor = 2.8704e21 / (9.80616 * 1.38044); // From set_ub_col

  // Atmospheric profile parameters (humid atmosphere with RH ~32%-98%)
  Real pblh = 1000;         // Planetary boundary layer height [m]
  const Real Tv0 = 300;     // Reference virtual temperature [K]
  const Real Gammav = 0.01; // Virtual temperature lapse rate [K/m]
  const Real qv0 = 0.015;   // Surface specific humidity [kg/kg]
  const Real qv1 = 7.5e-4;  // Specific humidity lapse rate [1/m]
  mam4::Atmosphere atm =
      mam4::init_atm_const_tv_lapse_rate(pver, pblh, Tv0, Gammav, qv0, qv1);

  // Create host views for inputs
  View1DHost pdel_host("pdel_host", pver);     // Pressure thickness [Pa]
  View1DHost mmr_o3_host("mmr_o3_host", pver); // Mass mixing ratio [kg/kg]
  View1DHost o3_col_dens_ref("o3_col_dens_ref", pver); // Reference output

  const auto pdel = atm.hydrostatic_dp; // Pressure thickness array [Pa]
  Kokkos::deep_copy(pdel_host, pdel);
  // Initialize random input data
  for (int k = 0; k < pver; ++k) {
    mmr_o3_host(k) = unif_dist(generator);
  }

  // ===== SERIAL REFERENCE IMPLEMENTATION =====
  // Step 1: Compute column density deltas for each level
  View1DHost o3_col_deltas_host("o3_col_deltas_host", pver + 1);
  o3_col_deltas_host(0) = o3_col_deltas_0;

  Real running_sum = 0.0;
  for (int kk = 0; kk < mam4::nlev; ++kk) {
    const Real vmr_o3_kk =
        mam4::conversions::vmr_from_mmr(mmr_o3_host(kk), mw_o3);
    const Real delta_kk = xfactor * pdel_host(kk) * vmr_o3_kk;

    o3_col_dens_ref(kk) = o3_col_deltas_0 + running_sum + 0.5 * delta_kk;

    running_sum += delta_kk;
  }

  // ===== PARALLEL IMPLEMENTATION USING compute_o3_column_density =====
  // Copy inputs to device
  View1D mmr_o3("mmr_o3", pver);
  View1D o3_col_dens("o3_col_dens", pver);
  Kokkos::deep_copy(mmr_o3, mmr_o3_host);

  auto team_policy = mam4::ThreadTeamPolicy(1, Kokkos::AUTO);
  Kokkos::parallel_for(
      "compute_o3_column", team_policy,
      KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
        mam4::microphysics::compute_o3_column_density(
            team,
            pdel,            // Pressure thickness array [nlev]
            mmr_o3,          // Ozone mass mixing ratio [nlev]
            o3_col_deltas_0, // Top-of-atmosphere column density [1/cm²]
            mw_o3,           // Ozone molecular weight [kg/mol]
            o3_col_dens      // Output: column density [1/cm²] [nlev]
        );
      });
  Kokkos::fence();

  // Copy results back to host for comparison
  auto o3_col_dens_host = Kokkos::create_mirror_view(o3_col_dens);
  Kokkos::deep_copy(o3_col_dens_host, o3_col_dens);

  // Compare serial reference with parallel implementation

  for (int k = 0; k < pver; ++k) {
    const Real diff =
        std::abs(o3_col_dens_host(k) - o3_col_dens_ref(k)) / o3_col_dens_ref(k);
    if (diff >= tol) {
      std::ostringstream ss;
      ss << "diff : [ ";
      ss << "Level " << k << ": diff = " << diff
         << ", parallel = " << o3_col_dens_host(k)
         << ", serial = " << o3_col_dens_ref(k) << "\n";
      ss << "]";
      logger.debug(ss.str());
    }
    REQUIRE(diff <= tol);
  }
}

// ============================================================================
// Helpers shared by interpolate_rsf and jlong BFB tests
// ============================================================================

// Bring mam4 types into scope for the sections below.
using namespace mam4;

// Table dimensions used by the two tests below.
// Using compile-time constants allows fixed-size stack arrays inside
// KOKKOS_LAMBDA / Kokkos::single device kernels.
constexpr int test_nw       = 4;  // wavelengths
constexpr int test_numj     = 1;  // photo-reactions (matches phtcnt)
constexpr int test_nump     = 5;  // RSF pressure levels
constexpr int test_numsza   = 3;  // zenith angles
constexpr int test_numcolo3 = 4;  // O3 column levels
constexpr int test_numalb   = 3;  // albedo levels
constexpr int test_nt       = 201; // temperatures (matches t_index range [0,200])
constexpr int test_np_xs    = 4;  // xsection pressure levels

// Solar zenith angle [degrees] – well within (0, 88.85) daylight range.
constexpr Real test_sza_in = 30.0;

// Build a small PhotoTableData with synthetic monotone lookup arrays and
// random table values.  All dimensions are the test_* constants above.
static mo_photo::PhotoTableData build_test_photo_table() {
  auto photo_table = mo_photo::create_photo_table_data(
      test_nw, test_nt, test_np_xs, test_numj,
      test_nump, test_numsza, test_numcolo3, test_numalb);

  std::default_random_engine gen(54321);
  std::uniform_real_distribution<Real> pos(1e-6, 1.0);

  // sza [degrees]: monotone increasing
  auto sza_h = Kokkos::create_mirror_view(photo_table.sza);
  sza_h(0) = 0.0;  sza_h(1) = 45.0; sza_h(2) = 88.0;
  Kokkos::deep_copy(photo_table.sza, sza_h);

  // del_sza = 1 / (sza[i+1] - sza[i])
  auto del_sza_h = Kokkos::create_mirror_view(photo_table.del_sza);
  for (int i = 0; i < test_numsza - 1; ++i)
    del_sza_h(i) = 1.0 / (sza_h(i + 1) - sza_h(i));
  Kokkos::deep_copy(photo_table.del_sza, del_sza_h);

  // alb [fraction]: monotone increasing
  auto alb_h = Kokkos::create_mirror_view(photo_table.alb);
  alb_h(0) = 0.0;  alb_h(1) = 0.3;  alb_h(2) = 0.8;
  Kokkos::deep_copy(photo_table.alb, alb_h);

  // del_alb = 1 / (alb[i+1] - alb[i])
  auto del_alb_h = Kokkos::create_mirror_view(photo_table.del_alb);
  for (int i = 0; i < test_numalb - 1; ++i)
    del_alb_h(i) = 1.0 / (alb_h(i + 1) - alb_h(i));
  Kokkos::deep_copy(photo_table.del_alb, del_alb_h);

  // press [hPa]: monotone DECREASING (surface first)
  auto press_h = Kokkos::create_mirror_view(photo_table.press);
  press_h(0) = 1000.0; press_h(1) = 500.0; press_h(2) = 100.0;
  press_h(3) =   50.0; press_h(4) =  10.0;
  Kokkos::deep_copy(photo_table.press, press_h);

  // del_p = 1 / (press[i] - press[i+1])  (size nump-1)
  auto del_p_h = Kokkos::create_mirror_view(photo_table.del_p);
  for (int i = 0; i < test_nump - 1; ++i)
    del_p_h(i) = 1.0 / (press_h(i) - press_h(i + 1));
  Kokkos::deep_copy(photo_table.del_p, del_p_h);

  // colo3 [molecules/cm^2]: decreasing with altitude
  auto colo3_h = Kokkos::create_mirror_view(photo_table.colo3);
  colo3_h(0) = 5e17; colo3_h(1) = 4e17; colo3_h(2) = 3e17;
  colo3_h(3) = 2e17; colo3_h(4) = 1e17;
  Kokkos::deep_copy(photo_table.colo3, colo3_h);

  // o3rat: monotone increasing
  auto o3rat_h = Kokkos::create_mirror_view(photo_table.o3rat);
  o3rat_h(0) = 0.5;  o3rat_h(1) = 1.0;  o3rat_h(2) = 1.5;  o3rat_h(3) = 2.0;
  Kokkos::deep_copy(photo_table.o3rat, o3rat_h);

  // del_o3rat = 1 / (o3rat[i+1] - o3rat[i])
  auto del_o3rat_h = Kokkos::create_mirror_view(photo_table.del_o3rat);
  for (int i = 0; i < test_numcolo3 - 1; ++i)
    del_o3rat_h(i) = 1.0 / (o3rat_h(i + 1) - o3rat_h(i));
  Kokkos::deep_copy(photo_table.del_o3rat, del_o3rat_h);

  // etfphot: positive random, size nw
  auto etfphot_h = Kokkos::create_mirror_view(photo_table.etfphot);
  for (int w = 0; w < test_nw; ++w)
    etfphot_h(w) = 1e13 * (0.8 + pos(gen));
  Kokkos::deep_copy(photo_table.etfphot, etfphot_h);

  // rsf_tab(nw, nump, numsza, numcolo3, numalb): random positive
  auto rsf_tab_h = Kokkos::create_mirror_view(photo_table.rsf_tab);
  for (int w  = 0; w  < test_nw;       ++w)
  for (int ip = 0; ip < test_nump;     ++ip)
  for (int is = 0; is < test_numsza;   ++is)
  for (int iv = 0; iv < test_numcolo3; ++iv)
  for (int ia = 0; ia < test_numalb;   ++ia)
    rsf_tab_h(w, ip, is, iv, ia) = pos(gen);
  Kokkos::deep_copy(photo_table.rsf_tab, rsf_tab_h);

  // prs [hPa]: monotone DECREASING
  auto prs_h = Kokkos::create_mirror_view(photo_table.prs);
  prs_h(0) = 500.0; prs_h(1) = 100.0; prs_h(2) = 10.0; prs_h(3) = 1.0;
  Kokkos::deep_copy(photo_table.prs, prs_h);

  // dprs = 1 / (prs[i] - prs[i+1])  (size np_xs-1)
  auto dprs_h = Kokkos::create_mirror_view(photo_table.dprs);
  for (int i = 0; i < test_np_xs - 1; ++i)
    dprs_h(i) = 1.0 / (prs_h(i) - prs_h(i + 1));
  Kokkos::deep_copy(photo_table.dprs, dprs_h);

  // pht_alias_mult_1: size 2
  auto pam_h = Kokkos::create_mirror_view(photo_table.pht_alias_mult_1);
  pam_h(0) = 1.0;  pam_h(1) = 0.0;
  Kokkos::deep_copy(photo_table.pht_alias_mult_1, pam_h);

  // lng_indexer: size 1; index 0 means "use first reaction"
  auto li_h = Kokkos::create_mirror_view(photo_table.lng_indexer);
  li_h(0) = 0;
  Kokkos::deep_copy(photo_table.lng_indexer, li_h);

  // xsqy(numj, nw, nt, np_xs): random positive
  auto xsqy_h = Kokkos::create_mirror_view(photo_table.xsqy);
  for (int j  = 0; j  < test_numj;   ++j)
  for (int w  = 0; w  < test_nw;     ++w)
  for (int it = 0; it < test_nt;     ++it)
  for (int ip = 0; ip < test_np_xs;  ++ip)
    xsqy_h(j, w, it, ip) = pos(gen);
  Kokkos::deep_copy(photo_table.xsqy, xsqy_h);

  return photo_table;
}

// ============================================================================
// Test 1: interpolate_rsf — serial Kokkos::single reference vs parallel
// ============================================================================
TEST_CASE("interpolate_rsf_serial_vs_parallel", "mo_photo") {

  ekat::Comm comm;
  ekat::logger::Logger<> logger("interpolate_rsf tests",
                                ekat::logger::LogLevel::debug, comm);

  constexpr int pver = nlev;
  constexpr Real Pa2mb = 1e-2; // Pa -> hPa

  auto photo_table = build_test_photo_table();

  // Atmosphere profile for pressure and temperature
  Atmosphere atm =
      init_atm_const_tv_lapse_rate(pver, 1000.0, 300.0, 0.01, 0.015, 7.5e-4);

  using View1D    = mo_photo::View1D;
  using View2D    = mo_photo::View2D;
  using View1DHost = Kokkos::View<Real*, Kokkos::HostSpace>;

  // Build alb_in, p_in, colo3_in (device views)
  View1DHost alb_in_host("alb_in_h", pver);
  View1DHost p_in_host("p_in_h", pver);
  View1DHost colo3_in_host("colo3_in_h", pver);

  auto pressure_host = Kokkos::create_mirror_view(atm.pressure);
  Kokkos::deep_copy(pressure_host, atm.pressure);
  for (int k = 0; k < pver; ++k) {
    alb_in_host(k)   = 0.15;
    p_in_host(k)     = pressure_host(k) * Pa2mb;
    colo3_in_host(k) = 3e17;
  }
  View1D alb_in_d("alb_in", pver);
  View1D p_in_d("p_in", pver);
  View1D colo3_in_d("colo3_in", pver);
  Kokkos::deep_copy(alb_in_d, alb_in_host);
  Kokkos::deep_copy(p_in_d, p_in_host);
  Kokkos::deep_copy(colo3_in_d, colo3_in_host);

  // ===== SERIAL REFERENCE =====
  // Reproduces the original Kokkos::single body of interpolate_rsf, including
  // shared 1D psum_l/psum_u scratch and the izl warm-start.
  View2D rsf_serial("rsf_serial", test_nw, pver);
  Kokkos::deep_copy(rsf_serial, 0.0);
  {
    const auto sza_d       = photo_table.sza;
    const auto del_sza_d   = photo_table.del_sza;
    const auto alb_d       = photo_table.alb;
    const auto press_d     = photo_table.press;
    const auto del_p_d     = photo_table.del_p;
    const auto colo3_d     = photo_table.colo3;
    const auto o3rat_d     = photo_table.o3rat;
    const auto del_alb_d   = photo_table.del_alb;
    const auto del_o3rat_d = photo_table.del_o3rat;
    const auto etfphot_d   = photo_table.etfphot;
    const auto rsf_tab_d   = photo_table.rsf_tab;

    Kokkos::parallel_for(
        "interp_rsf_serial_ref", ThreadTeamPolicy(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const ThreadTeam &team) {
      constexpr int nw_c       = test_nw;
      constexpr int pver_c     = nlev;
      constexpr int nump_c     = test_nump;
      constexpr int numsza_c   = test_numsza;
      constexpr int numcolo3_c = test_numcolo3;
      constexpr int numalb_c   = test_numalb;

      Kokkos::single(Kokkos::PerTeam(team), [=]() {
        const Real one = 1, zero = 0;

        int is = 0;
        mo_photo::find_index(sza_d, numsza_c, test_sza_in, is);

        Real dels[3] = {};
        dels[0] = utils::min_max_bound(
            zero, one, (test_sza_in - sza_d[is]) * del_sza_d[is]);
        const Real wrk0 = one - dels[0];

        // Shared 1D scratch (original behavior)
        Real psum_l_s[nw_c] = {};
        Real psum_u_s[nw_c] = {};

        int izl = 2; // warm-start (original serial code)
        for (int kk = pver_c - 1; kk >= 0; --kk) {
          int albind = 0;
          mo_photo::find_index(alb_d, numalb_c, alb_in_d[kk], albind);

          int pind = 0;
          Real wght1 = 0;
          if (p_in_d[kk] > press_d[0]) {
            pind = 1; wght1 = one;
          } else if (p_in_d[kk] <= press_d[nump_c - 1]) {
            pind = nump_c - 1; wght1 = zero;
          } else {
            int iz = 0;
            for (iz = izl - 1; iz < nump_c; ++iz) {
              if (press_d[iz] < p_in_d[kk]) { izl = iz; break; }
            }
            int t1 = iz < nump_c - 1 ? iz : nump_c - 1;
            pind  = t1 > 1 ? t1 : 1;
            wght1 = utils::min_max_bound(
                zero, one,
                (p_in_d[kk] - press_d[pind]) * del_p_d[pind - 1]);
          }

          const Real v3ratu = colo3_in_d[kk] / colo3_d[pind - 1];
          int ratindu = 0;
          mo_photo::find_index(o3rat_d, numcolo3_c, v3ratu, ratindu);

          Real v3ratl = zero;
          int ratindl = 0;
          if (colo3_d[pind] != zero) {
            v3ratl = colo3_in_d[kk] / colo3_d[pind];
            mo_photo::find_index(o3rat_d, numcolo3_c, v3ratl, ratindl);
          } else {
            ratindl = ratindu;
            v3ratl  = o3rat_d[ratindu];
          }

          int ial = albind;
          dels[2] = utils::min_max_bound(
              zero, one, (alb_in_d[kk] - alb_d[ial]) * del_alb_d[ial]);

          // calc_sum_wght for psum_l
          {
            int iv = ratindl;
            dels[1] = utils::min_max_bound(
                zero, one, (v3ratl - o3rat_d[iv]) * del_o3rat_d[iv]);
            const int isp1 = is + 1, ivp1 = iv + 1, ialp1 = ial + 1;
            Real wk = (one - dels[1]) * (one - dels[2]);
            const Real w000 = wrk0*wk, w100 = dels[0]*wk;
            wk = (one - dels[1]) * dels[2];
            const Real w001 = wrk0*wk, w101 = dels[0]*wk;
            wk = dels[1] * (one - dels[2]);
            const Real w010 = wrk0*wk, w110 = dels[0]*wk;
            wk = dels[1] * dels[2];
            const Real w011 = wrk0*wk, w111 = dels[0]*wk;
            for (int wn = 0; wn < nw_c; ++wn)
              psum_l_s[wn] =
                w000*rsf_tab_d(wn,pind,is,iv, ial) +
                w001*rsf_tab_d(wn,pind,is,iv, ialp1) +
                w010*rsf_tab_d(wn,pind,is,ivp1,ial) +
                w011*rsf_tab_d(wn,pind,is,ivp1,ialp1) +
                w100*rsf_tab_d(wn,pind,isp1,iv, ial) +
                w101*rsf_tab_d(wn,pind,isp1,iv, ialp1) +
                w110*rsf_tab_d(wn,pind,isp1,ivp1,ial) +
                w111*rsf_tab_d(wn,pind,isp1,ivp1,ialp1);
          }

          // calc_sum_wght for psum_u
          {
            int iv = ratindu;
            dels[1] = utils::min_max_bound(
                zero, one, (v3ratu - o3rat_d[iv]) * del_o3rat_d[iv]);
            const int iz_u = pind - 1;
            const int isp1 = is + 1, ivp1 = iv + 1, ialp1 = ial + 1;
            Real wk = (one - dels[1]) * (one - dels[2]);
            const Real w000 = wrk0*wk, w100 = dels[0]*wk;
            wk = (one - dels[1]) * dels[2];
            const Real w001 = wrk0*wk, w101 = dels[0]*wk;
            wk = dels[1] * (one - dels[2]);
            const Real w010 = wrk0*wk, w110 = dels[0]*wk;
            wk = dels[1] * dels[2];
            const Real w011 = wrk0*wk, w111 = dels[0]*wk;
            for (int wn = 0; wn < nw_c; ++wn)
              psum_u_s[wn] =
                w000*rsf_tab_d(wn,iz_u,is,iv, ial) +
                w001*rsf_tab_d(wn,iz_u,is,iv, ialp1) +
                w010*rsf_tab_d(wn,iz_u,is,ivp1,ial) +
                w011*rsf_tab_d(wn,iz_u,is,ivp1,ialp1) +
                w100*rsf_tab_d(wn,iz_u,isp1,iv, ial) +
                w101*rsf_tab_d(wn,iz_u,isp1,iv, ialp1) +
                w110*rsf_tab_d(wn,iz_u,isp1,ivp1,ial) +
                w111*rsf_tab_d(wn,iz_u,isp1,ivp1,ialp1);
          }

          for (int wn = 0; wn < nw_c; ++wn)
            rsf_serial(wn, kk) =
                (psum_l_s[wn] + wght1 * (psum_u_s[wn] - psum_l_s[wn]))
                * etfphot_d[wn];
        } // kk
      }); // Kokkos::single
    }); // parallel_for
    Kokkos::fence();
  }

  // ===== PARALLEL IMPLEMENTATION =====
  View2D rsf_parallel("rsf_parallel", test_nw, pver);
  Kokkos::deep_copy(rsf_parallel, 0.0);
  View2D psum_l_par("psum_l_par", pver, test_nw);
  View2D psum_u_par("psum_u_par", pver, test_nw);
  {
    const auto sza_d       = photo_table.sza;
    const auto del_sza_d   = photo_table.del_sza;
    const auto alb_d       = photo_table.alb;
    const auto press_d     = photo_table.press;
    const auto del_p_d     = photo_table.del_p;
    const auto colo3_d     = photo_table.colo3;
    const auto o3rat_d     = photo_table.o3rat;
    const auto del_alb_d   = photo_table.del_alb;
    const auto del_o3rat_d = photo_table.del_o3rat;
    const auto etfphot_d   = photo_table.etfphot;
    const auto rsf_tab_d   = photo_table.rsf_tab;

    Kokkos::parallel_for(
        "interp_rsf_parallel", ThreadTeamPolicy(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const ThreadTeam &team) {
      mo_photo::interpolate_rsf(
          team, alb_in_d, test_sza_in, p_in_d, colo3_in_d, nlev,
          sza_d, del_sza_d, alb_d, press_d, del_p_d, colo3_d,
          o3rat_d, del_alb_d, del_o3rat_d, etfphot_d, rsf_tab_d,
          test_nw, test_nump, test_numsza, test_numcolo3, test_numalb,
          rsf_parallel, psum_l_par, psum_u_par);
    });
    Kokkos::fence();
  }

  // ===== COMPARE (exact BFB equality) =====
  auto rsf_serial_h   = Kokkos::create_mirror_view(rsf_serial);
  auto rsf_parallel_h = Kokkos::create_mirror_view(rsf_parallel);
  Kokkos::deep_copy(rsf_serial_h,   rsf_serial);
  Kokkos::deep_copy(rsf_parallel_h, rsf_parallel);

  for (int wn = 0; wn < test_nw; ++wn) {
    for (int k = 0; k < pver; ++k) {
      if (rsf_parallel_h(wn, k) != rsf_serial_h(wn, k)) {
        std::ostringstream ss;
        ss << "RSF mismatch at wn=" << wn << " k=" << k
           << " parallel=" << rsf_parallel_h(wn, k)
           << " serial=" << rsf_serial_h(wn, k);
        logger.debug(ss.str());
      }
      REQUIRE(rsf_parallel_h(wn, k) == rsf_serial_h(wn, k));
    }
  }
}

// ============================================================================
// Test 2: jlong — serial Kokkos::single reference vs parallel
// ============================================================================
TEST_CASE("jlong_serial_vs_parallel", "mo_photo") {

  ekat::Comm comm;
  ekat::logger::Logger<> logger("jlong tests",
                                ekat::logger::LogLevel::debug, comm);

  constexpr int pver = nlev;
  constexpr Real Pa2mb = 1e-2;

  auto photo_table = build_test_photo_table();

  Atmosphere atm =
      init_atm_const_tv_lapse_rate(pver, 1000.0, 300.0, 0.01, 0.015, 7.5e-4);

  using View1D    = mo_photo::View1D;
  using View2D    = mo_photo::View2D;
  using View3D    = mo_photo::View3D;
  using View1DHost = Kokkos::View<Real*, Kokkos::HostSpace>;

  // Build alb_in, p_in, colo3_in
  View1DHost alb_in_host("alb_in_h", pver);
  View1DHost p_in_host("p_in_h", pver);
  View1DHost colo3_in_host("colo3_in_h", pver);

  auto pressure_host = Kokkos::create_mirror_view(atm.pressure);
  Kokkos::deep_copy(pressure_host, atm.pressure);
  for (int k = 0; k < pver; ++k) {
    alb_in_host(k)   = 0.15;
    p_in_host(k)     = pressure_host(k) * Pa2mb;
    colo3_in_host(k) = 3e17;
  }
  View1D alb_in_d("alb_in", pver);
  View1D p_in_d("p_in", pver);
  View1D colo3_in_d("colo3_in", pver);
  Kokkos::deep_copy(alb_in_d, alb_in_host);
  Kokkos::deep_copy(p_in_d, p_in_host);
  Kokkos::deep_copy(colo3_in_d, colo3_in_host);

  // Temperature from atmosphere (ConstColumnView inside kernel)
  const auto temper = atm.temperature;

  // ===== SERIAL REFERENCE =====
  // Reproduces the original Kokkos::single body of jlong, using a 2D stack
  // array xswk_s[test_numj][test_nw] of compile-time size.
  View2D j_long_serial("j_long_serial", test_numj, pver);
  Kokkos::deep_copy(j_long_serial, 0.0);

  // We also need rsf filled via serial interpolate_rsf for the serial reference.
  View2D rsf_for_serial("rsf_for_serial", test_nw, pver);
  Kokkos::deep_copy(rsf_for_serial, 0.0);
  {
    View2D psum_l_tmp("psum_l_tmp", pver, test_nw);
    View2D psum_u_tmp("psum_u_tmp", pver, test_nw);
    const auto sza_d       = photo_table.sza;
    const auto del_sza_d   = photo_table.del_sza;
    const auto alb_d       = photo_table.alb;
    const auto press_d     = photo_table.press;
    const auto del_p_d     = photo_table.del_p;
    const auto colo3_d     = photo_table.colo3;
    const auto o3rat_d     = photo_table.o3rat;
    const auto del_alb_d   = photo_table.del_alb;
    const auto del_o3rat_d = photo_table.del_o3rat;
    const auto etfphot_d   = photo_table.etfphot;
    const auto rsf_tab_d   = photo_table.rsf_tab;
    Kokkos::parallel_for(
        "rsf_for_jlong_serial", ThreadTeamPolicy(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const ThreadTeam &team) {
      mo_photo::interpolate_rsf(
          team, alb_in_d, test_sza_in, p_in_d, colo3_in_d, nlev,
          sza_d, del_sza_d, alb_d, press_d, del_p_d, colo3_d,
          o3rat_d, del_alb_d, del_o3rat_d, etfphot_d, rsf_tab_d,
          test_nw, test_nump, test_numsza, test_numcolo3, test_numalb,
          rsf_for_serial, psum_l_tmp, psum_u_tmp);
    });
    Kokkos::fence();
  }

  {
    const auto xsqy_d = photo_table.xsqy;
    const auto prs_d  = photo_table.prs;
    const auto dprs_d = photo_table.dprs;

    Kokkos::parallel_for(
        "jlong_serial_ref", ThreadTeamPolicy(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const ThreadTeam &team) {
      constexpr int nw_c    = test_nw;
      constexpr int numj_c  = test_numj;
      constexpr int pver_c  = nlev;
      constexpr int np_xs_c = test_np_xs;

      Kokkos::single(Kokkos::PerTeam(team), [=]() {
        const Real zero = 0;
        Real xswk_s[numj_c][nw_c] = {};

        for (int kk = 0; kk < pver_c; ++kk) {
          const int t_index =
              mam4::min(201, mam4::max(temper[kk] - 148.5, 1)) - 1;

          const Real ptarget = p_in_d[kk];
          if (ptarget >= prs_d[0]) {
            for (int wn = 0; wn < nw_c; ++wn)
              for (int i = 0; i < numj_c; ++i)
                xswk_s[i][wn] = xsqy_d(i, wn, t_index, 0);
          } else if (ptarget <= prs_d[np_xs_c - 1]) {
            for (int wn = 0; wn < nw_c; ++wn)
              for (int i = 0; i < numj_c; ++i)
                xswk_s[i][wn] = xsqy_d(i, wn, t_index, np_xs_c - 1);
          } else {
            Real delp = zero;
            int pndx  = 0;
            for (int km = 1; km < np_xs_c; ++km) {
              if (ptarget >= prs_d[km]) {
                pndx = km - 1;
                delp = (prs_d[pndx] - ptarget) * dprs_d[pndx];
                break;
              }
            }
            for (int wn = 0; wn < nw_c; ++wn)
              for (int i = 0; i < numj_c; ++i)
                xswk_s[i][wn] = xsqy_d(i, wn, t_index, pndx) +
                    delp * (xsqy_d(i, wn, t_index, pndx + 1) -
                            xsqy_d(i, wn, t_index, pndx));
          }

          for (int i = 0; i < numj_c; ++i) {
            Real suma = zero;
            for (int wn = 0; wn < nw_c; ++wn)
              suma += xswk_s[i][wn] * rsf_for_serial(wn, kk);
            j_long_serial(i, kk) = suma;
          }
        } // kk
      }); // Kokkos::single
    }); // parallel_for
    Kokkos::fence();
  }

  // ===== PARALLEL IMPLEMENTATION =====
  View2D j_long_parallel("j_long_parallel", test_numj, pver);
  Kokkos::deep_copy(j_long_parallel, 0.0);
  {
    View3D xswk_par("xswk_par", pver, test_numj, test_nw);
    View2D rsf_work("rsf_work", test_nw, pver); // filled by jlong's interpolate_rsf
    View2D psum_l_par("psum_l_par", pver, test_nw);
    View2D psum_u_par("psum_u_par", pver, test_nw);
    const auto sza_d       = photo_table.sza;
    const auto del_sza_d   = photo_table.del_sza;
    const auto alb_d       = photo_table.alb;
    const auto press_d     = photo_table.press;
    const auto del_p_d     = photo_table.del_p;
    const auto colo3_d     = photo_table.colo3;
    const auto o3rat_d     = photo_table.o3rat;
    const auto del_alb_d   = photo_table.del_alb;
    const auto del_o3rat_d = photo_table.del_o3rat;
    const auto etfphot_d   = photo_table.etfphot;
    const auto rsf_tab_d   = photo_table.rsf_tab;
    const auto xsqy_d      = photo_table.xsqy;
    const auto prs_d       = photo_table.prs;
    const auto dprs_d      = photo_table.dprs;

    Kokkos::parallel_for(
        "jlong_parallel", ThreadTeamPolicy(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const ThreadTeam &team) {
      mo_photo::jlong(
          team, test_sza_in, alb_in_d, p_in_d, temper, colo3_in_d,
          xsqy_d, sza_d, del_sza_d, alb_d, press_d, del_p_d, colo3_d,
          o3rat_d, del_alb_d, del_o3rat_d, etfphot_d, rsf_tab_d,
          prs_d, dprs_d,
          test_nw, test_nump, test_numsza, test_numcolo3, test_numalb,
          test_np_xs, test_numj,
          j_long_parallel,       // output
          rsf_work,              // rsf work array (filled internally)
          xswk_par, psum_l_par, psum_u_par);
    });
    Kokkos::fence();
  }

  // ===== COMPARE (exact BFB equality) =====
  auto j_serial_h   = Kokkos::create_mirror_view(j_long_serial);
  auto j_parallel_h = Kokkos::create_mirror_view(j_long_parallel);
  Kokkos::deep_copy(j_serial_h,   j_long_serial);
  Kokkos::deep_copy(j_parallel_h, j_long_parallel);

  for (int i = 0; i < test_numj; ++i) {
    for (int k = 0; k < pver; ++k) {
      if (j_parallel_h(i, k) != j_serial_h(i, k)) {
        std::ostringstream ss;
        ss << "j_long mismatch at reaction=" << i << " k=" << k
           << " parallel=" << j_parallel_h(i, k)
           << " serial="   << j_serial_h(i, k);
        logger.debug(ss.str());
      }
      REQUIRE(j_parallel_h(i, k) == j_serial_h(i, k));
    }
  }
}
