#include <mam4xx/mam4.hpp>
#include <mam4xx/utils.hpp>

#include <catch2/catch.hpp>
#include <ekat/ekat_pack_kokkos.hpp>
#include <ekat/logging/ekat_logger.hpp>
#include <ekat/mpi/ekat_comm.hpp>

// if you need something from the data/ directory
// std::string data_file = MAM4_TEST_DATA_DIR;
// #include <mam4_test_config.hpp>

using namespace haero;

TEST_CASE("test_constructor", "mam4_rename_process") {
  mam4::AeroConfig mam4_config;
  mam4::RenameProcess process(mam4_config);
  REQUIRE(process.name() == "MAM4 rename");
  REQUIRE(process.aero_config() == mam4_config);
}

// TEST_CASE("test_find_renaming_pairs", "mam4_rename_process") {
//   // mam4::AeroConfig mam4_config;
//   // mam4::RenameProcess process(mam4_config);
//   mam4::rename::find_renaming_pairs();
//   REQUIRE(1 == 1);
// }

TEST_CASE("test_compute_dryvol_change_in_src_mode", "mam4_rename_process") {
  // mam4::AeroConfig mam4_config;
  // mam4::RenameProcess process(mam4_config);
  // mam4::rename::compute_dryvol_change_in_src_mode();
  REQUIRE(1 == 1);
}

TEST_CASE("test_do_inter_mode_transfer", "mam4_rename_process") {
  // mam4::AeroConfig mam4_config;
  // mam4::RenameProcess process(mam4_config);
  mam4::rename::do_inter_mode_transfer();
  REQUIRE(1 == 1);
}

TEST_CASE("test_compute_before_growth_dryvol_and_num", "mam4_rename_process") {
  // mam4::AeroConfig mam4_config;
  // mam4::RenameProcess process(mam4_config);
  mam4::rename::compute_before_growth_dryvol_and_num();
  REQUIRE(1 == 1);
}

TEST_CASE("test_total_interstitial_and_cloudborne", "mam4_rename_process") {
  // mam4::AeroConfig mam4_config;
  // mam4::RenameProcess process(mam4_config);
  Real outvar = mam4::rename::total_interstitial_and_cloudborne();
  REQUIRE(outvar == outvar);
}

TEST_CASE("test_mode_diameter", "mam4_rename_process") {
  Real diam[4] = {1.1e-7, 2.6e-8, 2e-6, 5e-8};
  Real sigma[4] = {1.8, 1.6, 1.8, 1.6};
  Real volume;
  Real number[4] = {1e3, 1e4, 1e5, 1e6};
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      volume = number[j] *
               mam4::conversions::mean_particle_volume_from_diameter(diam[i],
                                                                     sigma[i]);
      Real ln_sigma = log(sigma[i]);
      Real size_factor = Constants::pi_sixth * exp(4.5 * square(ln_sigma));
      Real mam_modeDiam =
          mam4::rename::mode_diameter(volume, number[j], size_factor);
      Real convDiam = mam4::conversions::mean_particle_diameter_from_volume(
          volume / number[j], sigma[i]);
      std::cout << "starting diam = " << diam[i] << "\n";
      std::cout << "mam_modeDiam = " << mam_modeDiam << "\n";
      std::cout << "convDiam = " << convDiam << "\n";
      REQUIRE(FloatingPoint<Real>::equiv(mam_modeDiam, convDiam));
      REQUIRE(FloatingPoint<Real>::equiv(mam_modeDiam, diam[i]));
      REQUIRE(FloatingPoint<Real>::equiv(convDiam, diam[i]));
    }
  }
}

TEST_CASE("test_compute_tail_fraction", "mam4_rename_process") {
  // mam4::AeroConfig mam4_config;
  // mam4::RenameProcess process(mam4_config);
  // test to see if it runs
  Real log_dia_tail_fac = 1.5;
  Real tail_fraction = 0.0;
  mam4::rename::compute_tail_fraction(1.0e-3, 3.0e-3, 2.0, log_dia_tail_fac,
                                      tail_fraction);
  CHECK(!isnan(tail_fraction));
}

// Everything below is leftover from calcsize but could be useful, so
// leaving it for now

// TEST_CASE("test_compute_tendencies", "mam4_calcsize_process") {
//   ekat::Comm comm;

//   ekat::logger::Logger<> logger("calcsize unit tests",
//                                 ekat::logger::LogLevel::debug, comm);

//   int nlev = 1;
//   Real pblh = 1000;
//   Atmosphere atm(nlev, pblh);
//   mam4::Prognostics progs(nlev);
//   mam4::Diagnostics diags(nlev);
//   mam4::Tendencies tends(nlev);

//   mam4::AeroConfig mam4_config;
//   mam4::CalcSizeProcess process(mam4_config);

//   const auto nmodes = mam4::AeroConfig::num_modes();

//   Kokkos::Array<Real, 21> interstitial = {
//       0.1218350564E-08, 0.3560443333E-08, 0.4203338951E-08,
//       0.3723412167E-09, 0.2330196615E-09, 0.1435909119E-10,
//       0.2704344376E-11, 0.2116400132E-11, 0.1326256343E-10,
//       0.1741610336E-16, 0.1280539377E-16, 0.1045693148E-08,
//       0.5358722850E-10, 0.2926142847E-10, 0.3986256848E-11,
//       0.3751267639E-10, 0.4679337373E-10, 0.9456518445E-13,
//       0.2272248527E-08, 0.2351792086E-09, 0.2733926271E-16};

//   Kokkos::Array<Real, nmodes> interstitial_num = {
//       0.8098354597E+09, 0.4425427527E+08, 0.1400840545E+06,
//       0.1382391601E+10};

//   std::ostringstream ss;
//   int count = 0;
//   for (int imode = 0; imode < nmodes; ++imode) {
//     auto h_prog_n_mode_i =
//     Kokkos::create_mirror_view(progs.n_mode_i[imode]);

//     for (int k = 0; k < nlev; ++k) {
//       // set all cell to same value.
//       h_prog_n_mode_i(k) = interstitial_num[imode];
//     }
//     Kokkos::deep_copy(progs.n_mode_i[imode], h_prog_n_mode_i);

//     ss << "progs.n_mode_i (mode No " << imode << ") [in]: [ ";
//     for (int k = 0; k < nlev; ++k) {
//       ss << h_prog_n_mode_i(k) << " ";
//     }
//     ss << "]";
//     logger.debug(ss.str());
//     ss.str("");

//     for (int k = 0; k < nlev; ++k) {
//       CHECK(!isnan(h_prog_n_mode_i(k)));
//     }

//     const auto n_spec = mam4::num_species_mode(imode);
//     for (int isp = 0; isp < n_spec; ++isp) {
//       auto h_prog_aero_i =
//           Kokkos::create_mirror_view(progs.q_aero_i[imode][isp]);
//       for (int k = 0; k < nlev; ++k) {
//         // set all cell to same value.
//         h_prog_aero_i(k) = interstitial[count];
//       }
//       Kokkos::deep_copy(progs.q_aero_i[imode][isp], h_prog_aero_i);
//       count++;

//       ss << "progs.q_aero_i (mode No " << imode << ", species No " << isp
//          << " ) [in]: [ ";
//       for (int k = 0; k < nlev; ++k) {
//         ss << h_prog_aero_i(k) << " ";
//       }
//       ss << "]";
//       logger.debug(ss.str());
//       ss.str("");

//       for (int k = 0; k < nlev; ++k) {
//         CHECK(!isnan(h_prog_aero_i(k)));
//       }

//     } // end species
//   }   // end modes

//   const int ncol = 1;
//   // Single-column dispatch.
//   auto team_policy = ThreadTeamPolicy(ncol, Kokkos::AUTO);
//   Real t = 0.0, dt = 30.0;
//   Kokkos::parallel_for(
//       team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
//         process.compute_tendencies(team, t, dt, atm, progs, diags,
//         tends);
//       });

//   for (int imode = 0; imode < nmodes; ++imode) {
//     auto h_tends_n_mode_i =
//     Kokkos::create_mirror_view(tends.n_mode_i[imode]);
//     Kokkos::deep_copy(h_tends_n_mode_i, tends.n_mode_i[imode]);

//     ss << "tends.n_mode_i (mode No " << imode << ") [out]: [ ";
//     for (int k = 0; k < nlev; ++k) {
//       ss << h_tends_n_mode_i(k) << " ";
//     }
//     ss << "]";
//     logger.debug(ss.str());
//     ss.str("");

//     for (int k = 0; k < nlev; ++k) {
//       CHECK(!isnan(h_tends_n_mode_i(k)));
//     }

//     const auto n_spec = mam4::num_species_mode(imode);
//     for (int isp = 0; isp < n_spec; ++isp) {
//       // const auto prog_aero_i =
//       ekat::scalarize(tends.q_aero_i[imode][i]); auto h_tends_aero_i =
//           Kokkos::create_mirror_view(tends.q_aero_i[imode][isp]);
//       Kokkos::deep_copy(h_tends_aero_i, tends.q_aero_i[imode][isp]);

//       ss << "tends.q_aero_i (mode No " << imode << ", species No " << isp
//          << " ) [out]: [ ";
//       for (int k = 0; k < nlev; ++k) {
//         ss << h_tends_aero_i(k) << " ";
//       }
//       ss << "]";
//       logger.debug(ss.str());
//       ss.str("");

//       for (int k = 0; k < nlev; ++k) {
//         CHECK(!isnan(h_tends_aero_i(k)));
//       }

//     } // end species
//   }   // end modes
// }
