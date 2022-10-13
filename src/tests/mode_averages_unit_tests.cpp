#include "atmosphere_init.hpp"

#include "mam4xx/mam4_types.hpp"
#include "mam4xx/aero_modes.hpp"
#include "mam4xx/aero_config.hpp"
#include "mam4xx/conversions.hpp"
#include "mam4xx/mode_dry_particle_size.hpp"
#include "mam4xx/mode_hygroscopicity.hpp"
#include "mam4xx/mode_wet_particle_size.hpp"

#include "haero/floating_point.hpp"
#include "haero/atmosphere.hpp"

#include "catch2/catch.hpp"

#include "ekat/ekat_pack_kokkos.hpp"
#include "ekat/logging/ekat_logger.hpp"
#include "ekat/mpi/ekat_comm.hpp"

using namespace mam4;

TEST_CASE("modal_averages", "") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("modal averages tests", ekat::logger::LogLevel::debug, comm);

  const int nlev = 72;
  const int nk = PackInfo::num_packs(nlev);

  Prognostics progs(nlev);
  Diagnostics diags(nlev);

  const Real number_mixing_ratio = 2e7;
  const Real mass_mixing_ratio = 3e-8;
  for (int m=0; m<4; ++m) {
    auto h_n_view = Kokkos::create_mirror_view(progs.n_mode[m]);
    for (int k=0; k<nlev; ++k) {
      const int pack_idx = PackInfo::pack_idx(k);
      const int vec_idx = PackInfo::vec_idx(k);
      h_n_view(pack_idx)[vec_idx] = number_mixing_ratio;
    }
    Kokkos::deep_copy(progs.n_mode[m], h_n_view);
    for (int aid=0; aid<7; ++aid) {
      const int s = aerosol_index_for_mode(static_cast<ModeIndex>(m),
        static_cast<AeroId>(aid));
      if (s>=0) {
        auto h_q_view = Kokkos::create_mirror_view(progs.q_aero_i[m][s]);
        for (int k=0; k<nlev; ++k) {
          const int pack_idx = PackInfo::pack_idx(k);
          const int vec_idx = PackInfo::vec_idx(k);
          h_q_view(pack_idx)[vec_idx] = mass_mixing_ratio;
        }
        Kokkos::deep_copy(progs.q_aero_i[m][s], h_q_view);
      }
    }
  }

  SECTION("dry particle size") {

    PackType dry_aero_mean_particle_volume[4];
    PackType dry_aero_mean_particle_diam[4];
    for (int m=0; m<4; ++m) {
      PackType dry_vol(0);
      for (int aid=0; aid<7; ++aid) {
        const int s = aerosol_index_for_mode(
          static_cast<ModeIndex>(m), static_cast<AeroId>(aid));
        if (s >= 0) {
          dry_vol += mass_mixing_ratio / aero_species[s].density;
        }
      }
      const PackType mean_vol = dry_vol / number_mixing_ratio;
      dry_aero_mean_particle_volume[m] = mean_vol;
      dry_aero_mean_particle_diam[m] =
        conversions::mean_particle_diameter_from_volume(mean_vol, modes[m].mean_std_dev);

      logger.info("{} mode has mean particle diameter {}",
        mode_str(static_cast<ModeIndex>(m)), dry_aero_mean_particle_diam[m]);
    }

    Kokkos::parallel_for("compute_dry_particle_size", nk,
      KOKKOS_LAMBDA (const int i) {
        mode_avg_dry_particle_diam(diags, progs, i);
      });

    for (int m=0; m<4; ++m) {
      auto h_diam = Kokkos::create_mirror_view(diags.dry_geometric_mean_diameter[m]);
      Kokkos::deep_copy(h_diam, diags.dry_geometric_mean_diameter[m]);
      for (int k=0; k<nk; ++k) {
        if (!FloatingPoint<PackType>::equiv(h_diam(k), dry_aero_mean_particle_diam[m])) {
          logger.debug("h_diam({}) = {}, dry_aero_mean_particle_diam[{}] = {}",
            k, h_diam(k), m, dry_aero_mean_particle_diam[m]);
        }
        REQUIRE(FloatingPoint<PackType>::equiv(h_diam(k), dry_aero_mean_particle_diam[m]));
      }
    }
    logger.info("dry particle size tests complete.");
  } // section (dry particle size)

  SECTION("hygroscopicity") {
    PackType hygro[4];
    for (int m=0; m<4; ++m) {
      PackType dry_vol(0);
      PackType hyg(0);
      for (int aid=0; aid<7; ++aid) {
        const int s = aerosol_index_for_mode(static_cast<ModeIndex>(m),
          static_cast<AeroId>(aid));
        if (s>=0) {
          dry_vol += mass_mixing_ratio / aero_species[s].density;
          hyg += mass_mixing_ratio * aero_species[s].hygroscopicity /
            aero_species[s].density;
        }
      }
      hygro[m] = hyg / dry_vol;
      logger.info("{} mode has hygroscopicity {}",
        mode_str(static_cast<ModeIndex>(m)), hygro[m]);
    }

    Kokkos::parallel_for("compute_hygroscopicity", nk,
      KOKKOS_LAMBDA (const int i) {
        mode_hygroscopicity(diags, progs, i);
      });

    for (int m=0; m<4; ++m) {
      auto h_hyg = Kokkos::create_mirror_view(diags.hygroscopicity[m]);
      Kokkos::deep_copy(h_hyg, diags.hygroscopicity[m]);
      for (int k=0; k<nk; ++k) {
        if (!FloatingPoint<PackType>::equiv(h_hyg(k), hygro[m])) {
          logger.debug("h_hyg({}) = {}, hygro[{}] = {}",
            k, h_hyg(k), m, hygro[m]);
        }
        REQUIRE(FloatingPoint<PackType>::equiv(h_hyg(k), hygro[m]));
      }
    }
    logger.info("hygroscopicity tests complete.");
  } // section (hygroscopicity)

  SECTION("wet particle size") {
    const Real pblh = 0;
    Atmosphere atm(nlev, pblh);
    init_atm_const_lapse_rate(atm);

    Kokkos::parallel_for("compute_wet_particle_size", nk,
      KOKKOS_LAMBDA (const int i) {
        mode_avg_dry_particle_diam(diags, progs, i);
        Kokkos::fence();
        mode_hygroscopicity(diags, progs, i);
        Kokkos::fence();
        mode_avg_wet_particle_diam(diags, progs, atm, i);
      });

  } // section wet particle size
} // test case
