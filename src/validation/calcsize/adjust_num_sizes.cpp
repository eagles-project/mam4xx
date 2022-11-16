#include <mam4xx/calcsize.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void adjust_num_sizes(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    // Fetch ensemble parameters

    Real dt = input.get("dt");

    const auto nmodes = AeroConfig::num_modes();

    // these variables depend on mode No and k
    auto in_drv_i = input.get_array("dryvol_i");
    auto in_drv_c = input.get_array("dryvol_c");
    auto in_init_num_i = input.get_array("interstitial_num");
    auto in_init_num_c = input.get_array("cloud_borne_num");

    // mode dependent
    auto in_v2nmin = input.get_array("v2nmin");
    auto in_v2nmax = input.get_array("v2nmax");

    auto in_v2nminrl = input.get_array("v2nminrl");
    auto in_v2nmaxrl = input.get_array("v2nmaxrl");

    Real drv_i[nmodes];
    Real drv_c[nmodes];

    Real init_num_i[nmodes];
    Real init_num_c[nmodes];

    Real v2nmin[nmodes];
    Real v2nmax[nmodes];

    Real v2nminrl[nmodes];
    Real v2nmaxrl[nmodes];

    for (int m = 0; m < nmodes; ++m) {
      drv_i[m] = in_drv_i[m];
      drv_c[m] = in_drv_c[m];
      init_num_i[m] = in_init_num_i[m];
      init_num_c[m] = in_init_num_c[m];

      v2nmin[m] = in_v2nmin[m];
      v2nmax[m] = in_v2nmax[m];

      v2nminrl[m] = in_v2nminrl[m];
      v2nmaxrl[m] = in_v2nmaxrl[m];
    }

    static constexpr Real close_to_one = 1.0 + 1.0e-15;
    static constexpr Real seconds_in_a_day = 86400.0;
    const auto adj_tscale = max(seconds_in_a_day, dt);
    const auto adj_tscale_inv = 1.0 / (adj_tscale * close_to_one);

    Real interstitial_tend[nmodes] = {0};
    Real cloudborne_tend[nmodes] = {0};

    Real num_i[nmodes];
    Real num_c[nmodes];
    const Real zero = 0;

    Kokkos::parallel_for("adjust_num_sizes", 1, [&] KOKKOS_FUNCTION(int i) {
      for (int m = 0; m < nmodes; ++m) {
        num_i[m] = init_num_i[m] < 0 ? zero : init_num_i[m];
        num_c[m] = init_num_c[m] < 0 ? zero : init_num_c[m];
        calcsize::adjust_num_sizes(
            drv_i[m], drv_c[m], init_num_i[m], init_num_c[m], dt, v2nmin[m],
            v2nmax[m], v2nminrl[m], v2nmaxrl[m], adj_tscale_inv, // in
            close_to_one, num_i[m], num_c[m], interstitial_tend[m],
            cloudborne_tend[m]);
      }
    });

    std::vector<Real> interstitial_tend_values, cloudborne_tend_values;
    std::vector<Real> num_i_values, num_c_values;

    for (int m = 0; m < nmodes; ++m) {
      interstitial_tend_values.push_back(interstitial_tend[m]);
      cloudborne_tend_values.push_back(cloudborne_tend[m]);
      num_i_values.push_back(num_i[m]);
      num_c_values.push_back(num_c[m]);
    }

    output.set("interstitial_tend", interstitial_tend_values);
    output.set("cloudborne_tend", cloudborne_tend_values);
    output.set("num_i", num_i_values);
    output.set("num_c", num_c_values);
  });
}
