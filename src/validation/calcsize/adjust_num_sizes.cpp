#include <haero/constants.hpp>
#include <iostream>
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

    Pack drv_i[nmodes];
    Pack drv_c[nmodes];

    Pack init_num_i[nmodes];
    Pack init_num_c[nmodes];

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

    Pack interstitial_tend[nmodes] = {0};
    Pack cloudborne_tend[nmodes] = {0};

    Kokkos::parallel_for("adjust_num_sizes", 1, [&] KOKKOS_FUNCTION(int i) {
      for (int m = 0; m < nmodes; ++m) {
        auto num_i = Pack(init_num_i[m] < 0, Pack(0.0), init_num_i[m]);
        auto num_c = Pack(init_num_c[m] < 0, Pack(0.0), init_num_c[m]);
        calcsize::adjust_num_sizes(drv_i[m], drv_c[m], init_num_i[m],
                                   init_num_c[m], dt, v2nmin[m], v2nmax[m],
                                   v2nminrl[m], v2nmaxrl[m], num_i, num_c,
                                   interstitial_tend[m], cloudborne_tend[m]);
      }
    });

    std::vector<Real> interstitial_tend_values, cloudborne_tend_values;

    for (int m = 0; m < nmodes; ++m) {
      interstitial_tend_values.push_back(interstitial_tend[m][0]);
      cloudborne_tend_values.push_back(cloudborne_tend[m][0]);
    }

    output.set("interstitial_tend", interstitial_tend_values);
    output.set("cloudborne_tend", cloudborne_tend_values);
  });


}
