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

    // these variables depend on mode No and k
    const Pack drv_i = input.get("drv_i");
    const Pack drv_c = input.get("drv_c");
    const Pack init_num_i = input.get("init_num_i");
    const Pack init_num_c = input.get("init_num_c");

    auto num_i = Pack(init_num_i < 0, Pack(0.0), init_num_i);
    auto num_c = Pack(init_num_c < 0, Pack(0.0), init_num_c);

    // mode dependent
    Real v2nmin = input.get("v2nmin");
    Real v2nmax = input.get("v2nmax");

    Real v2nminrl = input.get("v2nminrl");
    Real v2nmaxrl = input.get("v2nmaxrl");

    Pack dqdt = input.get("dqdt");
    Pack dqqcwdt = input.get("dqqcwdt");
    Kokkos::parallel_for("adjust_num_sizes", 1, [&] KOKKOS_FUNCTION(int i) {
      calcsize::adjust_num_sizes(drv_i, drv_c, init_num_i, init_num_c, dt,
                                 v2nmin, v2nmax, v2nminrl, v2nmaxrl, num_i,
                                 num_c, dqdt, dqqcwdt);
    });

    output.set("dqqcwdt", dqqcwdt[0]);
    output.set("dqdt", dqdt[0]);
  });
}
