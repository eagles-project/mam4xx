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
using namespace modal_aer_opt;
// using namespace ndrop;

void calc_refin_complex(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    constexpr Real zero = 0;

    using View1DHost = typename HostType::view_1d<Real>;

    // lwsw   ! indicator if this is lw or sw lw =0 and sw =1
    // ncol, ilwsw
    const int lwsw = int(input.get_array("lwsw")[0]);
    // Fortran to C++ indexing
    const int ilwsw = int(input.get_array("ilwsw")[0]) - 1;
    const Real qaerwat_kk = input.get_array("qaerwat_kk")[0];
    const auto specvol_db = input.get_array("specvol");
    const int nspec = specvol_db.size();

    auto specvol_host = View1DHost((Real *)specvol_db.data(), nspec);
    View1D specvol("specvol", nspec);
    Kokkos::deep_copy(specvol, specvol_host);

    const auto real_specrefindex = input.get_array("specrefindex_real");
    const auto imag_specrefindex = input.get_array("specrefindex_imag");

    int nbands = 0;
    if (lwsw == 0) {
      nbands = nlwbands;
    } else if (lwsw == 1) {
      nbands = nswbands;
    }

    ComplexView2D specrefindex("specrefindex", nspec, nbands);
    const auto specrefindex_host = Kokkos::create_mirror_view(specrefindex);

    int count = 0;
    for (int j = 0; j < nbands; ++j) {
      for (int i = 0; i < nspec; ++i) {
        specrefindex_host(i, j).real() = real_specrefindex[count];
        specrefindex_host(i, j).imag() = imag_specrefindex[count];
        count += 1;
      }
    }

    Kokkos::deep_copy(specrefindex, specrefindex_host);

    const auto crefwlw_real = input.get_array("crefwlw_real");
    const auto crefwlw_imag = input.get_array("crefwlw_imag");

    ComplexView1D crefwlw("crefwlw", nlwbands);
    const auto crefwlw_host = Kokkos::create_mirror_view(crefwlw);

    for (int j = 0; j < nlwbands; ++j) {
      crefwlw_host(j).real() = crefwlw_real[j];
      crefwlw_host(j).imag() = crefwlw_imag[j];
    }

    Kokkos::deep_copy(crefwlw, crefwlw_host);

    const auto crefwsw_real = input.get_array("crefwsw_real");
    const auto crefwsw_imag = input.get_array("crefwsw_imag");

    ComplexView1D crefwsw("crefwsw", nswbands);
    const auto crefwsw_host = Kokkos::create_mirror_view(crefwsw);

    for (int j = 0; j < nswbands; ++j) {
      crefwsw_host(j).real() = crefwsw_real[j];
      crefwsw_host(j).imag() = crefwsw_imag[j];
    }

    Kokkos::deep_copy(crefwsw, crefwsw_host);

    View1D outputs("outputs", 5);
    auto team_policy = ThreadTeamPolicy(1u, Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
          Real refr = zero;
          Real refi = zero;
          Real dryvol = zero;
          Real wetvol = zero;
          Real watervol = zero;
          Kokkos::complex<Real> crefin{};

          calc_refin_complex(lwsw, ilwsw, qaerwat_kk, specvol.data(),
                             specrefindex, nspec, crefwlw, crefwsw, dryvol,
                             wetvol, watervol, crefin, refr, refi);

          outputs(0) = dryvol;
          outputs(1) = wetvol;
          outputs(2) = watervol;
          outputs(3) = refr;
          outputs(4) = refi;
        });

    const auto ouputs_host = Kokkos::create_mirror_view(outputs);
    Kokkos::deep_copy(ouputs_host, outputs);

    output.set("dryvol", std::vector<Real>(1, ouputs_host(0)));
    output.set("wetvol", std::vector<Real>(1, ouputs_host(1)));
    output.set("watervol", std::vector<Real>(1, ouputs_host(2)));
    output.set("refr", std::vector<Real>(1, ouputs_host(3)));
    output.set("refi", std::vector<Real>(1, ouputs_host(4)));
  });
}
