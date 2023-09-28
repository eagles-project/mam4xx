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

    // lwsw   ! indicator if this is lw or sw lw =0 and sw =1
    // ncol, ilwsw
    const int lwsw = int(input.get_array("lwsw")[0]);
    // Fortran to C++ indexing
    const int ilwsw = int(input.get_array("ilwsw")[0]) - 1;
    const Real qaerwat_kk = input.get_array("qaerwat_kk")[0];
    const auto specvol = input.get_array("specvol");

    const int nspec = specvol.size();

    const auto real_specrefindex = input.get_array("specrefindex_real");
    const auto imag_specrefindex = input.get_array("specrefindex_imag");

    int nbands = 0;
    if (lwsw == 0) {
      nbands = nlwbands;
    } else if (lwsw == 1) {
      nbands = nswbands;
    }

    ComplexView2D specrefindex("specrefindex", nspec, nbands);

    int count = 0;
    for (int j = 0; j < nbands; ++j) {
      for (int i = 0; i < nspec; ++i) {

        specrefindex(i, j).real() = real_specrefindex[count];
        specrefindex(i, j).imag() = imag_specrefindex[count];
        count += 1;
      }
    }

    const auto crefwlw_real = input.get_array("crefwlw_real");
    const auto crefwlw_imag = input.get_array("crefwlw_imag");

    Kokkos::complex<Real> crefwlw[nlwbands];

    for (int j = 0; j < nlwbands; ++j) {
      crefwlw[j].real() = crefwlw_real[j];
      crefwlw[j].imag() = crefwlw_imag[j];
    }

    const auto crefwsw_real = input.get_array("crefwsw_real");
    const auto crefwsw_imag = input.get_array("crefwsw_imag");
    Kokkos::complex<Real> crefwsw[nswbands];

    for (int j = 0; j < nswbands; ++j) {
      crefwsw[j].real() = crefwsw_real[j];
      crefwsw[j].imag() = crefwsw_imag[j];
    }

    Real dryvol = zero;
    Real wetvol = zero;
    Real watervol = zero;
    Kokkos::complex<Real> crefin = {};
    Real refr = zero;
    Real refi = zero;
    calc_refin_complex(lwsw, ilwsw, qaerwat_kk, specvol.data(), specrefindex,
                       nspec, crefwlw, crefwsw, dryvol, wetvol, watervol,
                       crefin, refr, refi);

    output.set("dryvol", std::vector<Real>(1, dryvol));
    output.set("wetvol", std::vector<Real>(1, wetvol));
    output.set("watervol", std::vector<Real>(1, watervol));
    output.set("refr", std::vector<Real>(1, refr));
    output.set("refi", std::vector<Real>(1, refi));
  });
}
