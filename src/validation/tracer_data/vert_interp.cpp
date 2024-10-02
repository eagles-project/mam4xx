// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <skywalker.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4;

void vert_interp(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Retrieve input arrays from the ensemble
    const auto pin_db = input.get_array("pin");
    const auto pmid_db = input.get_array("pmid");
    const auto datain_db = input.get_array("datain");
    const auto levsiz = static_cast<int>(input.get_array("levsiz")[0]);
    const auto pver = static_cast<int>(input.get_array("pver")[0]);

    // Data was written for only one column.
    const int ncol = 1; // Number of columns, example value
    // Define the Kokkos views based on the input data
    using View2D = typename DeviceType::view_2d<Real>;
    using View1DInt = typename DeviceType::view_1d<int>;

    View2D pin("pin", ncol, levsiz);
    View2D pmid("pmid", ncol, pver);
    View2D datain("datain", ncol, levsiz);
    View2D dataout("dataout", ncol, pver); // Output array
    View1DInt kupper("kupper", ncol);      // Work array

    // Convert input data from std::vector or similar structure to Kokkos views
    mam4::validation::convert_1d_vector_to_2d_view_device(pin_db, pin);
    mam4::validation::convert_1d_vector_to_2d_view_device(pmid_db, pmid);
    mam4::validation::convert_1d_vector_to_2d_view_device(datain_db, datain);

    // Perform the vertical interpolation
    mam4::vertical_interpolation::vert_interp(ncol, levsiz, pver, pin, pmid,
                                              datain, dataout, kupper);

    // Convert the output data from Kokkos view to a format suitable for the
    // ensemble
    std::vector<Real> dataout_db(pver * ncol);
    mam4::validation::convert_2d_view_device_to_1d_vector(dataout, dataout_db);

    // Set the output data in the ensemble
    output.set("dataout", dataout_db);
  });
}
