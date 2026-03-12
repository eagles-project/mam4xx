// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_SURFACE_HPP
#define MAM4XX_SURFACE_HPP

#include <ekat_assert.hpp>

namespace mam4 {

class Surface final {

  // use only for creating containers of Surface related variables!
public:
  Surface() = default;

  // these are supported for initializing containers of Surface
  Surface(const Surface &rhs) = default;
  Surface &operator=(const Surface &rhs) = default;

  /// destructor, valid on both host and device
  KOKKOS_FUNCTION
  ~Surface() {}

  // land fraction [unitless]
  Real land_frac;

  // ice fraction [unitless]
  Real ice_frac;

  // ocean fraction [unitless]
  Real ocn_frac;

  // friction velocity from land model [m/s]
  Real ustar;

  // aerodynamical resistance from land model [s/m]
  Real ram1in;
};

} // namespace mam4

#endif
