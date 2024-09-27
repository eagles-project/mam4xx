// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PHYSICAL_LIMITS_HPP
#define PHYSICAL_LIMITS_HPP
#include <utility>
#include <string>
#include <map>

namespace mam4 {

  inline const std::pair<Real,Real>&  physical_min_max(const std::string &field_name) {
    static const std::map<std::string, std::pair<Real,Real>> 
      limits = {
        {"T_min", {100, 500} },
        {"qv", {1e-13, 0.2} },
        {"qc", {0, 0.1} },
        {"qt", {0, 0.1} },
        {"nc", {0, 0.1e11} },
        {"nr", {0, 0.1e10} },
        {"ni", {0, 0.1e10} },
        {"nmr", {0, 0.1e13} },
        {"mmr", {0, 0.1e-5} }
      };
    return limits.at(field_name);
  }

  inline Real physical_min(const std::string &field_name) {
    return physical_min_max(field_name).first;
  }
  inline Real physical_max(const std::string &field_name) {
    return physical_min_max(field_name).second;
  }
} // namespace mam4

#endif
