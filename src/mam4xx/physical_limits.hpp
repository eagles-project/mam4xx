// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PHYSICAL_LIMITS_HPP
#define PHYSICAL_LIMITS_HPP
#include <string>
#include <map>

namespace mam4 {

  inline Real physical_min_max(const std::string &field_name) {
    const std::map<std::string, std::pair<Real,Real>> 
      limits = {
        {"T_min", {100,500} }
      };
    auto iter = limits.find(field_name);
    EKAT_REQUIRE_MSG(iter != limits.end(),
      std::string("Bounds not defined for field name:"+field_name);

    return iter->first;
  }

  inline Real physical_min(const std::string &field_name) {
    return physical_min_max(field_name).second;
  }
} // namespace mam4

#endif
