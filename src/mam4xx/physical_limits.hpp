// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PHYSICAL_LIMITS_HPP
#define PHYSICAL_LIMITS_HPP
#include <map>
#include <string>
#include <utility>

namespace mam4 {

#if 0
  // If a device callable version of physical_min and physical_max is needed along
  // with the CPU only version, somthing like this would work. Strings do not do
  // well on device but enums are fast.
  
  enum FieldNameIndex { T_mid, qv , qc , qt , nc , nr , ni , nmr , mmr, NUMFIELD };
  struct min_max {const Real min; const Real max;};

  KOKKOS_INLINE_FUNCTION constexpr min_max physical_min_max(const FieldNameIndex ind) {
    const min_max field_min_max[NUMFIELD] = {
      {100, 500},         // T_mid
      {1e-13, 0.2},       // qv 
      {0, 0.1},           // qc 
      {0, 0.1},           // qt 
      {0, 0.1e11},        // nc 
      {0, 0.1e10},        // nr 
      {0, 0.1e10},        // ni 
      {100, 0.1e13},      // nmr 
      {100, 0.1e-5}       // mmr
    };
    return field_min_max[ind];
  }

  KOKKOS_INLINE_FUNCTION constexpr Real physical_min(const FieldNameIndex ind) {
    return physical_min_max(ind).min;
  }

  KOKKOS_INLINE_FUNCTION constexpr Real physical_max(const FieldNameIndex ind) {
    return physical_min_max(ind).max;
  }

  inline const std::pair<Real,Real>&  physical_min_max(const std::string &field_name) {
    static const std::map<std::string, std::pair<Real,Real>> 
      limits = {
        {"T_mid", {physical_min(T_mid), physical_max(T_mid)} },
        {"qv",    {physical_min(qv), physical_max(qv)} },
        {"qc",    {physical_min(qc), physical_max(qc)} },
        {"qt",    {physical_min(qt), physical_max(qt)} },
        {"nc",    {physical_min(nc), physical_max(nc)} },
        {"nr",    {physical_min(nr), physical_max(nr)} },
        {"ni",    {physical_min(ni), physical_max(ni)} },
        {"nmr",   {physical_min(nmr), physical_max(nmr)} },
        {"mmr",   {physical_min(mmr), physical_max(mmr)} }
      };
    return limits.at(field_name);
  }
#endif
inline const std::pair<Real, Real> &
physical_min_max(const std::string &field_name) {
  static const std::map<std::string, std::pair<Real, Real>> limits = {
      {"T_mid", {100, 500}}, {"qv", {1e-13, 0.2}}, {"qc", {0, 0.1}},
      {"qt", {0, 0.1}},      {"nc", {0, 0.1e11}},  {"nr", {0, 0.1e10}},
      {"ni", {0, 0.1e10}},   {"nmr", {0, 0.1e13}}, {"mmr", {0, 0.1e-5}}};
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
