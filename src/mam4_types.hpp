#ifndef MAM4_TYPES_HPP
#define MAM4_TYPES_HPP

// This header defines common Haero aliases for use in MAM4.

#include <haero/atmosphere.hpp>
#include <haero/constants.hpp>
#include <haero/floating_point.hpp>

namespace mam4 {

using DeviceType = haero::DeviceType;
using Constants = haero::Constants;
using PackType = haero::PackType;
using PackInfo = haero::PackInfo;
using MaskType = haero::MaskType;
using Real = haero::Real;
using ColumnView = haero::ColumnView;
template <typename ST> using FloatingPoint = haero::FloatingPoint<ST>;
using Atmosphere = haero::Atmosphere;
using AeroSpecies = haero::AeroSpecies;
using IntPack = haero::IntPackType;
using ThreadTeam = haero::ThreadTeam;

} // namespace mam4

#endif
