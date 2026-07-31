// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_SPECIES_HPP
#define MAM4XX_AERO_SPECIES_HPP

#include <mam4xx/mam4_config.hpp>

#include <ekat_kokkos_types.hpp>

#include <string>

namespace mam4 {

/// Identifiers for aerosol species that inhabit MAM4 modes.
enum class AeroId {
  SOA = 0,        // secondary organic aerosol
  SO4 = 1,        // sulphate
  POM = 2,        // primary organic matter
  BC = 3,         // black carbon
  NaCl = 4,       // sodium chloride
  DST = 5,        // dust
  MOM = 6,        // marine organic matter,
  NumSpecies = 7, // number of aerosol species
  None = 8        // invalid aerosol species
};

/// @struct AeroSpecies
/// This type represents an aerosol species.
struct AeroSpecies {
  // Canonical species identifier
  AeroId id;
  // Molecular weight [kg/mol]
  Real molecular_weight;

  /// Material density [kg/m^3]
  Real density;

  /// Hygroscopicity
  Real hygroscopicity;
};

/// This wraps a view of AeroSpecies, allowing only certain operations, and
/// indexing by AeroId, which creates a distinction between AeroId and integer
/// aerosol-mode indices, which are a rich source of bugs.
template <typename DeviceType> class AeroSpeciesList {
public:
  explicit AeroSpeciesList(const std::string &label)
      : view_(label, int(AeroId::NumSpecies)) {}
  AeroSpeciesList(const std::string &label,
                  std::initializer_list<AeroSpecies> species)
      : view_(label, int(AeroId::NumSpecies)) {
    for (size_t i = 0; i < species.size(); ++i)
      view_[i] = species.begin()[i];
  }

  // access to species via AeroIds
  KOKKOS_INLINE_FUNCTION
  const AeroSpecies &operator[](AeroId id) const { return view_[int(id)]; }
  KOKKOS_INLINE_FUNCTION
  AeroSpecies &operator[](AeroId id) { return view_[int(id)]; }

  KOKKOS_INLINE_FUNCTION
  constexpr size_t size() const { return view_.extent(0); }

private:
  typename ekat::KokkosTypes<DeviceType>::view_1d<AeroSpecies> view_;

  // these functions get access to view_
  friend AeroSpeciesList<ekat::DefaultDevice>
  aero_species_on_device(const AeroSpeciesList<ekat::HostDevice> &);
  friend AeroSpeciesList<ekat::HostDevice>
  aero_species_on_host(const AeroSpeciesList<ekat::DefaultDevice> &);
};

/// TODO: Rename these types to indicate that they're not **exactly** Kokkos
/// views, but rather thin wrappers around them to enforce aerosol identifier
/// type safety.
using AeroSpeciesView = AeroSpeciesList<ekat::DefaultDevice>;
using AeroSpeciesHostView = AeroSpeciesList<ekat::HostDevice>;

// This iterator allows you to loop over all aerosol species with a C++ range
// iterator. E.g. for (AeroId aid: all_aerosol_ids()) {
//   // your logic goes here
// }
class AeroIdIterator {
  int value_;

public:
  KOKKOS_INLINE_FUNCTION
  explicit AeroIdIterator(int v) : value_(v) {}
  KOKKOS_INLINE_FUNCTION
  AeroId operator*() const { return static_cast<AeroId>(value_); }
  KOKKOS_INLINE_FUNCTION
  AeroIdIterator &operator++() {
    ++value_;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  bool operator!=(const AeroIdIterator &other) const {
    return value_ != other.value_;
  }
};

class AeroIdRange {
  int begin_value_, end_value_;

public:
  KOKKOS_INLINE_FUNCTION
  AeroIdRange() : begin_value_(0), end_value_(int(AeroId::NumSpecies)) {}
  KOKKOS_INLINE_FUNCTION
  AeroIdIterator begin() const { return AeroIdIterator(begin_value_); }
  KOKKOS_INLINE_FUNCTION
  AeroIdIterator end() const { return AeroIdIterator(end_value_); }
};

KOKKOS_INLINE_FUNCTION
AeroIdRange all_aerosol_ids() { return AeroIdRange(); }

// Here's a generic container you can associate with an aerosol species,
// accessing it with an AeroId instead of an integer index.
template <typename T> class AeroSpeciesData {
public:
  AeroSpeciesData() : view_() {}
  explicit AeroSpeciesData(const std::string &label)
      : view_(label, int(AeroId::NumSpecies)) {}
  AeroSpeciesData(const std::string &label, std::initializer_list<T> data)
      : view_(label, int(AeroId::NumSpecies)) {
    for (size_t i = 0; i < data.size(); ++i)
      view_[i] = data.begin()[i];
  }
  KOKKOS_INLINE_FUNCTION
  AeroSpeciesData(const AeroSpeciesData &) = default;

  KOKKOS_INLINE_FUNCTION
  AeroSpeciesData &operator=(const AeroSpeciesData &) = default;

  // access to species via AeroIds
  KOKKOS_INLINE_FUNCTION
  const T &operator[](AeroId id) const { return view_[int(id)]; }
  KOKKOS_INLINE_FUNCTION
  T &operator[](AeroId id) { return view_[int(id)]; }
  KOKKOS_INLINE_FUNCTION
  constexpr size_t size() const { return view_.extent(0); }

private:
  typename ekat::KokkosTypes<ekat::DefaultDevice>::view_1d<T> view_;
};

// default values for aerosol species properties
namespace defaults {

/// Molecular weight of mam4 dust aerosol [kg/mol]
static constexpr Real mam4_molec_weight_dst = 0.135065;

/// Molecular weight of mam4 marine organic matter [kg/mol]
static constexpr Real mam4_molec_weight_mom = 250.093;

/// mam4 aerosol densities [kg/m3]
static constexpr Real mam4_density_soa = 1000.0;
static constexpr Real mam4_density_so4 = 1770.0;
static constexpr Real mam4_density_pom = 1000.0;
static constexpr Real mam4_density_bc = 1700.0;
static constexpr Real mam4_density_nacl = 1900.0;
static constexpr Real mam4_density_dst = 2600.0;
static constexpr Real mam4_density_mom = 1601.0;

/// mam4 aerosol hygroscopicities
static constexpr Real mam4_hyg_soa = 0.1;
static constexpr Real mam4_hyg_so4 = 0.507;
static constexpr Real mam4_hyg_pom = 1e-10;
static constexpr Real mam4_hyg_bc = 1e-10;
static constexpr Real mam4_hyg_nacl = 1.16;
static constexpr Real mam4_hyg_dst = 0.14;
static constexpr Real mam4_hyg_mom = 0.1;

} // namespace defaults

//--------------------------------------------------------
// The following functions can only be called on the host
//--------------------------------------------------------

/// Returns a newly-created host view containing the default configuration for
/// aerosol species. Create this on the host, override properties as desired,
/// and copy to device with Kokkos::deep_copy.
AeroSpeciesHostView default_aero_species();

/// Return a newly-created device view whose data is copied from the given host
/// view.
AeroSpeciesView
aero_species_on_device(const AeroSpeciesHostView &species_on_host);

/// Return a newly-created host view whose data is copied from the given device
/// view.
AeroSpeciesHostView
aero_species_on_host(const AeroSpeciesView &species_on_device);

/// Maps an AeroId to the name of its species.
std::string aero_id_str(const AeroId id);

/// Maps an AeroId to a shortened name for its species.
std::string aero_id_short_name(const AeroId id);

} // namespace mam4

#endif
