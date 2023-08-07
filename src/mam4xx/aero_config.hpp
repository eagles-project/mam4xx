// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_AERO_CONFIG_HPP
#define MAM4XX_AERO_CONFIG_HPP

#include <mam4xx/aero_modes.hpp>

#include <algorithm>
#include <map>
#include <numeric>

namespace mam4 {

class Prognostics; // fwd decl
class Diagnostics; // fwd decl
// Tendencies are identical in structure to prognostics.
using Tendencies = Prognostics; // fwd decl

/// @struct MAM4::AeroConfig: for use with all MAM4 process implementations
class AeroConfig final {
public:
  // Types.
  using Prognostics = ::mam4::Prognostics;
  using Diagnostics = ::mam4::Diagnostics;
  using Tendencies = ::mam4::Tendencies;

  bool calculate_gas_uptake_coefficient = false;
  int number_gauss_points_for_integration = 2;
  // Default constructor.
  AeroConfig() {}

  // Copy constructor.
  AeroConfig(const AeroConfig &) = default;

  // Destructor.
  ~AeroConfig() = default;

  // Assignment operator.
  AeroConfig &operator=(const AeroConfig &) = default;

  // Comparison operators.
  inline bool operator==(const AeroConfig &other) const {
    return true; // all MAM4 configs are equivalent
  }
  inline bool operator!=(const AeroConfig &other) const {
    return false; // all MAM4 configs are equivalent
  }

  /// Returns the number of aerosol modes.
  static constexpr int num_modes() { return 4; }

  /// Returns the number of aerosol ids. This is the number of enums in
  /// mam4::AeroId.
  static constexpr int num_aerosol_ids() { return 7; }

  /// Returns the number of gas ids. This is the number of enums in mam4::GasId.
  static constexpr int num_gas_ids() { return 3; }

  /// Returns the number of aging pairs
  static constexpr int max_agepair() { return 1; }
};

/// MAM4 column-wise prognostic aerosol fields (also used for tendencies).
class Prognostics final {
  // number of vertical levels
  int nlev_;

public:
  using ColumnView = haero::ColumnView;
  using ThreadTeam = haero::ThreadTeam;

  /// Creates a container for prognostic variables on the specified number of
  /// vertical levels. All views must be set manually.
  explicit Prognostics(int num_levels) : nlev_(num_levels) {}

  Prognostics() = default; // use only for creating containers of Prognostics!
  ~Prognostics() = default;

  // these are used to populate containers of Prognostics objects
  Prognostics(const Prognostics &rhs) = default;
  Prognostics &operator=(const Prognostics &rhs) = default;

  ///  modal interstitial aerosol number mixing ratios (see aero_mode.hpp for
  ///  indexing)
  ColumnView n_mode_i[AeroConfig::num_modes()];

  /// modal cloudborne aerosol number mixing ratios (see aero_mode.hpp for
  /// indexing)
  ColumnView n_mode_c[AeroConfig::num_modes()];

  /// interstitial aerosol mass mixing ratios within each mode
  /// (see aero_mode.hpp for indexing)
  ColumnView q_aero_i[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()];

  /// cloudborne aerosol mass mixing ratios within each mode
  /// (see aero_mode.hpp for indexing)
  ColumnView q_aero_c[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()];

  /// gas mass mixing ratios (see aero_mode.hpp for indexing)
  ColumnView q_gas[AeroConfig::num_gas_ids()];

  /// time average of the gas mix ratios over the time step of
  /// integration (see aero_mode.hpp for indexing)
  ColumnView q_gas_avg[AeroConfig::num_gas_ids()];

  /// Uptate Rate for each gas species and each mode.
  /// i.e. Gas to aerosol mass transfer rate (1/s)
  ColumnView uptkaer[AeroConfig::num_gas_ids()][AeroConfig::num_modes()];

  KOKKOS_INLINE_FUNCTION
  int num_levels() const { return nlev_; }

  /// Returns true iff all prognostic quantities are nonnegative, using the
  /// given thread team to parallelize the check.
  KOKKOS_INLINE_FUNCTION
  bool quantities_nonnegative(const ThreadTeam &team) const {
    const int nk = num_levels();
    int violations = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, nk),
        KOKKOS_CLASS_LAMBDA(int k, int &violation) {
          for (int mode = 0; mode < AeroConfig::num_modes();
               ++mode) { // check mode mmrs
            if ((n_mode_i[mode](k) < 0) || (n_mode_c[mode](k) < 0)) {
              ++violation;
            } else {
              for (int spec = 0; spec < AeroConfig::num_aerosol_ids();
                   ++spec) { // check aerosol mmrs
                if ((q_aero_i[mode][spec](k) < 0) ||
                    (q_aero_c[mode][spec](k) < 0)) {
                  ++violation;
                  break;
                }
              }
            }
            if (violation > 0)
              break;
          }
          if (violation == 0) {
            for (int gas = 0; gas < AeroConfig::num_gas_ids();
                 ++gas) { // check gas mmrs
              if (q_gas[gas](k) < 0)
                ++violation;
              if (q_gas_avg[gas](k) < 0)
                ++violation;
            }
          }
        },
        violations);
    return (violations == 0);
  }
};

/// MAM4 column-wise diagnostic aerosol fields.
class Diagnostics final {
  // number of vertical levels
  int nlev_;

public:
  using ColumnView = haero::ColumnView;

  /// Creates a container for diagnostic variables on the specified number of
  /// vertical levels. All views must be set manually.
  explicit Diagnostics(int num_levels) : nlev_(num_levels) {}

  Diagnostics() = default; // use only for creating containers of Diagnostics!
  ~Diagnostics() = default;

  // these are used to populate containers of Diagnostics objects
  Diagnostics(const Diagnostics &rhs) = default;
  Diagnostics &operator=(const Diagnostics &rhs) = default;

  int num_levels() const { return nlev_; }

  /// Hygroscopicity is a modal mass-weighted average over all species
  /// in a mode
  ColumnView hygroscopicity[AeroConfig::num_modes()];

  /// Total dry particle diameter is a modal mass-weighted average over
  /// all species of interstitial AND cloudborne aerosols in a mode
  ColumnView dry_geometric_mean_diameter_total[AeroConfig::num_modes()];

  /// Dry particle diameter is a modal mass-weighted average over all species
  /// of interstitial aerosols in a mode
  ColumnView dry_geometric_mean_diameter_i[AeroConfig::num_modes()];

  /// Dry particle diameter is a modal mass-weighted average over all species
  /// of cloudborne aerosols in a mode
  ColumnView dry_geometric_mean_diameter_c[AeroConfig::num_modes()];

  /// Wet particle diameter is a modal mass-weighted average over all species
  /// of interstitial aerosols in a mode
  ColumnView wet_geometric_mean_diameter_i[AeroConfig::num_modes()];

  /// Wet particle diameter is a modal mass-weighted average over all species
  /// of cloudborne in a mode
  ColumnView wet_geometric_mean_diameter_c[AeroConfig::num_modes()];

  // Aerosol wet density
  ColumnView wet_density[AeroConfig::num_modes()];

  // Activation fraction
  ColumnView activation_fraction[AeroConfig::num_modes()];

  /// For gas-aerosol exchange process
  /// Uptake rate coefficient of H2SO4 gas, summed over all modes
  ColumnView uptkrate_h2so4;
  /// Ambient SOA gas equilib mixing rate (mol/mol at actual mw)
  ColumnView g0_soa_out;

  /// boolean indicating whether cloudborne aerosols are present in a cell
  haero::DeviceType::view_1d<bool> is_cloudy;

  /// Number of time substeps needed to converge in mam_soaexch_advance_in_time
  haero::DeviceType::view_1d<int> num_substeps;

  // Output variables for nucleate_ice process:
  // Ask experts for better names for: icenuc_num_hetfrz, icenuc_num_immfrz,
  // nihf
  // icenuc_num_depnuc,  icenuc_num_meydep output number conc of ice nuclei due
  // to heterogeneous freezing [1/m3]
  ColumnView icenuc_num_hetfrz;
  // niimm
  //  output number conc of ice nuclei due to immersion freezing (hetero nuc)
  //  [1/m3]
  ColumnView icenuc_num_immfrz;
  // nidep
  // output number conc of ice nuclei due to deposition nucleation (hetero nuc)
  // [1/m3]
  ColumnView icenuc_num_depnuc;
  // nimey
  // !output number conc of ice nuclei due to meyers deposition [1/m3]
  ColumnView icenuc_num_meydep;
  // number of activated aerosol for ice nucleation (homogeneous freezing only)
  // [#/kg]
  ColumnView num_act_aerosol_ice_nucle_hom;
  // number of activated aerosol for ice nucleation [#/kg]
  ColumnView num_act_aerosol_ice_nucle;
  // stratiform cloud fraction (called AST in F90 MAM4)
  ColumnView stratiform_cloud_fraction;
  // heterogenous freezing by immersion nucleation [cm^-3 s^-1]
  // in mam4 this is called frzimm
  ColumnView hetfrz_immersion_nucleation_tend;
  // heterogenous freezing by contact nucleation [cm^3 s^-1]
  // in mam4 this is called frzcnt
  ColumnView hetfrz_contact_nucleation_tend;
  // heterogenous freezing by deposition [cm^3 s^-1]
  // in mam4 this is called frzdep
  ColumnView hetfrz_depostion_nucleation_tend;
  // total bc number [#/cm3]
  ColumnView bc_num;
  // total dst1 number [#/cm3]
  ColumnView dst1_num;
  // total dst3 number [#/cm3]
  ColumnView dst3_num;
  // coated bc number [#/cm3]
  ColumnView bcc_num;
  // coated dst1 number [#/cm3]
  ColumnView dst1c_num;
  // coated dst3 number [#/cm3]
  ColumnView dst3c_num;
  // uncoated bc number [#/cm3]
  ColumnView bcuc_num;
  // uncoated dst1 number [#/cm3]
  ColumnView dst1uc_num;
  // uncoated dst3 number [#/cm3]
  ColumnView dst3uc_num;
  // interstitial bc number [#/cm3]
  ColumnView bc_a1_num;
  // interstitial dst1 number [#/cm3]
  ColumnView dst_a1_num;
  // interstitial dst3 number [#/cm3]
  ColumnView dst_a3_num;
  // cloud borne bc number [#/cm3]
  ColumnView bc_c1_num;
  // cloud borne dst1 number [#/cm3]
  ColumnView dst_c1_num;
  // cloud borne dst3 number [#/cm3]
  ColumnView dst_c3_num;
  // cloud borne bc number derived from fn [#/cm3]
  ColumnView fn_bc_c1_num;
  // cloud borne dst1 number derived from fn [#/cm3]
  ColumnView fn_dst_c1_num;
  // cloud borne dst3 number derived from fn [#/cm3]
  ColumnView fn_dst_c3_num;
  // interstitial aerosol number with D>500 nm [#/cm3]
  ColumnView na500;
  // total aerosol number with D>500 nm [#/cm3]
  ColumnView totna500;
  // Fractional occurance of immersion  freezing [fraction]
  ColumnView freqimm;
  // Fractional occurance of contact  freezing [fraction]
  ColumnView freqcnt;
  // Fractional occurance of deposition freezing [fraction]
  ColumnView freqdep;
  // Fractional occurance of mixed-phase clouds [fraction]
  ColumnView freqmix;
  // dust immersion  freezing rate [m-3s-1]
  ColumnView dstfrezimm;
  // dust contact    freezing rate [m-3s-1]
  ColumnView dstfrezcnt;
  // dust deposition freezing rate [m-3s-1]
  ColumnView dstfrezdep;
  // bc immersion  freezing rate [m-3s-1]
  ColumnView bcfrezimm;
  // bc contact    freezing rate [m-3s-1]
  ColumnView bcfrezcnt;
  // bc deposition freezing rate [m-3s-1]
  ColumnView bcfrezdep;
  // Activated Ice Number Concentration due to het immersion freezing in Mixed
  // Clouds [#/m3]
  ColumnView nimix_imm;
  // Activated Ice Number Concentration due to het contact freezing in Mixed
  // Clouds [#/m3]
  ColumnView nimix_cnt;
  // Activated Ice Number Concentration due to het deposition freezing in Mixed
  // Clouds [#/m3]
  ColumnView nimix_dep;
  // Activated Ice Number Concentration due to dst dep freezing in Mixed Clouds
  // [#/m3]
  ColumnView dstnidep;
  // Activated Ice Number Concentration due to dst cnt freezing in Mixed Clouds
  // [#/m3]
  ColumnView dstnicnt;
  // Activated Ice Number Concentration due to dst imm freezing in Mixed Clouds
  // [#/m3]
  ColumnView dstniimm;
  // Activated Ice Number Concentration due to bc dep freezing in Mixed Clouds
  // [#/m3]
  ColumnView bcnidep;
  // Activated Ice Number Concentration due to bc cnt freezing in Mixed Clouds
  // [#/m3]
  ColumnView bcnicnt;
  // Activated Ice Number Concentration due to bc imm freezing in Mixed Clouds
  // [#/m3]
  ColumnView bcniimm;
  // Ice Number Concentration due to het freezing in Mixed Clouds during 10-s
  // period [#/m3]
  ColumnView numice10s;
  // Ice Number Concentration due to imm freezing by dst in Mixed Clouds during
  // 10-s period [#/m3]
  ColumnView numimm10sdst;
  // Ice Number Concentration due to imm freezing by bc in Mixed Clouds during
  // 10-s period [#/m3]
  ColumnView numimm10sbc;

  // ************************************************************************
  // ********** Begin Convective Process Diagnostic Arrays ******************

  // INPUTS:
  // Values consumed by the convective processes to be filled upstream.
  // From hydrostatic_dry_dp to d_tracer_mixing_ratio_dt inclusive.
  // Delta pressure between interfaces for dry pressure [mb]
  ColumnView hydrostatic_dry_dp;
  // Deep cloud convective fraction [fraction]
  ColumnView deep_convective_cloud_fraction;
  // Shallow cloud convective fraction [fraction]
  ColumnView shallow_convective_cloud_fraction;

  // Deep cloud convective condensate [kg/kg]
  ColumnView deep_convective_cloud_condensate;
  // Shallow cloud convective condensate [kg/kg]
  // shallow_convective_cloud_condensate is in the
  // convproc process but not acutally used. There
  // is a note by "Shuaiqi Tang 2023.2.25" that
  // shwllow convection is not computed in the
  // Fortran version.
  ColumnView shallow_convective_cloud_condensate;

  // Deep convective precipitation production (grid avg) [kg/kg/s]
  ColumnView deep_convective_precipitation_production;
  // Shallow convective precipitation production (grid avg) [kg/kg/s]
  // Shallow convection is not currently computed by convproc,
  // See note with shallow_convective_cloud_condensate.
  ColumnView shallow_convective_precipitation_production;

  // rate of evaporation of falling precipitation [kg/kg/s].
  ColumnView evaporation_of_falling_precipitation;
  // Deep convective precipitation evaporation (grid avg) [kg/kg/s]
  ColumnView deep_convective_precipitation_evaporation;
  // Shallow convective precipitation evaporation (grid avg) [kg/kg/s]
  // Shallow convection is not currently computed by convproc,
  // See note with shallow_convective_cloud_condensate.
  ColumnView shallow_convective_precipitation_evaporation;

  // Shallow+Deep convective detrainment [kg/kg/s]
  ColumnView total_convective_detrainment;
  // Shallow convective detrainment [kg/kg/s]
  // Shallow convection is not currently computed by convproc,
  // See note with shallow_convective_cloud_condensate.
  ColumnView shallow_convective_detrainment;

  // Shallow convective ratio: [entrainment/(entrainment+detrainment)]
  // [fraction]
  // Shallow convection is not currently computed by convproc,
  // See note with shallow_convective_cloud_condensate.
  ColumnView shallow_convective_ratio;

  // Next three are "d(massflux)/dp" and are all positive [1/s]
  ColumnView mass_entrain_rate_into_updraft;
  ColumnView mass_entrain_rate_into_downdraft;
  ColumnView mass_detrain_rate_from_updraft;

  // Delta pressure between interfaces [mb]
  ColumnView delta_pressure;

  // Tracer mixing ratio (TMR) including water vapor [kg/kg]
  using ColumnTracerView = ekat::Unmanaged<typename DeviceType::view_2d<Real>>;
  ColumnTracerView tracer_mixing_ratio;

  // OUTPUTS:
  // Time tendency of tracer mixing ratio (TMR) [kg/kg/s]
  // This is the only output of convproc and is to be applied to the
  // array tracer_mixing_ratio to update.
  ColumnTracerView d_tracer_mixing_ratio_dt;
  // ********** End Convective Process Diagnostic Arrays ******************
  // ************************************************************************

  // ************************************************************************
  // ********** Begin Wet Deposition Diagnostic Arrays ******************

  // INPUTS:
  // All the inputs are shared with other processes and are already defined:
  //   deep_convective_cloud_fraction
  //   shallow_convective_cloud_fraction
  //   deep_convective_cloud_fraction
  //   shallow_convective_cloud_fraction
  //   deep_convective_precipitation_evaporation
  //   shallow_convective_precipitation_evaporation
  //   stratiform_cloud_fraction
  //   evaporation_of_falling_precipitation
  //   deep_convective_precipitation_production
  //   shallow_convective_precipitation_production
  //   deep_convective_cloud_condensate
  //   shallow_convective_cloud_condensate
  //   tracer_mixing_ratio
  //   d_tracer_mixing_ratio_dt

  // OUTPUTS:
  // aerosol wet deposition (interstitial) [kg/m2/s]
  ColumnView aerosol_wet_deposition_interstitial;
  // aerosol wet deposition (cloud water)  [kg/m2/s]
  ColumnView aerosol_wet_deposition_cloud_water;

  // ********** End Wet Deposition Diagnostic Arrays ******************
  // ************************************************************************
};

} // namespace mam4

#endif
