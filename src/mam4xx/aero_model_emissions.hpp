#ifndef MAM4XX_AERO_MODEL_EMISSIONS_HPP
#define MAM4XX_AERO_MODEL_EMISSIONS_HPP

#include <haero/atmosphere.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/gas_chem_mechanism.hpp>
#include <mam4xx/mam4_types.hpp>

namespace mam4 {
namespace aero_model_emissions {
// // number of invariants
// constexpr int nfs = mam4::gas_chemistry::nfs;
// static const int num_tracer_cnst = 4;

// using View1D = DeviceType::view_1d<Real>;
// using View2D = DeviceType::view_2d<Real>;

// struct Config {
//   // FIXME: BAD CONSTANTS!
//   // conversion factor for Pascals to dyne/cm^2
//   Real Pa_xfac = 10.0;
//   // presumably, the boltzmann constant, in CGS units
//   Real boltz_cgs = 0.13806500000000001E-015;
//   // NOTE: the indices (xyz_ndx) are translated to zero-indexed in the
//   // constructor
//   bool has_n2 = true;
//   int m_ndx = 0;
//   int n2_ndx = 1;
//   bool has_o2 = true;
//   int o2_ndx = 2;
//   bool has_h2o = true;
//   int h2o_ndx = 3;
//   // FIXME: this will likely need to be changed to match the order of how the
//   // "cnst_offline" species are passed in
//   int coff_idx_list[num_tracer_cnst] = {7, 4, 5, 6};

//   Config() = default;
//   Config(const Config &) = default;
//   ~Config() = default;
//   Config &operator=(const Config &) = default;
// };

// // NOTE: vmr isn't actually used in the fortran code, but I've kept it here to
// // keep the function signature the same
// KOKKOS_INLINE_FUNCTION
// void setinv_single_level(Real invariants[nfs], const Real tfld,
//                          const Real h2ovmr,
//                          const Real vmr[AeroConfig::num_gas_phase_species()],
//                          const Real pmid,
//                          const Real cnst_offline[num_tracer_cnst],
//                          const Config config_) {
//   // -----------------------------------------------------------------
//   //  set the invariant densities (molecules/cm**3)
//   // -----------------------------------------------------------------

//   for (int i = 0; i < nfs; ++i) {
//     invariants[i] = 0.0;
//   }

//   const int m_idx = config_.m_ndx;
//   // NOTE: these species index are <xyz>_ndx =  <xyz_fortran>_ndx - 1
//   // NOTE: invariants are in cgs density units. => the pmid array is in pascals
//   // and must be multiplied by 10 (Pa_xfac) to yield dynes/cm**2.
//   invariants[m_idx] = config_.Pa_xfac * pmid / (config_.boltz_cgs * tfld);
//   const Real inv_m = invariants[m_idx];
//   // FIXME: BAD CONSTANTS!!
//   if (config_.has_n2) {
//     invariants[config_.n2_ndx] = 0.79 * inv_m;
//   }
//   if (config_.has_o2) {
//     invariants[config_.o2_ndx] = 0.21 * inv_m;
//   }
//   if (config_.has_h2o) {
//     invariants[config_.h2o_ndx] = h2ovmr * inv_m;
//   }
//   const auto c_off = cnst_offline;
//   for (int i = 0; i < num_tracer_cnst; ++i) {
//     invariants[config_.coff_idx_list[i]] = c_off[i] * inv_m;
//   }

//   // NOTE: not porting for now
//   // Writing out diagnostics
//   // do ifs = 1,nfs
//   //   tmp_out(:ncol,:) =  invariants(:ncol,:,ifs)
//   //   call outfld( trim(inv_lst(ifs))//'_dens', tmp_out(:ncol,:), ncol, lchnk )
//   //   tmp_out(:ncol,:) =  invariants(:ncol,:,ifs) / invariants(:ncol,:,m_ndx)
//   //   call outfld( trim(inv_lst(ifs))//'_vmr',  tmp_out(:ncol,:), ncol, lchnk )
//   // enddo
// } // end setinv_single_level()

// KOKKOS_INLINE_FUNCTION
// void setinv(const ThreadTeam &team, const ColumnView invariants[nfs],
//             const ColumnView &tfld, const ColumnView &h2ovmr,
//             const ColumnView vmr[AeroConfig::num_gas_phase_species()],
//             const ColumnView cnst_offline[num_tracer_cnst],
//             const ColumnView &pmid) {

//   Config setinv_config_;
//   constexpr int nk = mam4::nlev;

//   Kokkos::parallel_for(
//       Kokkos::TeamThreadRange(team, nk), KOKKOS_LAMBDA(int k) {
//         const Real tfld_k = tfld(k);
//         const Real h2ovmr_k = h2ovmr(k);
//         const Real pmid_k = pmid(k);
//         Real invariants_k[nfs];
//         const int nspec = AeroConfig::num_gas_phase_species();
//         Real vmr_k[nspec];
//         for (int i = 0; i < nspec; ++i) {
//           vmr_k[i] = vmr[i](k);
//         }
//         Real cnst_offline_k[num_tracer_cnst];
//         for (int i = 0; i < num_tracer_cnst; ++i) {
//           cnst_offline_k[i] = cnst_offline[i](k);
//         }
//         setinv_single_level(invariants_k, tfld_k, h2ovmr_k, vmr_k, pmid_k,
//                             cnst_offline_k, setinv_config_);
//         for (int i = 0; i < nfs; ++i) {
//           invariants[i](k) = invariants_k[i];
//         }
//       }); // end kokkos::parfor(k)
// } // end setinv_nlev()

// KOKKOS_INLINE_FUNCTION
// void setinv(const ThreadTeam &team, const ColumnView invariants[nfs],
//             const ColumnView &tfld, const ColumnView &h2ovmr, const View2D &vmr,
//             const View2D &cnst_offline, const ColumnView &pmid) {

//   Config setinv_config_;
//   constexpr int nk = mam4::nlev;

//   Kokkos::parallel_for(
//       Kokkos::TeamThreadRange(team, nk), KOKKOS_LAMBDA(int k) {
//         const Real tfld_k = tfld(k);
//         const Real h2ovmr_k = h2ovmr(k);
//         const Real pmid_k = pmid(k);
//         Real invariants_k[nfs];
//         View1D vmr_k = Kokkos::subview(vmr, k, Kokkos::ALL());
//         View1D cnst_offline_k = Kokkos::subview(cnst_offline, k, Kokkos::ALL());

//         setinv_single_level(invariants_k, tfld_k, h2ovmr_k, vmr_k.data(),
//                             pmid_k, cnst_offline_k.data(), setinv_config_);

//         for (int i = 0; i < nfs; ++i) {
//           invariants[i](k) = invariants_k[i];
//         }
//       }); // end kokkos::parfor(k)
// } // end setinv_nlev()
} // namespace aero_model_emissions
} // namespace mam4
#endif
