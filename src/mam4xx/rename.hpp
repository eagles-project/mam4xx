#ifndef MAM4XX_RENAME_HPP
#define MAM4XX_RENAME_HPP

#include <ekat/util/ekat_math_utils.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>

namespace mam4 {

using haero::max;
using haero::min;
using haero::sqrt;

namespace rename {

KOKKOS_INLINE_FUNCTION
void find_renaming_pairs(const int dest_mode_of_mode[AeroConfig::num_modes()],
                         int num_pairs) {
  // NOTE: original function signature
  // (nmodes, dest_mode_of_mode, &  // input
  // num_pairs, sz_factor, fmode_dist_tail_fac, v2n_lo_rlx, & // output
  // v2n_hi_rlx, ln_diameter_tail_fac, diameter_cutoff, &     // output
  // ln_dia_cutoff, diameter_threshold)

  // ------------------------------------------------------------------------
  // Find number of pairs which participates in the inter-mode transfer
  // ------------------------------------------------------------------------
  // Number of mode pairs allowed to do inter-mode particle transfer
  // (e.g. if we have a pair "mode_1<-->mode_2", mode_1 and mode_2 can
  // participate in the inter-mode aerosol particle transfer where particles
  // of the same species in mode_1 can be transferred to mode_2 and vice-versa)
  // ------------------------------------------------------------------------

  // // arguments (intent-ins)
  // integer, intent(in) :: nmodes               // total number of modes
  // integer, intent(in) :: dest_mode_of_mode(:) // array carry info about the
  // destination mode of a particular mode

  // // intent-outs
  // integer,  intent(out) :: num_pairs          // total number of pairs to be
  // found real(r8), intent(out) :: sz_factor(:), fmode_dist_tail_fac(:) //
  // precomputed factors to be used later [unitless] real(r8), intent(out) ::
  // v2n_lo_rlx(:), v2n_hi_rlx(:)         // relaxed volume to num high and low
  // ratio limits [m^-3] real(r8), intent(out) :: ln_diameter_tail_fac(:) // log
  // of diameter factor for distribution tail [unitless] real(r8), intent(out)
  // :: diameter_cutoff(:), ln_dia_cutoff(:) // cutoff (threshold) for deciding
  // the  do inter-mode transfer [m] real(r8), intent(out) ::
  // diameter_threshold(:)                // threshold diameter(99% of
  // cutoff)[m]

  const int nmodes = AeroConfig::num_modes();

  //  local variables
  int src_mode, dest_mode;

  // some parameters
  const Real sqrt_half = haero::sqrt(0.5);
  // (3^3): relaxing 3 * diameter, which makes it 3^3 for volume
  const Real frelax = 27.0;

  //  Let us assume there are none to start with
  num_pairs = 0;

  // if there can be no possible pairs, just return
  // if (all(dest_mode_of_mode(:)<=0)) return
  // TODO: will these indices be negative or just 0?
  // NOTE: is there a way to use std::all_of() here?
  // it doesn't look like ekat has its own version
  int dmode_sum = 0;
  for (int i = 0; i < nmodes; ++i)
  {
    dmode_sum += dest_mode_of_mode[i];
  }
  if (dmode_sum <= 0)
    return;

  // // Go through all the modes to find if we have at least one or more than
  // one pairs
  for (int imode = 0; imode < nmodes; imode++) {
    dest_mode = dest_mode_of_mode[imode]; // "destination" mode for mode "imode"

    // if dest_mode is <=0, transfer is not possible for this mode, cycle
    //    the loop for the next mode
    if (dest_mode <= 0) continue;

    src_mode = imode; //  transfer "src" mode is the current mode (i.e. imode)

    // ^^At this point, we know that particles can be transferred from the
    // "src_mode" to "dest_mode". "src_mode" is the current mode (i.e.
    // imode)

    // update number of pairs found so far
      num_pairs = num_pairs + 1; // increment npair

    // -------------------------------------------------------
    // Now precompute some common factors to be used later
    // -------------------------------------------------------

    // size factor for "src mode"
    // compute_size_factor(src_mode,  sz_factor);
    // size factor for "dest mode"
    // compute_size_factor(dest_mode, sz_factor);

  //    //
  //    ---------------------------------------------------------------------------------------------------------
  //    //  We compute few factors below for the "src_mode", which will be used
  //    for inter-mode particle transfer
  //    //
  //    ---------------------------------------------------------------------------------------------------------

  //    fmode_dist_tail_fac(src_mode) = sqrt_half/alnsg_amode(src_mode) //
  //    factor for computing distribution tails of the  "src mode"

  //    // compute volume to number high and low limits with relaxation
  //    coefficients (watch out for the repeated calculations)
  //    v2n_lo_rlx(src_mode) = vol_to_num_ratio(src_mode, dgnumlo_amode) *
  //    frelax v2n_hi_rlx(src_mode) = vol_to_num_ratio(src_mode, dgnumhi_amode)
  //    / frelax

  //    // A factor for computing diameter at the tails of the distribution
  //    ln_diameter_tail_fac(src_mode) = 3.0 * (alnsg_amode(src_mode)**2)

  //    // cutoff (based on geometric mean) for making decision to do inter-mode
  //    transfers
  //    // We took geometric mean of the participating modes (source and
  //    destination)
  //    // to find a cutoff or threshold from moving particles from the source
  //    to the
  //    // destination mode.
  //    diameter_cutoff(src_mode) = sqrt(   &
  //         dgnum_amode(src_mode)*exp(1.5*(alnsg_amode(src_mode)**2)) *   &
  //         dgnum_amode(dest_mode)*exp(1.5*(alnsg_amode(dest_mode)**2)) )

  //    ln_dia_cutoff(src_mode) = log(diameter_cutoff(src_mode)) // log of
  //    cutoff diameter_threshold(src_mode) = 0.99*diameter_cutoff(src_mode) //
  //    99% of the cutoff
  } // end for(imode)
}

KOKKOS_INLINE_FUNCTION
void compute_dryvol_change_in_src_mode() {}
// NOTE: original function signature
// (nmode, nspec, dest_mode_of_mode, & // input
// q_mmr, q_del_growth, & // input
// dryvol, deldryvol )

KOKKOS_INLINE_FUNCTION
void do_inter_mode_transfer() {}
// NOTE: original function signature
// (nmode, nspec, dest_mode_of_mode, &
// iscldy, v2nlorlx, v2nhirlx, dryvol_a, dryvol_c, deldryvol_a, deldryvol_c, &
// sz_factor, fmode_dist_tail_fac, ln_diameter_tail_fac, ln_dia_cutoff,
// diameter_threshold, & qaer_cur, qnum_cur, qaercw_cur, qnumcw_cur)

} // namespace rename

/// @class Rename
/// This class implements MAM4's rename parameterization.
class Rename {
public:
  // rename-specific configuration
  struct Config {
    // default constructor -- sets default values for parameters

    Config() {}

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  Config config_;

public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 rename"; }

  // init -- initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig &aero_config,
            const Config &rename_config = Config()) {
    // Set rename-specific config parameters.
    config_ = rename_config;

  } // end(init)

  // NOTE: it looks like this will probably correspond to mam_rename_1subarea()
  // in the fortran refactor code
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atmosphere,
                          const Prognostics &prognostics,
                          const Diagnostics &diagnostics,
                          const Tendencies &tendencies) const {

    // NOTE: original subroutine signature
    // (iscldy, dest_mode_of_mode, nmode, &
    //        qnum_cur, qaer_cur, qaer_del_grow4rnam, qnumcw_cur,         &
    //        qaercw_cur,        qaercw_del_grow4rnam                    )

    //     use modal_aero_data, only: ntot_amode

    //     implicit none
    //     //
    //     -------------------------------------------------------------------------------------
    //     //  DESCRIPTION:
    //     //
    //     -------------------------------------------------------------------------------------
    //     //  Computes TMR (tracer mixing ratio) tendencies for "mode renaming"
    //     (i.e. transferring
    //     //  particles from one mode to another or "renaming" mode of the
    //     particles) during a
    //     //  continuous growth process.
    //     //
    //     //  Currently this transfers number and mass (and surface) from the
    //     aitken to accumulation
    //     //  mode after gas condensation or stratiform-cloud aqueous chemistry
    //     //  (convective cloud aqueous chemistry not yet implemented)
    //     //
    //     -------------------------------------------------------------------------------------

    //     // input
    //     logical,  intent(in) :: iscldy                      // true if
    //     sub-area is cloudy integer,  intent(in) ::
    //     dest_mode_of_mode(max_mode) // destination mode of a mode integer,
    //     intent(in) :: nmode                       // number of modes
    //     real(r8), intent(in) :: qaer_del_grow4rnam(1:max_aer, 1:max_mode) //
    //     growth in aerosol molar mixing ratio [kmol/kmol-air] real(r8),
    //     intent(in), optional :: qaercw_del_grow4rnam(1:max_aer, 1:max_mode)//
    //     growth in aerosol molar mixing ratio (cld borne) [kmol/kmol-air]

    //     // output
    //     real(r8), intent(inout) :: qnum_cur(1:max_mode)           // aerosol
    //     number mixing ratio [#/kmol-air] real(r8), intent(inout) ::
    //     qaer_cur(1:max_aer, 1:max_mode)// aerosol molar mixing ratio
    //     [kmol/kmol-air] real(r8), intent(inout), optional ::
    //     qnumcw_cur(1:max_mode)           // aerosol number mixing ratio (cld
    //     borne) [#/kmol-air] real(r8), intent(inout), optional ::
    //     qaercw_cur(1:max_aer, 1:max_mode)// aerosol molar mixing ratio (cld
    //     borne) [kmol/kmol-air]

    //     //  local variables
    //     integer :: npair // number of pairs in different modes for the
    //     transfer

    //     real(r8) :: deldryvol_a(ntot_amode)   // change in dry volume
    //     [m3/kmol-air] real(r8) :: deldryvol_c(ntot_amode)   // change in dry
    //     volume (cld borne)[m3/kmol-air] real(r8) :: diameter_cutoff(max_mode)
    //     // cutoff for threshold [m] real(r8) :: diameter_threshold(max_mode)
    //     // Threshold to decide arosol transfer (99% of cutoff) [m] real(r8)
    //     :: dryvol_a(ntot_amode)      // dry volume [m3/kmol-air] real(r8) ::
    //     dryvol_c(ntot_amode)      // dry volume (cld borne)[m3/kmol-air]
    //     real(r8) :: sz_factor(ntot_amode)     // size factor for each mode
    //     [unitless] real(r8) :: fmode_dist_tail_fac(ntot_amod// ) // tail
    //     distribution factor for each mode [unitless] real(r8) ::
    //     lndiameter_cutoff(max_mode) // log of diamter cutoff [m] real(r8) ::
    //     ln_diameter(max_mode)     // log of diameter [m] real(r8) ::
    //     v2nhirlx(ntot_amode), v2nlorlx(ntot_amode) // high and low volume to
    //     num ratios[m^-3]

    //     //
    //     ------------------------------------------------------------------------
    //     // Find mapping between different modes, so that we can move aerosol
    //     // particles from one mode to another
    //     //
    //     ------------------------------------------------------------------------

    //     // FIXME: All the arrays in find_renaming_pairs subroutine call
    //     should be
    //     // initialized to HUGE or NaNs as they are partially populated

    //     // Find (src->destination) pairs of modes (e.g., if only mode #1 and
    //     mode #2 can participate in the
    //     // inter-mode transfer, number of pairs will be 1 and so on) which
    //     can participate in
    //     // inter-mode species transfer

    //     call find_renaming_pairs (ntot_amode, dest_mode_of_mode, & // input
    //          npair, sz_factor, fmode_dist_tail_fac, v2nlorlx, &    // output
    //          v2nhirlx, ln_diameter, diameter_cutoff, &             // output
    //          lndiameter_cutoff, diameter_threshold)                // output

    //     if (npair <= 0) return // if no transfer required, return

    //     // Interstitial aerosols: Compute initial (before growth) aerosol dry
    //     volume and
    //     // also the growth in dryvolume of the "src" mode
    //     call compute_dryvol_change_in_src_mode(ntot_amode, naer,
    //     dest_mode_of_mode,  & // input
    //          qaer_cur, qaer_del_grow4rnam, & // input
    //          dryvol_a, deldryvol_a                 ) // output

    //     // Cloudborne aerosols: Compute initial (before growth) aerosol dry
    //     volume and
    //     // also the growth in dryvolume of the "src" mode
    //     if (iscldy) then
    //        call compute_dryvol_change_in_src_mode(ntot_amode, naer,
    //        dest_mode_of_mode,  & // input
    //             qaercw_cur, qaercw_del_grow4rnam, & // input
    //             dryvol_c, deldryvol_c                     ) // output
    //     endif

    //     // Find fractions (mass and number) to transfer and complete the
    //     transfer call do_inter_mode_transfer(ntot_amode, naer,
    //     dest_mode_of_mode, &                         // input
    //          iscldy, v2nlorlx, v2nhirlx, dryvol_a, dryvol_c, deldryvol_a,
    //          deldryvol_c, &           // input sz_factor,
    //          fmode_dist_tail_fac, ln_diameter, lndiameter_cutoff,
    //          diameter_threshold, & // input qaer_cur, qnum_cur, qaercw_cur,
    //          qnumcw_cur ) // output
  }

private:
}; // end(compute_tendencies)

} // namespace mam4

#endif
