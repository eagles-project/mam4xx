#ifndef MAM4XX_RENAME_HPP
#define MAM4XX_RENAME_HPP

#include <ekat/util/ekat_math_utils.hpp>

#include <haero/atmosphere.hpp>
#include <haero/math.hpp>

#include <mam4xx/aero_config.hpp>
#include <mam4xx/conversions.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/utils.hpp>

namespace mam4 {

using haero::max;
using haero::min;
using haero::sqrt;
using haero::erfc;
using haero::log;
using Constants = haero::Constants;
using haero::square;

namespace rename {

// FIXME: there's almost certainly a better way to do this
// FIXME: should this go here? Other option would be to put it in
    // do_inter_mode_transfer(), but that's not where it is in the fortran mam4
const Real smallest_dryvol_value = 1.0e-25; // BAD CONSTANT!

KOKKOS_INLINE_FUNCTION
void compute_dryvol_change_in_src_mode(
    const int nmode,              // in
    const int nspec,              // in
    const int *dest_mode_of_mode, // in
    const Real q_mmr[AeroConfig::num_modes()]
                    [AeroConfig::num_aerosol_ids()], // in
    const Real q_del_growth[AeroConfig::num_modes()]
                           [AeroConfig::num_aerosol_ids()], // in
    const Real mass_2_vol[AeroConfig::num_aerosol_ids()],   // in
    Real dryvol[AeroConfig::num_modes()],
    Real deldryvol[AeroConfig::num_modes()] // out
) {
  for (int m = 0; m < nmode; ++m) {
    int dest_mode = dest_mode_of_mode[m];

    if (dest_mode > 0) {

      // For each mode, we compute a dry volume by combining (accumulating)
      // mass/density for each species in that mode.
      //  conversion from mass to volume is accomplished by multiplying with
      //  precomputed "mass_2_vol" factor

      // s_spec_ind = 1     !start species index for this mode [These will be
      // subroutine args] e_spec_ind = nspec !end species index for this mode

      // initialize tmp accumulators
      Real tmp_dryvol = 0.0;     // dry volume accumulator
      Real tmp_del_dryvol = 0.0; // dry volume growth(change) accumulator

      // Notes on mass_2_vol factor: Units:[m3/kmol-species]; where kmol-species
      // is the amount of a species "s" This factor is obtained by
      // (molecular_weight/density) of a species. That is, [ (g/mol-species) /
      // (kg-species/m3) ]; where molecular_weight has units [g/mol-species] and
      // density units are [kg-species/m3] which results in the units of
      // m3/kmol-species

      for (int ispec = 0; ispec < nspec; ++ispec) {
        // Multiply by mass_2_vol[m3/kmol-species] to convert
        // q_mmr[kmol-species/kmol-air]) to volume units[m3/kmol-air]
        tmp_dryvol += q_mmr[m][ispec] * mass_2_vol[ispec];
        // accumulate the "growth" in volume units as well
        tmp_del_dryvol += q_del_growth[m][ispec] * mass_2_vol[ispec];
      }

      dryvol[m] =
          tmp_dryvol - tmp_del_dryvol; // This is dry volume before the growth
      deldryvol[m] = tmp_del_dryvol;   // change in dry volume due to growth
    }
  }
}

KOKKOS_INLINE_FUNCTION
Real total_inter_cldbrn() {
  Real out = 1;
  return out;
}

KOKKOS_INLINE_FUNCTION
Real total_inter_cldbrn(const bool &iscloudy, const int &imode,
                        const Real interstitial[AeroConfig::num_modes()],
                        const Real cloudborne[AeroConfig::num_modes()]) {
  // NOTE: original function signature
  // (iscldy, imode, interstitial, cldbrn)

  // // compute total (dry volume or number) of interstitial and cloud borne
  // species

  // logical,  intent(in) :: iscldy       // TRUE, if a cell has cloud
  // integer,  intent(in) :: imode
  // real(r8), intent(in) :: interstitial(:) // interstitial part [unit depends on the input]
  // real(r8), intent(in), optional :: cldbrn(:) // cloudborne part [unit depends on the input]

  // // return value
  // real(r8) :: total

  // if there is no cloud, total is just the interstitial value
  Real total = interstitial[imode];

  if (iscloudy) {
    // FIXME: is this needed?
    // if(.not.present(cldbrn))then
    //    call endrun("If a grid cell is cloudy, cloud borne aerosol values
    //    must be present:"//errmsg(__FILE__,__LINE__))
    // end if
    total = total + cloudborne[imode];
  }
  return total;
}

// TODO: this is a placeholder until we create a proper unit test
KOKKOS_INLINE_FUNCTION
void compute_before_growth_dryvol_and_num() {}

KOKKOS_INLINE_FUNCTION
void compute_before_growth_dryvol_and_num(
    // in
    const int &iscloudy, const bool &src_mode,
    const Real dryvol_i[AeroConfig::num_modes()],
    const Real dryvol_c[AeroConfig::num_modes()],
    Real qnum_cur[AeroConfig::num_modes()],
    Real qnumcw_cur[AeroConfig::num_modes()],
    const Real &v2nlo, const Real &v2nhi,
    // out
    Real &bef_grwth_dryvol, Real &bef_grwth_dryvolbnd, Real &bef_grwth_numbnd) {
  // NOTE: original function signature
  // (iscloudy, src_mode, dryvol_a, dryvol_c, & // input
  // qnum_cur, qnumcw_cur, v2nhi, v2nlo, // input
  // bef_grwth_dryvol, bef_grwth_dryvolbnd, bef_grwth_numbnd)

  // Compute total(i.e. cloud borne and interstitial) of dry volume (before
  // growth)
  //  and delta in dry volume in the source mode [units: (m3 of species)/(kmol
  // of air)]
  // NOTE(mam4 refactor): cloudborne input can be optional, so we are sending
  // "src_mode" as a argument
  // as we cannot reference a member of an optional array if it is not present
  // NOTE: as long as dryvol_i(src_mode) is initialized to 0 when that mode is
  // "not cloudy", this function call is a one-liner. e.g.,
  // pregrowth_dryvol = dryvol_i[src_mode] + dryvol_c[src_mode];
  // at worst, we could give it an if(dryvol_c[src_mode] > 0) statement
  bef_grwth_dryvol = total_inter_cldbrn(iscloudy, src_mode, dryvol_i, dryvol_c);

  // FIXME: is it feasible that pregrowth_dryvol would be smaller than 1e-25?
  bef_grwth_dryvolbnd = haero::max(bef_grwth_dryvol, smallest_dryvol_value);

  // Compute total before growth number [units: #/kmol-air]
  Real bef_grwth_num =
      total_inter_cldbrn(iscloudy, src_mode, qnum_cur, qnumcw_cur);
  bef_grwth_num = max(0.0, bef_grwth_num); // bound to have minimum of 0

  // // bound number within min and max of the source mode
  bef_grwth_numbnd = utils::min_max_bound(bef_grwth_dryvolbnd * v2nhi, // min value
                                   bef_grwth_dryvolbnd * v2nlo, // max value
                                   bef_grwth_num); // input
}

//FIXME: check if a function to compute mode_diameter exist. 
KOKKOS_INLINE_FUNCTION
Real mode_diameter(const Real volume, const Real number, const Real size_factor)
{
    // compute diameter inputs:
    // volume      ![m3]
    // number      ![#/kmol-air]
    // size_factor ![unitless]
  // BAD CONSTANT 
 Real onethird = Real(1.0)/Real(3.0);  

 return pow(volume/(number*size_factor),onethird); 
} // end mode_diameter


KOKKOS_INLINE_FUNCTION
void compute_tail_fraction(const Real diameter, 
                           const Real log_dia_cutoff,
                           const Real tail_dist_fac,
                           const Real log_dia_tail_fac, // input
                           Real& tail_fraction
                           ) {
  // Compute tail fraction to be used for inter-mode species transfer
  // rename use present function for this if statement.
  // However, we will check if log_dia_tail_fac is bigger than zero.
  // Thus, if we do not want to include it we set its values to -1. 
  // in current implementation log_dia_tail_fac is not present. 
  const Real log_diameter  = log(diameter) + min(Real(0), log_dia_tail_fac);
  const Real tail = (log_dia_cutoff - log_diameter) * tail_dist_fac;
  // erfc error function
  // FIXME: check that we are using same function than E3SM.  
  tail_fraction = Real(0.5) * erfc(tail);

} // end compute_tail_fraction


KOKKOS_INLINE_FUNCTION
void compute_xfer_fractions(const Real bef_grwth_dryvol,
                            const Real aft_grwth_dryvol,
                            const Real bef_grwth_tail_fr_vol,
                            const Real aft_grwth_tail_fr_vol, // in
                            const Real aft_grwth_tail_fr_num,
                            const Real bef_grwth_tail_fr_num, // in
                            bool & is_xfer_frac_zero, //out
                            Real & xfer_vol_frac,
                            Real & xfer_num_frac //out
                            ){

    // BAD CONSTANT
    //1-eps (this number is little less than 1, e.g. 0.99) // FIXME: this comment is nonsense
    const Real xferfrac_max = 0.99; //1.0 - 10.0*epsilon(1.0_r8) ; 
    // assume we have fractions to transfer, so we will not skip the rest of the calculations
    is_xfer_frac_zero = false;
    const Real zero = 0.0;

    // transfer fraction is difference between new and old tail-fractions
    const Real volume_fraction = aft_grwth_tail_fr_vol*aft_grwth_dryvol - bef_grwth_tail_fr_vol*bef_grwth_dryvol;

    if (volume_fraction <= zero ) {
      is_xfer_frac_zero = true;
      return; 
     } 

    xfer_vol_frac = min(volume_fraction, aft_grwth_dryvol) / aft_grwth_dryvol;
    xfer_vol_frac = min(xfer_vol_frac, xferfrac_max) ;
    xfer_num_frac = aft_grwth_tail_fr_num - bef_grwth_tail_fr_num; 

    // transfer fraction for number cannot exceed that of mass
    xfer_num_frac = max(zero, min(xfer_num_frac, xfer_vol_frac));

} // end compute_xfer_fractions

KOKKOS_INLINE_FUNCTION
void do_num_and_mass_transfer(const int src_mode, const int dest_mode,
                              const Real xfer_vol_frac,
                              const Real xfer_num_frac, // input
                              Real qaer[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
                              Real qnum[AeroConfig::num_modes()]) {
  // compute changes to number and species masses
  const Real num_trans = qnum[src_mode] * xfer_num_frac;
  qnum[src_mode] -= num_trans;
  qnum[dest_mode] += num_trans;

  for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
    const Real vol_trans = qaer[src_mode][ispec] * xfer_vol_frac;
    qaer[src_mode][ispec] -= vol_trans;
    qaer[dest_mode][ispec] += vol_trans;
  }
} // end do_num_and_mass_transfer

// TODO: this is a placeholder until we create a proper unit test
KOKKOS_INLINE_FUNCTION
void do_inter_mode_transfer() {}

KOKKOS_INLINE_FUNCTION
void do_inter_mode_transfer(
    const int dest_mode_of_mode[AeroConfig::num_modes()],
    const bool &iscloudy,
    // volume to number relaxation limits [m^-3]
    const Real v2nlorlx[AeroConfig::num_modes()],
    const Real v2nhirlx[AeroConfig::num_modes()],
    const Real sz_factor[AeroConfig::num_modes()],
    const Real fmode_dist_tail_fac[AeroConfig::num_modes()],
    const Real ln_diameter_tail_fac[AeroConfig::num_modes()],
    const Real ln_dia_cutoff[AeroConfig::num_modes()],
    const Real diameter_threshold[AeroConfig::num_modes()],
    const Real dgnum_amode[AeroConfig::num_modes()],
    // dry volume [m3/kmol-air]
    const Real dryvol_a[AeroConfig::num_modes()],
    const Real dryvol_c[AeroConfig::num_modes()],
    const Real deldryvol_a[AeroConfig::num_modes()],
    const Real deldryvol_c[AeroConfig::num_modes()],
    Real qaer_cur[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
    // aerosol number mixing ratios [#/kmol-air]
    Real qnum_cur[AeroConfig::num_modes()],
    Real qaercw_cur[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
    Real qnumcw_cur[AeroConfig::num_modes()]) 
{
  // NOTE: original function signature
  // (nmode, nspec, dest_mode_of_mode, &
  // iscloudy, v2nlorlx, v2nhirlx, dryvol_a, dryvol_c, deldryvol_a, deldryvol_c,
  // & sz_factor, fmode_dist_tail_fac, ln_diameter_tail_fac, ln_dia_cutoff,
  // diameter_threshold, & qaer_cur, qnum_cur, qaercw_cur, qnumcw_cur)

  // local variables
  const int nmodes = AeroConfig::num_modes();
  int src_mode, dest_mode;

  Real bef_grwth_dryvol;    // [m3/kmol-air]
  Real bef_grwth_dryvolbnd; // [m3/kmol-air]
  Real bef_grwth_numbnd;    //  [#/kmol-air]

  const Real zero = 0;

  // Loop through the modes and do the transfer
  for (int imode = 0; imode < nmodes; ++imode) {
    src_mode = imode;                     // source mode
    dest_mode = dest_mode_of_mode[imode]; // destination mode

    // if destination mode doesn't exist for the source mode, cycle loop
    if (dest_mode <= 0)
      continue;

    // compute before growth dry volume and number
    compute_before_growth_dryvol_and_num(
        // in
        iscloudy, src_mode, dryvol_a, dryvol_c, qnum_cur, qnumcw_cur,
        v2nlorlx[src_mode], v2nhirlx[src_mode],
        // out
        bef_grwth_dryvol, bef_grwth_dryvolbnd, bef_grwth_numbnd);

    // change (delta) in dryvol
    const Real dryvol_del =
        total_inter_cldbrn(iscloudy, src_mode, deldryvol_a, deldryvol_c);

    // Total dry volume after growth (add delta growth)
    Real aft_grwth_dryvol = bef_grwth_dryvol + dryvol_del;

    // Skip inter-mode transfer for this mode if dry after growth is ~ 0
    if (aft_grwth_dryvol <= smallest_dryvol_value) {
      break;
    }
    // // Total dry volume after growth (add delta growth)

    // compute before growth diameter
    Real bef_grwth_diameter = mode_diameter(
        bef_grwth_dryvolbnd, bef_grwth_numbnd, sz_factor[src_mode]);

    // if the before growth diameter is more than the threshold
    // (diameter_threshold), we restrict diameter to the threshold and change
    // dry volume accordingly
    if (bef_grwth_diameter > diameter_threshold[src_mode]) {
      //  this revised volume corresponds to bef_grwth_diameter ==
      //    diameter_threshold, and same number conc
      bef_grwth_dryvol = bef_grwth_dryvol * cube(diameter_threshold[src_mode] /
                                                 bef_grwth_diameter);
      bef_grwth_diameter = diameter_threshold[src_mode];
    }

    // FIXME
    //  BAD CONSTANT
    if ((aft_grwth_dryvol - bef_grwth_dryvol) <=
        Real(1.0e-6) * bef_grwth_dryvolbnd) {
      break;
    }

    // Compute after growth diameter; if it is less than the "nominal" or
    // "base" diameter for the source mode, skip inter-mode transfer
    Real aft_grwth_diameter =
        mode_diameter(aft_grwth_dryvol, bef_grwth_numbnd, sz_factor[src_mode]);

    if (aft_grwth_diameter <= dgnum_amode[src_mode]) {
      break;
    }

    // compute before growth number fraction in the tail

    // call compute_tail_fraction(bef_grwth_diameter,ln_dia_cutoff(src_mode),
    // log_dia_tail_fac is not presented in original call to
    // compute_tail_fraction. Thus do not no include its value
    const Real default_log_dia_tail_fac = Real(-1);
    Real bef_grwth_tail_fr_num = zero;
    compute_tail_fraction(bef_grwth_diameter, ln_dia_cutoff[src_mode],
                          fmode_dist_tail_fac[src_mode],
                          default_log_dia_tail_fac,
                          bef_grwth_tail_fr_num // out
    );

    // compute before growth volume (or mass) fraction in the tail
    Real bef_grwth_tail_fr_vol = zero;
    compute_tail_fraction(bef_grwth_diameter, ln_dia_cutoff[src_mode],
                          fmode_dist_tail_fac[src_mode],
                          ln_diameter_tail_fac[src_mode],
                          bef_grwth_tail_fr_vol // out
    );

    // compute after growth number fraction in the tail
    // log_dia_tail_fac is not presented in original call to
    // compute_tail_fraction.
    // Thus do not no include its value
    Real aft_grwth_tail_fr_num = zero;
    compute_tail_fraction(aft_grwth_diameter, ln_dia_cutoff[src_mode],
                          fmode_dist_tail_fac[src_mode],
                          default_log_dia_tail_fac,
                          aft_grwth_tail_fr_num // out
    );

    // compute after growth volume (or mass) fraction in the tail
    Real aft_grwth_tail_fr_vol = zero;
    compute_tail_fraction(aft_grwth_diameter, ln_dia_cutoff[src_mode],
                          fmode_dist_tail_fac[src_mode],
                          ln_diameter_tail_fac[src_mode],
                          aft_grwth_tail_fr_vol // out
    );

    // compute transfer fraction (volume and mass) - if less than zero, cycle
    // loop
    bool is_xfer_frac_zero = false;
    Real xfer_vol_frac = zero;
    Real xfer_num_frac = zero;
    compute_xfer_fractions(bef_grwth_dryvol, aft_grwth_dryvol,
                           bef_grwth_tail_fr_vol,
                           aft_grwth_tail_fr_vol, // in
                           aft_grwth_tail_fr_num,
                           bef_grwth_tail_fr_num, // in
                           is_xfer_frac_zero,     // out
                           xfer_vol_frac,
                           xfer_num_frac // out
    );

    if (is_xfer_frac_zero) {
      break;
    }

    // do the transfer for the interstitial species
    do_num_and_mass_transfer(src_mode, dest_mode, xfer_vol_frac,
                             xfer_num_frac, // input
                             qaer_cur, qnum_cur);
    if (iscloudy) {
      do_num_and_mass_transfer(src_mode, dest_mode, xfer_vol_frac,
                               xfer_num_frac, // input
                               qaercw_cur, qnumcw_cur);
    }
  } // end for(imode)
} // end do_inter_mode_transfer()

KOKKOS_INLINE_FUNCTION
void find_renaming_pairs(
    int *dest_mode_of_mode,                             // in
    Real sz_factor[AeroConfig::num_modes()],            // out
    Real fmode_dist_tail_fac[AeroConfig::num_modes()],  // out
    Real v2n_lo_rlx[AeroConfig::num_modes()],           // out
    Real v2n_hi_rlx[AeroConfig::num_modes()],           // out
    Real ln_diameter_tail_fac[AeroConfig::num_modes()], // out
    Real num_pairs,                                     // out
    Real diameter_cutoff[AeroConfig::num_modes()],      // out
    Real ln_dia_cutoff[AeroConfig::num_modes()],
    Real diameter_threshold[AeroConfig::num_modes()], // out
    Real mass_2_vol[AeroConfig::num_aerosol_ids()]) {
  const Real sqrt_half = haero::sqrt(0.5);
  // (3^3): relaxing 3 * diameter, which makes it 3^3 for volume
  const Real frelax = 27.0;

  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    const int dest_mode =
        dest_mode_of_mode[m]; // "destination" mode for mode "imode"
    const Real alnsg_amode = log(modes(m).mean_std_dev);
    // FIXME : check where _sz_factor is used and try to use function calls
    // from conversions.hpp
    sz_factor[m] = Constants::pi_sixth * exp(4.5 * square(alnsg_amode));
    // factor for computing distribution tails of the  "src mode"
    fmode_dist_tail_fac[m] = sqrt_half / alnsg_amode;
    // compute volume to number high and low limits with relaxation
    // coefficients (watch out for the repeated calculations)
    v2n_lo_rlx[m] = Real(1) /
                    conversions::mean_particle_volume_from_diameter(
                        modes(m).min_diameter, modes(m).mean_std_dev) *
                    frelax;
    v2n_hi_rlx[m] = Real(1) /
                    conversions::mean_particle_volume_from_diameter(
                        modes(m).max_diameter, modes(m).mean_std_dev) /
                    frelax;
    // A factor for computing diameter at the tails of the distribution
    ln_diameter_tail_fac[m] = Real(3.0) * square(alnsg_amode);

    const int src_mode =
        m; //  transfer "src" mode is the current mode (i.e. imode)

    // ^^At this point, we know that particles can be transferred from the
    // "src_mode" to "dest_mode". "src_mode" is the current mode (i.e.
    // imode)

    if (dest_mode > 0) {
      // update number of pairs found so far
      num_pairs += num_pairs; // increment npair

      // cutoff (based on geometric mean) for making decision to do inter-mode
      // transfers We took geometric mean of the participating modes (source
      // and destination) to find a cutoff or threshold from moving particles
      // from the source to the destination mode.
      const Real alnsg_amode_dest_mode = log(modes(dest_mode - 1).mean_std_dev);
      diameter_cutoff[src_mode] =
          sqrt(modes(src_mode).nom_diameter * exp(1.5 * square(alnsg_amode)) *
               modes(dest_mode).nom_diameter *
               exp(1.5 * square(alnsg_amode_dest_mode)));

      ln_dia_cutoff[src_mode] = log(diameter_cutoff[src_mode]); // log of cutoff
      diameter_threshold[src_mode] =
          0.99 * diameter_cutoff[src_mode]; // 99% of the cutoff
    }
  }

  // Factor, mass_2_vol, to convert from
  // q_mmr[kmol-species/kmol-air]) to volume units[m3/kmol-air]
  for (int iaero = 0; iaero < AeroConfig::num_aerosol_ids(); ++iaero) {
    mass_2_vol[iaero] =
        aero_species(iaero).molecular_weight / aero_species(iaero).density;
  }
} // end find_renaming_pairs





} // namespace rename

/// @class Rename
/// This class implements MAM4's rename parameterization.
class Rename {
public:
  // rename-specific configuration
  struct Config {
    // default constructor -- sets default values for parameters

    int _dest_mode_of_mode[AeroConfig::num_modes()];
    bool _iscldy;
    Config() : _dest_mode_of_mode{0, 1, 0, 0}, _iscldy{false} {}

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  Config config_;

  int _num_pairs;
  Real _sz_factor[AeroConfig::num_modes()],
      _fmode_dist_tail_fac[AeroConfig::num_modes()],
      _v2n_lo_rlx[AeroConfig::num_modes()],
      _v2n_hi_rlx[AeroConfig::num_modes()],
      _ln_diameter_tail_fac[AeroConfig::num_modes()],
      _diameter_cutoff[AeroConfig::num_modes()],
      _ln_dia_cutoff[AeroConfig::num_modes()],
      _diameter_threshold[AeroConfig::num_modes()],
      _mass_2_vol[AeroConfig::num_aerosol_ids()],
      _dgnum_amode[AeroConfig::num_modes()];

public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 rename"; }

  // init -- initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig &aero_config,
            const Config &rename_config = Config()) {
    // Set rename-specific config parameters.
    rename::find_renaming_pairs(config_._dest_mode_of_mode, // in
                                _sz_factor,                 // out
                                _fmode_dist_tail_fac,       // out
                                _v2n_lo_rlx,                // out
                                _v2n_hi_rlx,                // out
                                _ln_diameter_tail_fac,      // out
                                _num_pairs,                 // out
                                _diameter_cutoff,           // out
                                _ln_dia_cutoff,
                                _diameter_threshold, // out
                                _mass_2_vol);

    for (int m = 0; m < AeroConfig::num_modes(); ++m) {
      _dgnum_amode[m] = modes(m).nom_diameter;
    }

  } // end(init)

  // NOTE: this corresponds to mam_rename_1subarea() in the fortran refactor
  // code, which we include as a private function below
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atmosphere,
                          const Prognostics &prognostics,
                          const Diagnostics &diagnostics,
                          const Tendencies &tendencies) const {

    if (_num_pairs <= 0)
      return;

    const int nk = atmosphere.num_levels();

    const auto dest_mode_of_mode = config_._dest_mode_of_mode;
    const auto iscldy = config_._iscldy;
    const auto mass_2_vol = _mass_2_vol;

    const auto sz_factor = _sz_factor;
    const auto fmode_dist_tail_fac = _fmode_dist_tail_fac;
    const auto v2n_lo_rlx = _v2n_lo_rlx;
    const auto v2n_hi_rlx = _v2n_hi_rlx;
    const auto ln_diameter_tail_fac = _ln_diameter_tail_fac;
    const auto num_pairs = _num_pairs;
    const auto diameter_cutoff = _diameter_cutoff;
    const auto ln_dia_cutoff = _ln_dia_cutoff;
    const auto diameter_threshold = _diameter_threshold;
    const auto dgnum_amode = _dgnum_amode; 

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, nk), KOKKOS_CLASS_LAMBDA(int k) {
          Real qnum_cur[AeroConfig::num_modes()];
          Real qaer_cur[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()];
          Real qaer_del_grow4rnam[AeroConfig::num_modes()]
                                 [AeroConfig::num_aerosol_ids()];

          //
          Real qnumcw_cur[AeroConfig::num_modes()];
          Real qaercw_cur[AeroConfig::num_modes()]
                         [AeroConfig::num_aerosol_ids()];
          Real qaercw_del_grow4rnam[AeroConfig::num_modes()]
                                   [AeroConfig::num_aerosol_ids()];

          mam_rename_1subarea_(iscldy,
                               dest_mode_of_mode,    // in
                               sz_factor,            // in
                               fmode_dist_tail_fac,  // in
                               v2n_lo_rlx,           // in
                               v2n_hi_rlx,           // in
                               ln_diameter_tail_fac, // in
                               num_pairs,            // in
                               diameter_cutoff,      // in
                               ln_dia_cutoff,        // in
                               diameter_threshold,   // in
                               mass_2_vol, // in  
                               dgnum_amode, // in
                               qnum_cur, qaer_cur, // out
                               qaer_del_grow4rnam, qnumcw_cur, qaercw_cur, // out
                               qaercw_del_grow4rnam); // out
        }); // kokkos::parfor(k)

  }

private:
  KOKKOS_INLINE_FUNCTION
  void mam_rename_1subarea_(
      const bool iscldy,
      const int *dest_mode_of_mode,                             // in
      const Real sz_factor[AeroConfig::num_modes()],            // in
      const Real fmode_dist_tail_fac[AeroConfig::num_modes()],  // in
      const Real v2n_lo_rlx[AeroConfig::num_modes()],           // in
      const Real v2n_hi_rlx[AeroConfig::num_modes()],           // in
      const Real ln_diameter_tail_fac[AeroConfig::num_modes()], // in
      const Real num_pairs,                                     // in
      const Real diameter_cutoff[AeroConfig::num_modes()],      // in
      const Real ln_dia_cutoff[AeroConfig::num_modes()],        // in
      const Real diameter_threshold[AeroConfig::num_modes()],   // in
      const Real mass_2_vol[AeroConfig::num_aerosol_ids()],
      const Real dgnum_amode[AeroConfig::num_modes()],   // in
      Real qnum_cur[AeroConfig::num_modes()],
      Real qaer_cur[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
      Real qaer_del_grow4rnam[AeroConfig::num_modes()]
                             [AeroConfig::num_aerosol_ids()],
      Real qnumcw_cur[AeroConfig::num_modes()],
      Real qaercw_cur[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
      Real qaercw_del_grow4rnam[AeroConfig::num_modes()]
                               [AeroConfig::num_aerosol_ids()]) const {

    Real dryvol_i[mam4::AeroConfig::num_modes()];
    Real deldryvol_i[mam4::AeroConfig::num_modes()];

    // Interstitial aerosols: Compute initial (before growth) aerosol dry
    // volume and also the growth in dry volume of the "src" mode

    rename::compute_dryvol_change_in_src_mode(
        mam4::AeroConfig::num_modes(),       // in
        mam4::AeroConfig::num_aerosol_ids(), // in
        dest_mode_of_mode,                   // in
        qaer_cur,                            // in
        qaer_del_grow4rnam,                  // in
        mass_2_vol,                          // in
        dryvol_i,                            // out
        deldryvol_i                          // out
    );

    Real dryvol_c[mam4::AeroConfig::num_modes()];
    Real deldryvol_c[mam4::AeroConfig::num_modes()];

    if (iscldy) {

      rename::compute_dryvol_change_in_src_mode(
          AeroConfig::num_modes(),       // in
          AeroConfig::num_aerosol_ids(), // in
          dest_mode_of_mode,             // in
          qaercw_cur,                    // in
          qaercw_del_grow4rnam,          // in
          mass_2_vol,                    // in
          dryvol_c,                      // out
          deldryvol_c                    // out
      );

    } // end iscldy

    // Find fractions (mass and number) to transfer and complete the
    //     transfer

    rename::do_inter_mode_transfer(
        dest_mode_of_mode, iscldy,
        // volume to number relaxation limits [m^-3]
        v2n_lo_rlx, v2n_hi_rlx, sz_factor, fmode_dist_tail_fac,
        ln_diameter_tail_fac, ln_dia_cutoff, diameter_threshold, dgnum_amode,
        // dry volume [m3/kmol-air]
        dryvol_i, dryvol_c, deldryvol_i, deldryvol_c, qaer_cur,
        // aerosol number mixing ratios [#/kmol-air]
        qnum_cur, qaercw_cur, qnumcw_cur);
  }
}; // end(compute_tendencies)

} // namespace mam4

#endif
