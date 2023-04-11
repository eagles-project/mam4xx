// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

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

using haero::cube;
using haero::erfc;
using haero::log;
using haero::max;
using haero::min;
using haero::sqrt;
using Constants = haero::Constants;
using haero::square;

namespace rename {

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
  const Real zero = 0;
  for (int m = 0; m < nmode; ++m) {
    int dest_mode = dest_mode_of_mode[m];

    if (dest_mode >= 0) {

      // For each mode, we compute a dry volume by combining (accumulating)
      // mass/density for each species in that mode.
      // conversion from mass to volume is accomplished by multiplying with
      // precomputed "mass_2_vol" factor

      // initialize tmp accumulators
      Real tmp_dryvol = zero;     // dry volume accumulator
      Real tmp_del_dryvol = zero; // dry volume growth(change) accumulator

      // Notes on mass_2_vol factor: Units:[m3/kmol-species]. This factor is
      // obtained by (molecular_weight/density) of a species. That is,
      // [(g/mol-species) / (kg-species/m3)]; where molecular_weight has units
      // [g/mol-species] and density units are [kg-species/m3] which results in
      // the units of m3/kmol-species

      for (int ispec = 0; ispec < nspec; ++ispec) {
        // Multiply by mass_2_vol [m3/kmol-species] to convert
        // q_mmr [kmol-species/kmol-air] to volume units [m3/kmol-air]
        tmp_dryvol += q_mmr[m][ispec] * mass_2_vol[ispec];
        // accumulate the "growth" in volume units as well
        tmp_del_dryvol += q_del_growth[m][ispec] * mass_2_vol[ispec];
      }

      // This is dry volume before the growth
      dryvol[m] = tmp_dryvol - tmp_del_dryvol;
      // change in dry volume due to growth
      deldryvol[m] = tmp_del_dryvol;
    }
  }
} // end compute_dryvol_change_in_src_mode()

KOKKOS_INLINE_FUNCTION
Real total_interstitial_and_cloudborne() {
  Real out = 1;
  return out;
}

// this function determines the total quantity of interest (both interstitial
// and cloudborne) for a given mode, whether that be number or mixing ratio
KOKKOS_INLINE_FUNCTION
Real total_interstitial_and_cloudborne(
    const bool &is_cloudy, const int &imode,
    const Real interstitial[AeroConfig::num_modes()],
    const Real cloudborne[AeroConfig::num_modes()]) {
  // if there is no cloud, total is just the interstitial value
  Real total = interstitial[imode];
  if (is_cloudy) {
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
    const bool &is_cloudy, const int &src_mode,
    const Real &smallest_dryvol_value,
    const Real dryvol_i[AeroConfig::num_modes()],
    const Real dryvol_c[AeroConfig::num_modes()],
    Real qnum_i_cur[AeroConfig::num_modes()],
    Real qnum_c_cur[AeroConfig::num_modes()], const Real &num2vol_ratiolo,
    const Real &num2vol_ratiohi,
    // out
    Real &b4_growth_dryvol, Real &b4_growth_dryvol_bounded,
    Real &b4_growth_qnum_bounded) {

  // Compute total (i.e. cloud borne and interstitial) of dry volume (before
  // growth) and delta in dry volume in the source mode
  // [units: (m3 of species)/(kmol of air)]
  const Real zero = 0.0;
  b4_growth_dryvol = total_interstitial_and_cloudborne(is_cloudy, src_mode,
                                                       dryvol_i, dryvol_c);

  // FIXME: is it feasible that pregrowth_dryvol would be smaller than 1e-25?
  // NOTE: we get rid of this and use safe_divide() in do_inter_mode_transfer()
  b4_growth_dryvol_bounded =
      haero::max(b4_growth_dryvol, smallest_dryvol_value);

  // Compute total before growth number [units: #/kmol-air]
  Real b4_growth_qnum = total_interstitial_and_cloudborne(
      is_cloudy, src_mode, qnum_i_cur, qnum_c_cur);
  b4_growth_qnum = max(zero, b4_growth_qnum); // bound to have minimum of 0

  // // bound number within min and max of the source mode
  b4_growth_qnum_bounded = utils::min_max_bound(
      b4_growth_dryvol_bounded * num2vol_ratiohi, // min value
      b4_growth_dryvol_bounded * num2vol_ratiolo, // max value
      b4_growth_qnum);                            // input
} // end compute_before_growth_dryvol_and_num()

// Compute tail fraction with log_dia_tail_fac

KOKKOS_INLINE_FUNCTION
void compute_tail_fraction(const Real diameter, const Real log_dia_cutoff,
                           const Real tail_dist_fac,
                           const Real log_dia_tail_fac, // input
                           Real &tail_fraction) {
  // Compute tail fraction to be used for inter-mode species transfer
  // rename use present function for this if statement.
  const Real log_diameter = log(diameter) + log_dia_tail_fac;
  const Real tail = (log_dia_cutoff - log_diameter) * tail_dist_fac;
  // complimentary error function (erfc)
  tail_fraction = Real(0.5) * erfc(tail);

} // end compute_tail_fraction

// Compute tail fraction without log_dia_tail_fac
KOKKOS_INLINE_FUNCTION
void compute_tail_fraction(const Real diameter, const Real log_dia_cutoff,
                           const Real tail_dist_fac, Real &tail_fraction) {
  // Compute tail fraction to be used for inter-mode species transfer we use
  // this function if log_dia_tail_fac is not present in the function call
  const Real tail = (log_dia_cutoff - log(diameter)) * tail_dist_fac;
  // complimentary error function (erfc)
  tail_fraction = Real(0.5) * erfc(tail);

} // end compute_tail_fraction

KOKKOS_INLINE_FUNCTION
void compute_xfer_fractions(const Real b4_growth_dryvol,
                            const Real after_growth_dryvol,
                            const Real b4_growth_tail_fr_vol,
                            const Real after_growth_tail_fr_vol, // in
                            const Real after_growth_tail_fr_num,
                            const Real b4_growth_tail_fr_qnum, // in
                            bool &is_xfer_frac_zero,           // out
                            Real &xfer_vol_frac,
                            Real &xfer_num_frac // out
) {

  // BAD CONSTANT
  // 1-eps (this number is little less than 1, e.g. 0.99) // FIXME: this comment
  // is nonsense
  constexpr Real xferfrac_max =
      Real(1.0) - 10.0 * std::numeric_limits<Real>::epsilon();
  // assume we have fractions to transfer, so we will not skip the rest of the
  // calculations
  is_xfer_frac_zero = false;
  const Real zero = 0.0;

  // transfer fraction is difference between new and old tail-fractions
  const Real volume_fraction = after_growth_tail_fr_vol * after_growth_dryvol -
                               b4_growth_tail_fr_vol * b4_growth_dryvol;

  if (volume_fraction <= zero) {
    is_xfer_frac_zero = true;
    return;
  }

  xfer_vol_frac =
      min(volume_fraction, after_growth_dryvol) / after_growth_dryvol;
  xfer_vol_frac = min(xfer_vol_frac, xferfrac_max);
  xfer_num_frac = after_growth_tail_fr_num - b4_growth_tail_fr_qnum;

  // transfer fraction for number cannot exceed that of mass
  xfer_num_frac = max(zero, min(xfer_num_frac, xfer_vol_frac));

} // end compute_xfer_fractions

KOKKOS_INLINE_FUNCTION
void do_num_and_mass_transfer(
    const int src_mode, const int dest_mode, const Real xfer_vol_frac,
    const Real xfer_num_frac, // input
    // FIXME: will qmol be updated this way?--verify with unit test
    // aerosol molar mixing ratio [kmol/kmol-dry-air]
    Real qmol[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
    // aerosol number mixing ratios [#/kmol-air]
    Real qnum[AeroConfig::num_modes()]) {
  // compute changes to number and species masses
  const Real num_trans = qnum[src_mode] * xfer_num_frac;
  qnum[src_mode] -= num_trans;
  qnum[dest_mode] += num_trans;

  for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
    const Real vol_trans = qmol[src_mode][ispec] * xfer_vol_frac;
    qmol[src_mode][ispec] -= vol_trans;
    qmol[dest_mode][ispec] += vol_trans;
  }
} // end do_num_and_mass_transfer

// TODO: this is a placeholder until we create a proper unit test
KOKKOS_INLINE_FUNCTION
void do_inter_mode_transfer() {}

KOKKOS_INLINE_FUNCTION
void do_inter_mode_transfer(
    const int dest_mode_of_mode[AeroConfig::num_modes()], const bool &is_cloudy,
    const Real &smallest_dryvol_value,
    // volume to number relaxation limits [m^-3]
    const Real num2vol_ratiolorlx[AeroConfig::num_modes()],
    const Real num2vol_ratiohirlx[AeroConfig::num_modes()],
    const Real mean_std_dev[AeroConfig::num_modes()],
    const Real fmode_dist_tail_fac[AeroConfig::num_modes()],
    const Real ln_diameter_tail_fac[AeroConfig::num_modes()],
    const Real ln_dia_cutoff[AeroConfig::num_modes()],
    const Real diameter_threshold[AeroConfig::num_modes()],
    const Real dgnum_amode[AeroConfig::num_modes()],
    // dry volume [m3/kmol-air]
    const Real dryvol_i[AeroConfig::num_modes()],
    const Real dryvol_c[AeroConfig::num_modes()],
    const Real deldryvol_i[AeroConfig::num_modes()],
    const Real deldryvol_c[AeroConfig::num_modes()],
    Real qmol_i_cur[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
    // aerosol number mixing ratios [#/kmol-air]
    Real qnum_i_cur[AeroConfig::num_modes()],
    Real qmol_c_cur[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
    Real qnum_c_cur[AeroConfig::num_modes()]) {
  // local variables
  const int nmodes = AeroConfig::num_modes();
  int src_mode, dest_mode;
  const Real zero = 0;

  Real b4_growth_dryvol = zero;         // [m3/kmol-air]
  Real b4_growth_dryvol_bounded = zero; // [m3/kmol-air]
  Real b4_growth_qnum_bounded = zero;   // [#/kmol-air]

  // Loop through the modes and do the transfer
  for (int imode = 0; imode < nmodes; ++imode) {
    src_mode = imode;                     // source mode
    dest_mode = dest_mode_of_mode[imode]; // destination mode

    // if destination mode doesn't exist for the source mode, cycle loop
    if (dest_mode < 0) {
      continue;
    }
    // compute before growth dry volume and number
    compute_before_growth_dryvol_and_num(
        // in
        is_cloudy, src_mode, smallest_dryvol_value, dryvol_i, dryvol_c,
        qnum_i_cur, qnum_c_cur, num2vol_ratiolorlx[src_mode],
        num2vol_ratiohirlx[src_mode],
        // out
        b4_growth_dryvol, b4_growth_dryvol_bounded, b4_growth_qnum_bounded);

    // change (delta) in dryvol
    const Real dryvol_del = total_interstitial_and_cloudborne(
        is_cloudy, src_mode, deldryvol_i, deldryvol_c);

    // Total dry volume after growth (add delta growth)
    Real after_growth_dryvol = b4_growth_dryvol + dryvol_del;

    // FIXME: maybe floating_point::zero(aft_growth_dryvol)?
    // TODO: ask about this in PR
    // Skip inter-mode transfer for this mode if dry after growth is ~ 0
    if (after_growth_dryvol <= smallest_dryvol_value) {
      continue;
    }

    // FIXME: use safe_divide() here
    const Real b4_growth_mode_mean_particle_volume =
        b4_growth_dryvol_bounded / b4_growth_qnum_bounded;
    Real b4_growth_diameter = conversions::mean_particle_diameter_from_volume(
        b4_growth_mode_mean_particle_volume, mean_std_dev[src_mode]);

    // if the before growth diameter is more than the threshold
    // (diameter_threshold), we restrict diameter to the threshold and change
    // dry volume accordingly
    if (b4_growth_diameter > diameter_threshold[src_mode]) {
      // this revised volume corresponds to b4_growth_diameter ==
      // diameter_threshold, and same number conc
      b4_growth_dryvol = b4_growth_dryvol * cube(diameter_threshold[src_mode] /
                                                 b4_growth_diameter);
      b4_growth_diameter = diameter_threshold[src_mode];
    }

    // FIXME: BAD CONSTANT
    // FIXME: maybe floating_point::zero(<quantity>)?
    // TODO: ask about this in PR
    if ((after_growth_dryvol - b4_growth_dryvol) <=
        Real(1.0e-6) * b4_growth_dryvol_bounded) {
      continue;
    }
    // Compute after growth diameter; if it is less than the "nominal" or
    // "base" diameter for the source mode, skip inter-mode transfer
    const Real after_growth_mode_mean_particle_volume =
        after_growth_dryvol / b4_growth_qnum_bounded;
    Real aft_growth_diameter = conversions::mean_particle_diameter_from_volume(
        after_growth_mode_mean_particle_volume, mean_std_dev[src_mode]);

    if (aft_growth_diameter <= dgnum_amode[src_mode]) {
      continue;
    }

    // compute before growth number fraction in the tail

    // call compute_tail_fraction(b4_growth_diameter,ln_dia_cutoff(src_mode),
    // log_dia_tail_fac is not present in original call to
    // compute_tail_fraction. Thus do not no include its value
    Real b4_growth_tail_fr_qnum = zero;
    compute_tail_fraction(b4_growth_diameter, ln_dia_cutoff[src_mode],
                          fmode_dist_tail_fac[src_mode],
                          b4_growth_tail_fr_qnum // out
    );

    // compute before growth volume (or mass) fraction in the tail
    Real b4_growth_tail_fr_vol = zero;

    compute_tail_fraction(b4_growth_diameter, ln_dia_cutoff[src_mode],
                          fmode_dist_tail_fac[src_mode],
                          ln_diameter_tail_fac[src_mode],
                          b4_growth_tail_fr_vol // out
    );

    // compute after growth number fraction in the tail log_dia_tail_fac is not
    // presented in original call to compute_tail_fraction.
    // Thus do not include its value
    Real after_growth_tail_fr_num = zero;
    compute_tail_fraction(aft_growth_diameter, ln_dia_cutoff[src_mode],
                          fmode_dist_tail_fac[src_mode],
                          after_growth_tail_fr_num // out
    );

    // compute after growth volume (or mass) fraction in the tail
    Real after_growth_tail_fr_vol = zero;
    compute_tail_fraction(aft_growth_diameter, ln_dia_cutoff[src_mode],
                          fmode_dist_tail_fac[src_mode],
                          ln_diameter_tail_fac[src_mode],
                          after_growth_tail_fr_vol // out
    );

    // compute transfer fraction (volume and mass) - if less than zero,
    // cycle loop
    bool is_xfer_frac_zero = false;
    Real xfer_vol_frac = zero;
    Real xfer_num_frac = zero;

    compute_xfer_fractions(b4_growth_dryvol, after_growth_dryvol,
                           b4_growth_tail_fr_vol,
                           after_growth_tail_fr_vol, // in
                           after_growth_tail_fr_num,
                           b4_growth_tail_fr_qnum, // in
                           is_xfer_frac_zero,      // out
                           xfer_vol_frac,
                           xfer_num_frac // out
    );

    if (is_xfer_frac_zero) {
      continue;
    }

    // do the transfer for the interstitial species
    do_num_and_mass_transfer(src_mode, dest_mode, xfer_vol_frac,
                             xfer_num_frac, // input
                             qmol_i_cur, qnum_i_cur);
    if (is_cloudy) {
      do_num_and_mass_transfer(src_mode, dest_mode, xfer_vol_frac,
                               xfer_num_frac, // input
                               qmol_c_cur, qnum_c_cur);
    }
  } // end for(imode)
} // end do_inter_mode_transfer()

KOKKOS_INLINE_FUNCTION
void find_renaming_pairs(
    const int *dest_mode_of_mode,                       // in
    Real mean_std_dev[AeroConfig::num_modes()],         // out
    Real fmode_dist_tail_fac[AeroConfig::num_modes()],  // out
    Real num2vol_ratio_lo_rlx[AeroConfig::num_modes()], // out
    Real num2vol_ratio_hi_rlx[AeroConfig::num_modes()], // out
    Real ln_diameter_tail_fac[AeroConfig::num_modes()], // out
    int &num_pairs,                                     // out
    Real diameter_cutoff[AeroConfig::num_modes()],      // out
    Real ln_dia_cutoff[AeroConfig::num_modes()],
    Real diameter_threshold[AeroConfig::num_modes()]) {
  const Real sqrt_half = haero::sqrt(0.5);
  // (3^3): relaxing 3 * diameter, which makes it 3^3 for volume
  const Real frelax = 27.0;
  const Real zero = 0;

  num_pairs = 0;

  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    const int dest_mode =
        dest_mode_of_mode[m]; // "destination" mode for mode "imode"

    if (dest_mode < 0) {
      mean_std_dev[m] = zero;
      fmode_dist_tail_fac[m] = zero;
      num2vol_ratio_lo_rlx[m] = zero;
      num2vol_ratio_hi_rlx[m] = zero;
      ln_diameter_tail_fac[m] = zero;
      diameter_cutoff[m] = zero;
      ln_dia_cutoff[m] = zero;
      diameter_threshold[m] = zero;

    } else {
      const Real alnsg_amode = log(modes(m).mean_std_dev);
      mean_std_dev[m] = modes(m).mean_std_dev;
      // factor for computing distribution tails of the "src mode"
      fmode_dist_tail_fac[m] = sqrt_half / alnsg_amode;
      // compute volume to number high and low limits with relaxation
      // coefficients (watch out for the repeated calculations)
      num2vol_ratio_lo_rlx[m] =
          Real(1) /
          conversions::mean_particle_volume_from_diameter(
              modes(m).min_diameter, modes(m).mean_std_dev) *
          frelax;
      num2vol_ratio_hi_rlx[m] =
          Real(1) /
          conversions::mean_particle_volume_from_diameter(
              modes(m).max_diameter, modes(m).mean_std_dev) /
          frelax;
      // A factor for computing diameter at the tails of the distribution
      ln_diameter_tail_fac[m] = Real(3.0) * square(alnsg_amode);

      // transfer "src" mode is the current mode (i.e. imode)
      const int src_mode = m;

      // ^^At this point, we know that particles can be transferred from the
      // "src_mode" to "dest_mode". "src_mode" is the current mode (i.e. imode)

      // update number of pairs found so far
      num_pairs += 1;

      // cutoff (based on geometric mean) for making decision to do inter-mode
      // transfers We took geometric mean of the participating modes (source
      // and destination) to find a cutoff or threshold from moving particles
      // from the source to the destination mode.
      // FIXME: This looks very strange to us, can someone take a look?
      // e.g., taking log then exp, units?
      const Real alnsg_amode_dest_mode = log(modes(dest_mode).mean_std_dev);
      diameter_cutoff[src_mode] =
          sqrt(modes(src_mode).nom_diameter * exp(1.5 * square(alnsg_amode)) *
               modes(dest_mode).nom_diameter *
               exp(1.5 * square(alnsg_amode_dest_mode)));

      // log of cutoff
      ln_dia_cutoff[src_mode] = log(diameter_cutoff[src_mode]);
      // 99% of the cutoff
      // FIXME: BAD CONSTANT!
      diameter_threshold[src_mode] = 0.99 * diameter_cutoff[src_mode];
    } // end if/else (dest_mode < 0)
  }   // end for(m)
} // end find_renaming_pairs
} // end namespace rename

/// @class Rename
/// This class implements MAM4's rename parameterization.
class Rename {
public:
  // rename-specific configuration
  struct Config {
    int _dest_mode_of_mode[AeroConfig::num_modes()];
    // Molecular weights in mam4-rename units kg/kmol
    Real _molecular_weight_soa = 150;
    // FIXME. MW for SO4 is not a standard MW.
    Real _molecular_weight_so4 = 115;
    Real _molecular_weight_pom = 150;

    Real _smallest_dryvol_value;
    // NOTE: rows are ordered in standard convention (accumulation, aitken,
    // coarse, primary carbon), and the species indexing works as follows:
    // rename_spec_arr[x \in {0,...,3}][y \in {0,...,6}] =
    //                                  m4x_spec_arr[x][mam4xx2rename_idx[x][y]]
    int _mam4xx2rename_idx[4][7];
    // default constructor--sets default values for parameters
    Config()
        : _dest_mode_of_mode{-1, 0, -1, -1},
          // NOTE: smallest_dryvol_value is a very small molar mixing ratio
          // [m3-spc/kmol-air] (where m3-species) is meter cubed volume of a
          // species) used for avoiding overflow. it corresponds to dp = 1 nm
          // and number = 1e-5 #/mg-air ~= 1e-5 #/cm3-air
          _smallest_dryvol_value{1.0e-25}, _mam4xx2rename_idx{
                                               {0, 1, 2, 3, 4, 5, 6},
                                               {0, 1, -1, -1, 4, 6, -1},
                                               {0, 1, 2, 3, 4, 5, 6},
                                               {2, 3, 6, -1, -1, -1, -1}} {}

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  Config config_;

  int _num_pairs;
  int _mam4xx2rename_idx[AeroConfig::num_modes()]
                        [AeroConfig::num_aerosol_ids()];
  Real _mean_std_dev[AeroConfig::num_modes()],
      _fmode_dist_tail_fac[AeroConfig::num_modes()],
      _num2vol_ratio_lo_rlx[AeroConfig::num_modes()],
      _num2vol_ratio_hi_rlx[AeroConfig::num_modes()],
      _ln_diameter_tail_fac[AeroConfig::num_modes()],
      _diameter_cutoff[AeroConfig::num_modes()],
      _ln_dia_cutoff[AeroConfig::num_modes()],
      _diameter_threshold[AeroConfig::num_modes()],
      // Notes on mass_2_vol factor:Units:[m^3/kmol-species]; where kmol-species
      // is the amount of a given species. This factor is obtained by
      // (molecular_weight/density) of a species. That is, [ (g/mol-species) /
      // (kg-species/m3)].
      _mass_2_vol[AeroConfig::num_aerosol_ids()],
      _dgnum_amode[AeroConfig::num_modes()];

public:
  // name--unique name of the process implemented by this class
  const char *name() const { return "MAM4 rename"; }

  // init--initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig &aero_config,
            const Config &rename_config = Config()) {
    rename::find_renaming_pairs(config_._dest_mode_of_mode, // in
                                _mean_std_dev,              // out
                                _fmode_dist_tail_fac,       // out
                                _num2vol_ratio_lo_rlx,      // out
                                _num2vol_ratio_hi_rlx,      // out
                                _ln_diameter_tail_fac,      // out
                                _num_pairs,                 // out
                                _diameter_cutoff,           // out
                                _ln_dia_cutoff, _diameter_threshold);

    for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
      _dgnum_amode[imode] = modes(imode).nom_diameter;
      for (int jspec = 0; jspec < AeroConfig::num_aerosol_ids(); ++jspec) {
        _mam4xx2rename_idx[imode][jspec] =
            config_._mam4xx2rename_idx[imode][jspec];
      }
    }

    // Factor, mass_2_vol, to convert from
    // q_mmr[kmol-species/kmol-air]) to volume units[m3/kmol-air]
    // Real molecular_weight_rename[AeroConfig::num_aerosol_ids()] = {
    //     150, 115, 150, 12, 58.5, 135, 250092}; // [kg/kmol]
    // for (int iaero = 0; iaero < AeroConfig::num_aerosol_ids(); ++iaero) {
    //   _mass_2_vol[iaero] =
    //      molecular_weight_rename[iaero] / aero_species(iaero).density;
    // }
    // FIXME.
    // Molecular weights (MW) of aerosol species have units of kg/mol,
    // MWs in rename have units of kg/kmol.
    // Additionally, MW of SOA, SO4, and POM in rename have different values
    // than the ones from aero_modes.hpp this uses the aero_modes.hpp values
    const Real unit_factor = 1000; // from kg/mol to kg/kmol

    for (int iaero = 0; iaero < AeroConfig::num_aerosol_ids(); ++iaero) {
      _mass_2_vol[iaero] = aero_species(iaero).molecular_weight /
                           aero_species(iaero).density * unit_factor;
    }
    // Correction because of differences in MWs between mam4xx and mam4
    int iaer_soa = aerosol_index_for_mode(ModeIndex::Accumulation, AeroId::SOA);
    int iaer_so4 = aerosol_index_for_mode(ModeIndex::Accumulation, AeroId::SO4);
    int iaer_pom = aerosol_index_for_mode(ModeIndex::Accumulation, AeroId::POM);

    _mass_2_vol[iaer_soa] =
        config_._molecular_weight_soa / aero_species(iaer_soa).density;
    _mass_2_vol[iaer_so4] =
        config_._molecular_weight_so4 / aero_species(iaer_so4).density;
    _mass_2_vol[iaer_pom] =
        config_._molecular_weight_pom / aero_species(iaer_pom).density;

  } // end(init)

  // NOTE: this corresponds to mam_rename_1subarea() in the fortran refactor
  // code, which we include as a private function below and call here
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atmosphere,
                          const Prognostics &prognostics,
                          const Diagnostics &diagnostics,
                          const Tendencies &tendencies) const {

    if (_num_pairs <= 0)
      return;

    const int nk = atmosphere.num_levels();
    const int nmodes = AeroConfig::num_modes();
    const int nspec = AeroConfig::num_aerosol_ids();

    const auto dest_mode_of_mode = config_._dest_mode_of_mode;
    const auto is_cloudy = diagnostics.is_cloudy;
    const auto mass_2_vol = _mass_2_vol;
    const Real smallest_dryvol_value = config_._smallest_dryvol_value;

    const auto mean_std_dev = _mean_std_dev;
    const auto fmode_dist_tail_fac = _fmode_dist_tail_fac;
    const auto num2vol_ratio_lo_rlx = _num2vol_ratio_lo_rlx;
    const auto num2vol_ratio_hi_rlx = _num2vol_ratio_hi_rlx;
    const auto ln_diameter_tail_fac = _ln_diameter_tail_fac;
    const auto num_pairs = _num_pairs;
    const auto diameter_cutoff = _diameter_cutoff;
    const auto ln_dia_cutoff = _ln_dia_cutoff;
    const auto diameter_threshold = _diameter_threshold;
    const auto dgnum_amode = _dgnum_amode;

    // =======================================================================
    // NOTE: these variables are renamed, to fit the _i/c convention (and for
    // clarity), so the mapping from mam4 -> mam4xx is:
    // =======================================================================
    // interstitial aerosol number mixing ratio [#/kmol-air], dim = num_modes
    // qnum_cur -> qnum_i_cur
    // interstitial aerosol molar mixing ratio [kmol/kmol-air], dim = num_modes
    // x num_aerosol_ids
    // qaer_cur -> qmol_i_cur
    // growth in aerosol molar mixing ratio [kmol/kmol-air]
    // qaer_del_grow4rnam -> qmol_i_del
    // cloudborne aerosol number mixing ratio [#/kmol-air], dim = num_modes
    // qnumcw_cur -> qnum_c_cur
    // cloudborne aerosol molar mixing ratio [kmol/kmol-air], dim = num_modes x
    // num_aerosol_ids
    // qaercw_cur -> qmol_c_cur
    // growth in aerosol molar mixing ratio [kmol/kmol-air]
    // qaercw_del_grow4rnam -> qmol_c_del
    // =======================================================================

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, nk), KOKKOS_CLASS_LAMBDA(int kk) {
          Real qnum_i_cur[AeroConfig::num_modes()];
          Real qmol_i_cur[AeroConfig::num_modes()]
                         [AeroConfig::num_aerosol_ids()];
          Real qmol_i_del[AeroConfig::num_modes()]
                         [AeroConfig::num_aerosol_ids()];

          //
          Real qnum_c_cur[AeroConfig::num_modes()];
          Real qmol_c_cur[AeroConfig::num_modes()]
                         [AeroConfig::num_aerosol_ids()];
          Real qmol_c_del[AeroConfig::num_modes()]
                         [AeroConfig::num_aerosol_ids()];

          const bool &is_cloudy_cur = is_cloudy(kk);
          int rename_idx = 0;

          // FIXME: adjust these to use mamRefactor's MW's
          for (int imode = 0; imode < nmodes; ++imode) {
            qnum_i_cur[imode] = prognostics.n_mode_i[imode](kk);
            qnum_c_cur[imode] = prognostics.n_mode_c[imode](kk);
            for (int jspec = 0; jspec < nspec; ++jspec) {
              // get the mapping from the mam4xx species ordering to rename's
              rename_idx = _mam4xx2rename_idx[imode][jspec];
              // convert mass mixing ratios to molar mixing ratios
              qmol_i_cur[imode][rename_idx] = conversions::vmr_from_mmr(
                  prognostics.q_aero_i[imode][rename_idx](kk),
                  aero_species(rename_idx).molecular_weight);
              qmol_i_del[imode][rename_idx] = conversions::vmr_from_mmr(
                  tendencies.q_aero_i[imode][rename_idx](kk),
                  aero_species(rename_idx).molecular_weight);
              qmol_c_cur[imode][rename_idx] = conversions::vmr_from_mmr(
                  prognostics.q_aero_c[imode][rename_idx](kk),
                  aero_species(rename_idx).molecular_weight);
              qmol_c_del[imode][rename_idx] = conversions::vmr_from_mmr(
                  tendencies.q_aero_c[imode][rename_idx](kk),
                  aero_species(rename_idx).molecular_weight);
            }
          }

          mam_rename_1subarea_(is_cloudy_cur, smallest_dryvol_value,
                               dest_mode_of_mode,                  // in
                               mean_std_dev,                       // in
                               fmode_dist_tail_fac,                // in
                               num2vol_ratio_lo_rlx,               // in
                               num2vol_ratio_hi_rlx,               // in
                               ln_diameter_tail_fac,               // in
                               num_pairs,                          // in
                               diameter_cutoff,                    // in
                               ln_dia_cutoff,                      // in
                               diameter_threshold,                 // in
                               mass_2_vol,                         // in
                               dgnum_amode,                        // in
                               qnum_i_cur, qmol_i_cur,             // out
                               qmol_i_del, qnum_c_cur, qmol_c_cur, // out
                               qmol_c_del);                        // out
        }); // end kokkos::parfor(kk)
    // FIXME: convert back to mass mixing ratios and store in progs/tends
    // prognostics = ???
    // tendencies = ???
  } // end compute_tendencies()

  // Make mam_rename_1subarea public for testing proposes.
  KOKKOS_INLINE_FUNCTION
  void mam_rename_1subarea_(
      const bool is_cloudy_cur, const Real &smallest_dryvol_value,
      const int *dest_mode_of_mode,                             // in
      const Real mean_std_dev[AeroConfig::num_modes()],         // in
      const Real fmode_dist_tail_fac[AeroConfig::num_modes()],  // in
      const Real num2vol_ratio_lo_rlx[AeroConfig::num_modes()], // in
      const Real num2vol_ratio_hi_rlx[AeroConfig::num_modes()], // in
      const Real ln_diameter_tail_fac[AeroConfig::num_modes()], // in
      const int num_pairs,                                      // in
      const Real diameter_cutoff[AeroConfig::num_modes()],      // in
      const Real ln_dia_cutoff[AeroConfig::num_modes()],        // in
      const Real diameter_threshold[AeroConfig::num_modes()],   // in
      const Real mass_2_vol[AeroConfig::num_aerosol_ids()],
      const Real dgnum_amode[AeroConfig::num_modes()], // in
                                                       //
      Real qnum_i_cur[AeroConfig::num_modes()],
      Real qmol_i_cur[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
      Real qmol_i_del[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],

      Real qnum_c_cur[AeroConfig::num_modes()],
      Real qmol_c_cur[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
      Real qmol_c_del[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()])

      const {
    const Real zero = 0;
    Real dryvol_i[mam4::AeroConfig::num_modes()] = {zero};
    Real deldryvol_i[mam4::AeroConfig::num_modes()] = {zero};

    // Interstitial aerosols: Compute initial (before growth) aerosol dry
    // volume and also the growth in dry volume of the "src" mode

    rename::compute_dryvol_change_in_src_mode(
        mam4::AeroConfig::num_modes(),       // in
        mam4::AeroConfig::num_aerosol_ids(), // in
        dest_mode_of_mode,                   // in
        qmol_i_cur,                          // in
        qmol_i_del,                          // in
        mass_2_vol,                          // in
        dryvol_i,                            // out
        deldryvol_i                          // out
    );

    Real dryvol_c[mam4::AeroConfig::num_modes()] = {zero};
    Real deldryvol_c[mam4::AeroConfig::num_modes()] = {zero};

    if (is_cloudy_cur) {

      rename::compute_dryvol_change_in_src_mode(
          AeroConfig::num_modes(),       // in
          AeroConfig::num_aerosol_ids(), // in
          dest_mode_of_mode,             // in
          qmol_c_cur,                    // in
          qmol_c_del,                    // in
          mass_2_vol,                    // in
          dryvol_c,                      // out
          deldryvol_c                    // out
      );

    } // end is_cloudy_cur

    // Find fractions (mass and number) to transfer and complete the transfer

    rename::do_inter_mode_transfer(
        dest_mode_of_mode, is_cloudy_cur, smallest_dryvol_value,
        // volume to number relaxation limits [m^-3]
        num2vol_ratio_lo_rlx, num2vol_ratio_hi_rlx, mean_std_dev,
        fmode_dist_tail_fac, ln_diameter_tail_fac, ln_dia_cutoff,
        diameter_threshold, dgnum_amode,
        // dry volume [m3/kmol-air]
        dryvol_i, dryvol_c, deldryvol_i, deldryvol_c, qmol_i_cur,
        // aerosol number mixing ratios [#/kmol-air]
        qnum_i_cur, qmol_c_cur, qnum_c_cur);
  } // end mam_rename_1subarea_()
};  // end class Rename

} // end namespace mam4

#endif
