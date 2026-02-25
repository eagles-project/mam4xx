// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "validation.hpp"
#include "bfbhash.hpp"

#include <ekat_fpe.hpp>

#include <ctime>
#include <iomanip>
#include <sstream>

// BFB hash for a skywalker::Ensemble (logic derived from EAMxx's
// atmosphere_process_hash.cpp)

namespace {

std::string exe_name_;

using namespace skywalker;
using mam4::bfbhash::HashType;

void hash(const std::vector<Real> &array, HashType &accum) {
  for (Real value : array) {
    mam4::bfbhash::hash(value, accum);
  }
}

void hash(const Input &input, HashType &accum) {
  // hash scalar inputs, then arrays
  std::string name;
  Real scalar;
  while (input.next_scalar(name, scalar)) {
    mam4::bfbhash::hash(scalar, accum);
  }
  std::vector<Real> array;
  while (input.next_array(name, array)) {
    hash(array, accum);
  }
}

void hash(const Output &output, HashType &accum) {
  // hash scalar inputs, then arrays
  std::string name;
  Real scalar;
  while (output.next_scalar(name, scalar)) {
    mam4::bfbhash::hash(scalar, accum);
  }
  std::vector<Real> array;
  while (output.next_array(name, array)) {
    hash(array, accum);
  }
}

// prints a single BFB hash for all
void print_bfbhash(Ensemble *e) {
  std::vector<std::string> hash_names;
  std::vector<HashType> accum;

  // we generate a single hash for input and another for output

  hash_names.push_back("input");
  accum.emplace_back();
  e->process([&accum](const Input &input, Output &output) {
    hash(input, accum.back());
  });

  hash_names.push_back("output");
  accum.emplace_back();
  e->process([&accum](const Input &input, Output &output) {
    hash(output, accum.back());
  });

  // pretty print the hashes

  int name_len = 0;
  for (const auto &name : hash_names) {
    name_len = std::max(name_len, static_cast<int>(name.length()) + 1);
  }

  time_t t0 = std::time(nullptr);
  std::tm *t = std::gmtime(&t0);
  int tod = 3600 * t->tm_hour + 60 * t->tm_min + t->tm_sec;

  std::stringstream ss;
  ss << "mam4xx hash> exe=" << exe_name_ << " date=" << std::setw(4)
     << std::setfill('0') << t->tm_year + 1900 << "-"          // year
     << std::setw(2) << std::setfill('0') << t->tm_mon << "-"  // month
     << std::setw(2) << std::setfill('0') << t->tm_mday << "-" // day
     << std::setw(5) << std::setfill('0') << tod; // time of day (seconds)
  std::cout << ss.str() << std::endl;

  for (int i = 0; i < accum.size(); ++i) {
    ss.str(""); // clear content
    ss.clear(); // clear error flags
    ss << std::setw(name_len) << std::setfill(' ') << hash_names[i] << ": "
       << std::hex << std::setfill('0') << std::setw(16) << accum[i];
    std::cout << ss.str() << std::endl;
  }
}

std::string determine_exe_name(const std::string argv0) {
  size_t last_slash = argv0.find_last_of("/");
  if (last_slash != std::string::npos) {
    return argv0.substr(last_slash + 1);
  }
  return argv0;
}

} // namespace

namespace mam4 {
namespace validation {

void initialize(int argc, char **argv) {
  exe_name_ = determine_exe_name(argv[0]);
  Kokkos::initialize(argc, argv);
}

void initialize(int argc, char **argv, int fpes_) {
  exe_name_ = determine_exe_name(argv[0]);
  Kokkos::initialize(argc, argv);
  ekat::enable_fpes(fpes_);
}

void finalize(Ensemble *ensemble) {
  print_bfbhash(ensemble);
  delete ensemble;
  testing::finalize();
  Kokkos::finalize();
}

void finalize(std::unique_ptr<Ensemble> &ensemble) {
  print_bfbhash(ensemble.get());
  testing::finalize();
  Kokkos::finalize();
}

std::string output_name(const std::string &input_file) {
  std::string output_file;
  size_t slash = input_file.find_last_of('/');
  size_t dot = input_file.find_last_of('.');
  if ((dot == std::string::npos) and (slash == std::string::npos)) {
    dot = input_file.length();
  }
  if (slash == std::string::npos) {
    slash = 0;
  } else {
    slash += 1;
    dot -= slash;
  }
  return std::string("mam4xx_") + input_file.substr(slash, dot) +
         std::string(".py");
}

void convert_vector_to_mass_mixing_ratios(
    const std::vector<Real> &vector_in,
    Real values[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()]) {
  int count = 0;
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
      values[m][ispec] = vector_in[count];
      count++;
    }
  }
}

void convert_modal_array_to_vector(const Real values[AeroConfig::num_modes()],
                                   std::vector<Real> &values_vector) {
  for (int i = 0; i < AeroConfig::num_modes(); ++i)
    values_vector[i] = values[i];
}

void convert_vector_to_modal_array(const std::vector<Real> &vector_in,
                                   Real values[AeroConfig::num_modes()]) {
  for (int m = 0; m < AeroConfig::num_modes(); ++m)
    values[m] = vector_in[m];
}

void convert_mass_mixing_ratios_to_vector(
    const Real values[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
    std::vector<Real> &values_vector) {
  int count = 0;
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
      values_vector[count] = values[m][ispec];
      count++;
    }
  }
}

void create_synthetic_rsf_tab(View5D &rsf_tab, const int nw, const int nump,
                              const int numsza, const int numcolo3,
                              const int numalb, Real *synthetic_values) {

  rsf_tab = View5D("rsf_tab", nw, nump, numsza, numcolo3, numalb);

  auto rsf_tab_1 = Kokkos::subview(rsf_tab, Kokkos::ALL(), 1, Kokkos::ALL(),
                                   Kokkos::ALL(), Kokkos::ALL());

  auto rsf_tab_2 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(), 6,
                                   Kokkos::ALL(), Kokkos::ALL());

  auto rsf_tab_3 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(),
                                   Kokkos::ALL(), 7, Kokkos::ALL());

  auto rsf_tab_4 = Kokkos::subview(rsf_tab, Kokkos::ALL(), Kokkos::ALL(),
                                   Kokkos::ALL(), Kokkos::ALL(), 3);

  auto rsf_tab_5 = Kokkos::subview(rsf_tab, 0, Kokkos::ALL(), Kokkos::ALL(),
                                   Kokkos::ALL(), Kokkos::ALL());

  auto rsf_tab_6 = Kokkos::subview(rsf_tab, 9, Kokkos::ALL(), Kokkos::ALL(),
                                   Kokkos::ALL(), Kokkos::ALL());

  Kokkos::deep_copy(rsf_tab, synthetic_values[0]);
  Kokkos::deep_copy(rsf_tab_1, synthetic_values[1]);
  Kokkos::deep_copy(rsf_tab_2, synthetic_values[2]);
  Kokkos::deep_copy(rsf_tab_3, synthetic_values[3]);
  Kokkos::deep_copy(rsf_tab_4, synthetic_values[4]);
  Kokkos::deep_copy(rsf_tab_5, synthetic_values[5]);
  Kokkos::deep_copy(rsf_tab_6, synthetic_values[6]);
}

void convert_1d_vector_to_2d_view_device(const std::vector<Real> &var_std,
                                         const View2D &var_device) {
  auto host = Kokkos::create_mirror_view(var_device);
  int count = 0;
  for (int d2 = 0; d2 < var_device.extent(1); ++d2) {
    for (int d1 = 0; d1 < var_device.extent(0); ++d1) {
      host(d1, d2) = var_std[count];
      count++;
    }
  }
  Kokkos::deep_copy(var_device, host);
}

void convert_1d_vector_to_transpose_2d_view_device(
    const std::vector<Real> &var_std, const View2D &var_device) {
  auto host = Kokkos::create_mirror_view(var_device);
  int count = 0;
  for (int d1 = 0; d1 < var_device.extent(0); ++d1) {
    for (int d2 = 0; d2 < var_device.extent(1); ++d2) {
      host(d1, d2) = var_std[count];
      count++;
    }
  }
  Kokkos::deep_copy(var_device, host);
}

void convert_1d_real_to_2d_view_device(const Real var_std[],
                                       const View2D &var_device) {
  auto host = Kokkos::create_mirror_view(var_device);
  int count = 0;
  for (int d2 = 0; d2 < var_device.extent(1); ++d2) {
    for (int d1 = 0; d1 < var_device.extent(0); ++d1) {
      host(d1, d2) = var_std[count];
      count++;
    }
  }
  Kokkos::deep_copy(var_device, host);
}

void convert_2d_view_device_to_1d_vector(const View2D &var_device,
                                         std::vector<Real> &var_std) {
  auto host = Kokkos::create_mirror_view(var_device);
  Kokkos::deep_copy(host, var_device);
  int count = 0;
  for (int d2 = 0; d2 < var_device.extent(1); ++d2) {
    for (int d1 = 0; d1 < var_device.extent(0); ++d1) {
      var_std[count] = host(d1, d2);
      count++;
    }
  }
}

void convert_2d_view_int_device_to_1d_vector(const View2DInt &var_device,
                                             std::vector<Real> &var_std) {
  auto host = Kokkos::create_mirror_view(var_device);
  Kokkos::deep_copy(host, var_device);
  int count = 0;
  for (int d2 = 0; d2 < var_device.extent(1); ++d2) {
    for (int d1 = 0; d1 < var_device.extent(0); ++d1) {
      var_std[count] = host(d1, d2);
      count++;
    }
  }
}

void convert_transpose_2d_view_device_to_1d_vector(const View2D &var_device,
                                                   std::vector<Real> &var_std) {
  auto host = Kokkos::create_mirror_view(var_device);
  Kokkos::deep_copy(host, var_device);
  int count = 0;
  for (int d1 = 0; d1 < var_device.extent(0); ++d1) {
    for (int d2 = 0; d2 < var_device.extent(1); ++d2) {
      var_std[count] = host(d1, d2);
      count++;
    }
  }
}

void convert_1d_vector_to_3d_view_device(const std::vector<Real> &var_std,
                                         const View3D &var_device) {
  auto host = Kokkos::create_mirror_view(var_device);
  int count = 0;
  for (int d3 = 0; d3 < var_device.extent(2); ++d3) {
    for (int d2 = 0; d2 < var_device.extent(1); ++d2) {
      for (int d1 = 0; d1 < var_device.extent(0); ++d1) {
        host(d1, d2, d3) = var_std[count];
        count++;
      }
    }
  }
  Kokkos::deep_copy(var_device, host);
}

void convert_1d_vector_to_3d_view_int_device(const std::vector<Real> &var_std,
                                             const View3DInt &var_device) {
  auto host = Kokkos::create_mirror_view(var_device);
  int count = 0;
  for (int d3 = 0; d3 < var_device.extent(2); ++d3) {
    for (int d2 = 0; d2 < var_device.extent(1); ++d2) {
      for (int d1 = 0; d1 < var_device.extent(0); ++d1) {
        // make sure that we are using int.
        host(d1, d2, d3) = static_cast<int>(var_std[count]);
        count++;
      }
    }
  }
  Kokkos::deep_copy(var_device, host);
}

void convert_3d_view_device_to_1d_vector(const View3D &var_device,
                                         std::vector<Real> &var_std) {
  auto host = Kokkos::create_mirror_view(var_device);
  Kokkos::deep_copy(host, var_device);
  int count = 0;
  for (int d3 = 0; d3 < var_device.extent(2); ++d3) {
    for (int d2 = 0; d2 < var_device.extent(1); ++d2) {
      for (int d1 = 0; d1 < var_device.extent(0); ++d1) {
        var_std[count] = host(d1, d2, d3);
        count++;
      }
    }
  }
}

// given input from skywalker, return a ColumnView with data from yaml file.
ColumnView get_input_in_columnview(const skywalker::Input &input,
                                   const std::string &name) {
  using View1DHost = typename haero::HostType::view_1d<Real>;
  int nlev = mam4::nlev;
  const auto host_vector = input.get_array(name);
  // inputs needs to be nlev.
  EKAT_ASSERT(host_vector.size() == nlev);
  ColumnView dev = haero::testing::create_column_view(nlev);
  auto host = View1DHost((Real *)host_vector.data(), nlev);
  Kokkos::deep_copy(dev, host);
  return dev;
}

} // namespace validation
} // namespace mam4
