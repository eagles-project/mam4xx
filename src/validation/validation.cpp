#include "validation.hpp"


namespace mam4 {
namespace validation {

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

 void 
 convert1D_Vector2D_RealMixRatios(const std::vector<Real> &vector_in,
                          Real values[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()]) {
  int count = 0;
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
      values[m][ispec] = vector_in[count];
      count++;
    }
  }
}

void 
convert1D_RealNumMode1D_Vector(const Real values[AeroConfig::num_modes()],
                           std::vector<Real> &values_vector) {
  for (int i = 0; i < AeroConfig::num_modes(); ++i)
    values_vector[i] = values[i];
}

void 
convert1D_Vector1D_RealNumMode(const std::vector<Real> &vector_in,
                               Real values[AeroConfig::num_modes()]) {
  for (int m = 0; m < AeroConfig::num_modes(); ++m) 
    values[m] = vector_in[m];
}


void 
convert2D_RealMixRatios1D_Vector(const Real values[AeroConfig::num_modes()][AeroConfig::num_aerosol_ids()],
                                   std::vector<Real> &values_vector) {
  int count = 0;
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    for (int ispec = 0; ispec < AeroConfig::num_aerosol_ids(); ++ispec) {
      values_vector[count] = values[m][ispec];
      count++;
    }
  }
}

} // namespace validation
} // namespace mam4
