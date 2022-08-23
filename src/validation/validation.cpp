#include "validation.hpp"

namespace haero {
namespace validation {

std::string output_name(const std::string& input_file) {
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
  return std::string("haero_") + input_file.substr(slash, dot) +
         std::string(".py");
}

}  // namespace validation
}  // namespace haero
