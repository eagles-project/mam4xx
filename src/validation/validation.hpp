#ifndef HAERO_VALIDATION_HPP
#define HAERO_VALIDATION_HPP

#include <string>

namespace haero {
namespace validation {

/// Given the name of a Skywalker input YAML file, determine the name of the
/// corresponding output Python module file.
/// @param [in] input_file The name of the Skywalker input YAML file.
std::string output_name(const std::string& input_file);

}  // namespace validation
}  // namespace haero

#endif
