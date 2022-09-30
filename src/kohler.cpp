#include "kohler.hpp"

namespace mam4 {

// ETI
// double precison is required by the KohlerPolynomial class, so we intstantiate
// the two most common types here.
template struct KohlerPolynomial<PackType>;
template struct KohlerPolynomial<double>;
template struct KohlerSolver<haero::math::NewtonSolver<KohlerPolynomial<PackType>>>;

}  // namespace haero
