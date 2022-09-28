#include "kohler.hpp"

#include <iomanip>
#include <sstream>
#include <fstream>

namespace mam4 {



// ETI
// double precison is required by the KohlerPolynomial class, so we intstantiate
// the two most common types here.
template struct KohlerPolynomial<PackType>;
template struct KohlerPolynomial<double>;
template struct KohlerSolver<haero::math::NewtonSolver<KohlerPolynomial<PackType>>>;
// template struct KohlerSolver<haero::math::BisectionSolver<KohlerPolynomial<PackType>>>;
// template struct KohlerSolver<haero::math::BracketedNewtonSolver<KohlerPolynomial<PackType>>>;

}  // namespace haero
