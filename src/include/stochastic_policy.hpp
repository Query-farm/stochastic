#pragma once

#include <boost/math/policies/error_handling.hpp>
#include <boost/math/policies/policy.hpp>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>

namespace duckdb {

//! Raised by the Boost.Math user error handlers below. The distribution executors turn it into an
//! InvalidInputException that names the distribution, or into NULL for undefined properties.
class MathError : public std::exception {
public:
	enum class Kind { INVALID_ARGUMENT, POLE, EVALUATION, ROUNDING };

	MathError(Kind kind, std::string message) : kind(kind), message(std::move(message)) {
	}

	const char *what() const noexcept override {
		return message.c_str();
	}

	Kind kind;
	std::string message;
};

//! Replaces wording in Boost messages that refers to its internals.
inline void RewriteBoostPhrases(std::string &result) {
	// Boost's root finders name themselves in this message
	const std::string no_root =
	    "There appears to be no root to be found in boost::math::tools::newton_raphson_iterate, "
	    "perhaps we have a local minima near current best guess of";
	const auto pos = result.find(no_root);
	if (pos != std::string::npos) {
		result.replace(pos, no_root.size(), "no solution was found; the closest value was");
	}
}

//! Removes Boost's "Error in function <signature>: " prefix and trailing punctuation from an exception message.
inline std::string CleanBoostMessage(const std::string &message) {
	std::string result = message;
	const std::string prefix = "Error in function ";
	if (result.compare(0, prefix.size(), prefix) == 0) {
		const auto separator = result.find(": ");
		result = separator == std::string::npos ? result.substr(prefix.size()) : result.substr(separator + 2);
	}
	RewriteBoostPhrases(result);
	while (!result.empty() && (result.back() == ' ' || result.back() == '!')) {
		result.pop_back();
	}
	return result;
}

template <class T>
std::string FormatMathErrorMessage(const char *message, const T &val) {
	std::string result = message ? message : "invalid argument";
	std::ostringstream formatted;
	formatted << std::setprecision(15) << val;
	for (auto pos = result.find("%1%"); pos != std::string::npos; pos = result.find("%1%", pos)) {
		result.replace(pos, 3, formatted.str());
	}
	RewriteBoostPhrases(result);
	while (!result.empty() && (result.back() == ' ' || result.back() == '!')) {
		result.pop_back();
	}
	return result;
}

//! Domain, pole, evaluation and rounding errors are reported through MathError. Overflow returns +/-infinity
//! (e.g. the normal quantile at p = 1) instead of raising an error. Calculations stay in DOUBLE: promoting to
//! long double is ~100x slower where it is software emulated (Linux aarch64) and makes results differ between
//! platforms.
using StochasticPolicy =
    boost::math::policies::policy<boost::math::policies::domain_error<boost::math::policies::user_error>,
                                  boost::math::policies::pole_error<boost::math::policies::user_error>,
                                  boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
                                  boost::math::policies::evaluation_error<boost::math::policies::user_error>,
                                  boost::math::policies::rounding_error<boost::math::policies::user_error>,
                                  boost::math::policies::promote_double<false>>;

//! StochasticPolicy for distributions with properties that are always undefined (e.g. the mean of a Cauchy
//! distribution): those raise a domain error at runtime rather than failing to compile. Only use it where it is
//! needed - with Apple clang it leaves Boost's exp_sinh quadrature (used by the non-central t) with unresolved
//! symbols.
using StochasticUndefinedPolicy =
    boost::math::policies::policy<boost::math::policies::domain_error<boost::math::policies::user_error>,
                                  boost::math::policies::pole_error<boost::math::policies::user_error>,
                                  boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
                                  boost::math::policies::evaluation_error<boost::math::policies::user_error>,
                                  boost::math::policies::rounding_error<boost::math::policies::user_error>,
                                  boost::math::policies::promote_double<false>,
                                  boost::math::policies::assert_undefined<false>>;

//! Discrete distributions round quantiles to the smallest integer x with CDF(x) >= p, which is also what
//! inverse-transform sampling needs.
using StochasticDiscretePolicy =
    boost::math::policies::policy<boost::math::policies::domain_error<boost::math::policies::user_error>,
                                  boost::math::policies::pole_error<boost::math::policies::user_error>,
                                  boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
                                  boost::math::policies::evaluation_error<boost::math::policies::user_error>,
                                  boost::math::policies::rounding_error<boost::math::policies::user_error>,
                                  boost::math::policies::promote_double<false>,
                                  boost::math::policies::discrete_quantile<boost::math::policies::integer_round_up>>;

} // namespace duckdb

namespace boost {
namespace math {
namespace policies {

template <class T>
T user_domain_error(const char *, const char *message, const T &val) {
	throw duckdb::MathError(duckdb::MathError::Kind::INVALID_ARGUMENT, duckdb::FormatMathErrorMessage(message, val));
}

template <class T>
T user_pole_error(const char *, const char *message, const T &val) {
	throw duckdb::MathError(duckdb::MathError::Kind::POLE, duckdb::FormatMathErrorMessage(message, val));
}

template <class T>
T user_evaluation_error(const char *, const char *message, const T &val) {
	throw duckdb::MathError(duckdb::MathError::Kind::EVALUATION, duckdb::FormatMathErrorMessage(message, val));
}

template <class T, class TargetType>
TargetType user_rounding_error(const char *, const char *message, const T &val, const TargetType &) {
	throw duckdb::MathError(duckdb::MathError::Kind::ROUNDING, duckdb::FormatMathErrorMessage(message, val));
}

} // namespace policies
} // namespace math
} // namespace boost
