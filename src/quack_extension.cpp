#define DUCKDB_EXTENSION_MAIN

#include "quack_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random.hpp>
#include <random>
#include <thread>

namespace duckdb {

// Global seed for all RNG streams
constexpr unsigned int GLOBAL_SEED = 12345;

// Map from std::thread::id to fixed thread index
std::unordered_map<std::thread::id, unsigned int> thread_id_map;
std::mutex thread_id_map_mutex;
unsigned int next_thread_index = 0;

unsigned int get_thread_index() {
	auto tid = std::this_thread::get_id();
	std::lock_guard<std::mutex> lock(thread_id_map_mutex);
	auto it = thread_id_map.find(tid);
	if (it == thread_id_map.end()) {
		unsigned int idx = next_thread_index++;
		thread_id_map[tid] = idx;
		return idx;
	}
	return it->second;
}

thread_local boost::random::mt19937 rng = [] {
	unsigned int tidx = get_thread_index();
	std::seed_seq seq {GLOBAL_SEED, tidx};
	std::vector<uint32_t> seed_data(1);
	seq.generate(seed_data.begin(), seed_data.end());

	boost::random::mt19937 local_rng;
	local_rng.seed(seed_data[0]);
	return local_rng;
}();

// Generic template for any distribution with two parameters
template <typename Distribution, typename Operation>
inline void TwoParameterDistributionFunc(DataChunk &args, ExpressionState &state, Vector &result, Operation op) {
	auto &param1_vector = args.data[0];
	auto &param2_vector = args.data[1];
	auto &x_vector = args.data[2];

	// If these parameters are constant vectors they should be treated as such.
	if (param1_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    param2_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {

		if (ConstantVector::IsNull(param1_vector) || ConstantVector::IsNull(param2_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}
		const auto param1 = ConstantVector::GetData<double>(param1_vector)[0];
		const auto param2 = ConstantVector::GetData<double>(param2_vector)[0];
		Distribution dist(param1, param2);
		UnaryExecutor::Execute<double, double>(x_vector, result, args.size(), [&](double x) { return op(dist, x); });
	}

	TernaryExecutor::Execute<double, double, double, double>(param1_vector, param2_vector, x_vector, result,
	                                                         args.size(), [&](double param1, double param2, double x) {
		                                                         Distribution dist(param1, param2);
		                                                         return op(dist, x);
	                                                         });
}

// Generic macro to define distribution functions for any distribution type
#define DEFINE_DISTRIBUTION_FUNC(func_name, distribution_type, boost_operation)                                        \
	inline void func_name(DataChunk &args, ExpressionState &state, Vector &result) {                                   \
		TwoParameterDistributionFunc<distribution_type>(args, state, result,                                           \
		                                                [](const auto &dist, auto x) { return boost_operation; });     \
	}

// Generate all normal distribution functions using the macro
DEFINE_DISTRIBUTION_FUNC(NormalPdfFunc, boost::math::normal_distribution<double>, boost::math::pdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(NormalLogPdfFunc, boost::math::normal_distribution<double>, boost::math::logpdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(NormalCdfFunc, boost::math::normal_distribution<double>, boost::math::cdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(NormalLogCdfFunc, boost::math::normal_distribution<double>, boost::math::logcdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(NormalQuantileFunc, boost::math::normal_distribution<double>, boost::math::quantile(dist, x))

DEFINE_DISTRIBUTION_FUNC(NormalCdfComplementFunc, boost::math::normal_distribution<double>,
                         boost::math::cdf(boost::math::complement(dist, x)))
DEFINE_DISTRIBUTION_FUNC(NormalLogCdfComplementFunc, boost::math::normal_distribution<double>,
                         boost::math::logcdf(boost::math::complement(dist, x)))
DEFINE_DISTRIBUTION_FUNC(NormalQuantileComplementFunc, boost::math::normal_distribution<double>,
                         boost::math::quantile(boost::math::complement(dist, x)))

// Bernoulli distribution functions using single parameter (p)
// Template for single-parameter distributions
template <typename Distribution, typename Operation>
inline void SingleParameterDistributionFunc(DataChunk &args, ExpressionState &state, Vector &result, Operation op) {
	auto &param_vector = args.data[0];
	auto &x_vector = args.data[1];

	// If parameter is constant, optimize the computation
	if (param_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(param_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}
		const auto param = ConstantVector::GetData<double>(param_vector)[0];
		Distribution dist(param);
		UnaryExecutor::Execute<double, double>(x_vector, result, args.size(), [&](double x) { return op(dist, x); });
	} else {
		BinaryExecutor::Execute<double, double, double>(param_vector, x_vector, result, args.size(),
		                                                [&](double param, double x) {
			                                                Distribution dist(param);
			                                                return op(dist, x);
		                                                });
	}
}

// Generic macro to define single-parameter distribution functions
#define DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(func_name, distribution_type, boost_operation)                           \
	inline void func_name(DataChunk &args, ExpressionState &state, Vector &result) {                                   \
		SingleParameterDistributionFunc<distribution_type>(args, state, result,                                        \
		                                                   [](const auto &dist, auto x) { return boost_operation; });  \
	}

// Generate Bernoulli distribution functions
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliPdfFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::pdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliLogPdfFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::logpdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliCdfFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::cdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliLogCdfFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::logcdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliQuantileFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::quantile(dist, x))

DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliCdfComplementFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::cdf(boost::math::complement(dist, x)))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliLogCdfComplementFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::logcdf(boost::math::complement(dist, x)))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliQuantileComplementFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::quantile(boost::math::complement(dist, x)))

// Bernoulli sampling function
inline void BernoulliSampleFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &p_vector = args.data[0];

	UnaryExecutor::Execute<double, double>(p_vector, result, args.size(), [&](double p) {
		boost::random::bernoulli_distribution<double> dist(p);
		return static_cast<double>(dist(rng));
	});
}

/*
// Example: To add Gamma distribution, you would just do:
DEFINE_DISTRIBUTION_FUNC(GammaPdfFunc, boost::math::gamma_distribution<double>, pdf)
DEFINE_DISTRIBUTION_FUNC(GammaCdfFunc, boost::math::gamma_distribution<double>, cdf)
// etc.
*/

// Normal distribution sampling functions using boost::random
// These functions generate random samples from a normal distribution with given parameters

// NormalRandFunc: Generate random samples from Normal(mean, stddev)
// Uses thread-local random number generator for performance
inline void NormalRandFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &mean_vector = args.data[0];
	auto &stddev_vector = args.data[1];

	BinaryExecutor::Execute<double, double, double>(mean_vector, stddev_vector, result, args.size(),
	                                                [&](double mean, double stddev) {
		                                                boost::random::normal_distribution<double> dist(mean, stddev);
		                                                return dist(rng);
	                                                });
}

// Helper function to create a scalar function with consistent signature
template <typename FuncType>
ScalarFunction CreateThreeParameterDistributionFunction(const std::string &name, FuncType func) {
	return ScalarFunction(name, {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      func);
}

// Helper function to create a binary scalar function (for sampling without seed)
template <typename FuncType>
ScalarFunction CreateBinaryFunction(const std::string &name, FuncType func) {
	return ScalarFunction(name, {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE, func, nullptr, nullptr,
	                      nullptr, nullptr, LogicalTypeId::INVALID, FunctionStability::VOLATILE,
	                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr);
}

// Helper function to create a single-parameter distribution function
template <typename FuncType>
ScalarFunction CreateTwoParameterDistributionFunction(const std::string &name, FuncType func) {
	return ScalarFunction(name, {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE, func);
}

// Helper function to create a unary scalar function (for single-parameter sampling)
template <typename FuncType>
ScalarFunction CreateUnaryFunction(const std::string &name, FuncType func) {
	return ScalarFunction(name, {LogicalType::DOUBLE}, LogicalType::DOUBLE, func, nullptr, nullptr, nullptr, nullptr,
	                      LogicalTypeId::INVALID, FunctionStability::VOLATILE,
	                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr);
}

// Helper function to register a three-parameter distribution function with description
template <typename FuncType>
void RegisterDistributionFunction(DatabaseInstance &instance, const std::string &name, FuncType func,
                                  const std::string &description, const std::string &example,
                                  const duckdb::vector<std::string> &param_names,
                                  const duckdb::vector<LogicalType> &param_types) {
	CreateScalarFunctionInfo info(CreateThreeParameterDistributionFunction(name, func));
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(example);
	desc.parameter_names = param_names;
	desc.parameter_types = param_types;
	info.descriptions.push_back(desc);
	ExtensionUtil::RegisterFunction(instance, info);
}

// Helper function to register a two-parameter sampling function with description
template <typename FuncType>
void RegisterSamplingFunction(DatabaseInstance &instance, const std::string &name, FuncType func,
                              const std::string &description, const std::string &example,
                              const duckdb::vector<std::string> &param_names,
                              const duckdb::vector<LogicalType> &param_types) {
	CreateScalarFunctionInfo info(CreateBinaryFunction(name, func));
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(example);
	desc.parameter_names = param_names;
	desc.parameter_types = param_types;
	info.descriptions.push_back(desc);
	ExtensionUtil::RegisterFunction(instance, info);
}

// Helper function to register a single-parameter distribution function with description
template <typename FuncType>
void RegisterSingleParamDistributionFunction(DatabaseInstance &instance, const std::string &name, FuncType func,
                                             const std::string &description, const std::string &example,
                                             const duckdb::vector<std::string> &param_names,
                                             const duckdb::vector<LogicalType> &param_types) {
	CreateScalarFunctionInfo info(CreateTwoParameterDistributionFunction(name, func));
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(example);
	desc.parameter_names = param_names;
	desc.parameter_types = param_types;
	info.descriptions.push_back(desc);
	ExtensionUtil::RegisterFunction(instance, info);
}

// Helper function to register a single-parameter sampling function with description
template <typename FuncType>
void RegisterSingleParamSamplingFunction(DatabaseInstance &instance, const std::string &name, FuncType func,
                                         const std::string &description, const std::string &example,
                                         const duckdb::vector<std::string> &param_names,
                                         const duckdb::vector<LogicalType> &param_types) {
	CreateScalarFunctionInfo info(CreateUnaryFunction(name, func));
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(example);
	desc.parameter_names = param_names;
	desc.parameter_types = param_types;
	info.descriptions.push_back(desc);
	ExtensionUtil::RegisterFunction(instance, info);
}

// Macro to register normal distribution functions with less boilerplate
#define REGISTER_NORMAL_FUNC(instance, func_name, func_ptr, description, example, third_param)                         \
	RegisterDistributionFunction(instance, func_name, func_ptr, description, example, {"mean", "stddev", third_param}, \
	                             normal_param_types)

// Macro to register Bernoulli distribution functions with less boilerplate
#define REGISTER_BERNOULLI_FUNC(instance, func_name, func_ptr, description, example, second_param)                     \
	RegisterSingleParamDistributionFunction(instance, func_name, func_ptr, description, example, {"p", second_param},  \
	                                        bernoulli_param_types)

static void LoadInternal(DatabaseInstance &instance) {
	// Normal distribution parameter types (reused for all functions)
	const duckdb::vector<LogicalType> normal_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE,
	                                                        LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> normal_sample_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};

	// Bernoulli distribution parameter types
	const duckdb::vector<LogicalType> bernoulli_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> bernoulli_sample_param_types = {LogicalType::DOUBLE};

	REGISTER_NORMAL_FUNC(
	    instance, "normal_pdf", NormalPdfFunc,
	    "Computes the probability density function (PDF) of the normal distribution. Returns the probability density "
	    "at point x for a normal distribution with specified mean and standard deviation.",
	    "normal_pdf(0.0, 1.0, 0.5)", "x");

	REGISTER_NORMAL_FUNC(instance, "normal_log_pdf", NormalLogPdfFunc,
	                     "Computes the natural logarithm of the probability density function (log-PDF) of the normal "
	                     "distribution. Useful for numerical stability when dealing with very small probabilities.",
	                     "normal_log_pdf(0.0, 1.0, 0.5)", "x");

	REGISTER_NORMAL_FUNC(instance, "normal_cdf", NormalCdfFunc,
	                     "Computes the cumulative distribution function (CDF) of the normal distribution. Returns the "
	                     "probability that a random variable X is less than or equal to x.",
	                     "normal_cdf(0.0, 1.0, 0.5)", "x");

	REGISTER_NORMAL_FUNC(instance, "normal_cdf_complement", NormalCdfComplementFunc,
	                     "Computes the complementary cumulative distribution function (1 - CDF) of the normal "
	                     "distribution. Returns the probability that X > x, equivalent to the survival function.",
	                     "normal_cdf_complement(0.0, 1.0, 0.5)", "x");

	REGISTER_NORMAL_FUNC(instance, "normal_quantile", NormalQuantileFunc,
	                     "Computes the quantile function (inverse CDF) of the normal distribution. Returns the value x "
	                     "such that P(X ≤ x) = p, where p is the cumulative probability.",
	                     "normal_quantile(0.0, 1.0, 0.5)", "p");

	REGISTER_NORMAL_FUNC(instance, "normal_quantile_complement", NormalQuantileComplementFunc,
	                     "Computes the complementary quantile function of the normal distribution. Returns the value x "
	                     "such that P(X > x) = p, useful for computing upper tail quantiles.",
	                     "normal_quantile_complement(0.0, 1.0, 0.5)", "p");

	// Sampling function
	RegisterSamplingFunction(
	    instance, "normal_sample", NormalRandFunc,
	    "Generates random samples from the normal distribution with specified mean and standard deviation.",
	    "normal_sample(0.0, 1.0)", {"mean", "stddev"}, normal_sample_param_types);

	// Register Bernoulli distribution functions
	REGISTER_BERNOULLI_FUNC(instance, "bernoulli_pdf", BernoulliPdfFunc,
	                        "Computes the probability mass function (PMF) of the Bernoulli distribution. Returns the "
	                        "probability of getting value x (0 or 1) for a Bernoulli trial with success probability p.",
	                        "bernoulli_pdf(0.3, 1)", "x");

	REGISTER_BERNOULLI_FUNC(
	    instance, "bernoulli_cdf", BernoulliCdfFunc,
	    "Computes the cumulative distribution function (CDF) of the Bernoulli distribution. Returns the probability "
	    "that X ≤ x for a Bernoulli distribution with success probability p.",
	    "bernoulli_cdf(0.3, 1)", "x");

	REGISTER_BERNOULLI_FUNC(instance, "bernoulli_quantile", BernoulliQuantileFunc,
	                        "Computes the quantile function (inverse CDF) of the Bernoulli distribution. Returns the "
	                        "smallest value x such that P(X ≤ x) ≥ p.",
	                        "bernoulli_quantile(0.3, 0.5)", "p");

	// Log functions
	REGISTER_BERNOULLI_FUNC(
	    instance, "bernoulli_log_pdf", BernoulliLogPdfFunc,
	    "Computes the natural logarithm of the probability mass function (log-PMF) of the Bernoulli distribution. "
	    "Useful for numerical stability when dealing with very small probabilities.",
	    "bernoulli_log_pdf(0.3, 1)", "x");

	REGISTER_BERNOULLI_FUNC(instance, "bernoulli_log_cdf", BernoulliLogCdfFunc,
	                        "Computes the natural logarithm of the cumulative distribution function (log-CDF) of the "
	                        "Bernoulli distribution. Useful for numerical stability in extreme tail calculations.",
	                        "bernoulli_log_cdf(0.3, 1)", "x");

	// Complement functions
	REGISTER_BERNOULLI_FUNC(instance, "bernoulli_cdf_complement", BernoulliCdfComplementFunc,
	                        "Computes the complementary cumulative distribution function (1 - CDF) of the Bernoulli "
	                        "distribution. Returns the probability that X > x, equivalent to the survival function.",
	                        "bernoulli_cdf_complement(0.3, 0)", "x");

	REGISTER_BERNOULLI_FUNC(
	    instance, "bernoulli_log_cdf_complement", BernoulliLogCdfComplementFunc,
	    "Computes the natural logarithm of the complementary cumulative distribution function of the Bernoulli "
	    "distribution. Useful for numerical stability in tail probability calculations.",
	    "bernoulli_log_cdf_complement(0.3, 0)", "x");

	REGISTER_BERNOULLI_FUNC(instance, "bernoulli_quantile_complement", BernoulliQuantileComplementFunc,
	                        "Computes the complementary quantile function of the Bernoulli distribution. Returns the "
	                        "value x such that P(X > x) = p, useful for computing upper tail quantiles.",
	                        "bernoulli_quantile_complement(0.3, 0.3)", "p");

	// Bernoulli sampling function
	RegisterSingleParamSamplingFunction(instance, "bernoulli_sample", BernoulliSampleFunc,
	                                    "Generates random samples from the Bernoulli distribution with success "
	                                    "probability p. Returns 1 with probability p, 0 with probability (1-p).",
	                                    "bernoulli_sample(0.3)", {"p"}, bernoulli_sample_param_types);
}

void QuackExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string QuackExtension::Name() {
	return "stochastic";
}

std::string QuackExtension::Version() const {
	return "0.0.1";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void quack_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::QuackExtension>();
}

DUCKDB_EXTENSION_API const char *quack_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
