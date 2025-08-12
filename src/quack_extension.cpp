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
#include "utils.hpp"

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

// Generic sampling function template for single parameter distributions
template <typename DistributionType, typename ReturnType>
inline void GenericSingleParamSampleFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &param_vector = args.data[0];

	if (param_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(param_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		const auto param = ConstantVector::GetData<double>(param_vector)[0];

		// Create distribution once and reuse for constant vectors
		DistributionType dist(param);
		const auto results = FlatVector::GetData<ReturnType>(result);

		for (size_t i = 0; i < args.size(); i++) {
			results[i] = static_cast<ReturnType>(dist(rng));
		}

		// Set vector type appropriately
		if (args.size() == 1) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	// Handle non-constant vectors
	UnaryExecutor::Execute<double, ReturnType>(param_vector, result, args.size(), [&](double param) {
		DistributionType dist(param);
		return static_cast<ReturnType>(dist(rng));
	});
}

// Generic sampling function template for two parameter distributions
template <typename DistributionType, typename ReturnType>
inline void GenericTwoParamSampleFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &param1_vector = args.data[0];
	auto &param2_vector = args.data[1];

	BinaryExecutor::Execute<double, double, ReturnType>(param1_vector, param2_vector, result, args.size(),
	                                                    [&](double param1, double param2) {
		                                                    DistributionType dist(param1, param2);
		                                                    return static_cast<ReturnType>(dist(rng));
	                                                    });
}

// Specialized Bernoulli sampling function with validation
inline void BernoulliSampleFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	GenericSingleParamSampleFunc<boost::random::bernoulli_distribution<double>, int8_t>(args, state, result);
}

inline void NormalRandFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	GenericTwoParamSampleFunc<boost::random::normal_distribution<double>, double>(args, state, result);
}

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
	RegisterSingleParamSamplingFunction(instance, "bernoulli_sample", LogicalType::TINYINT, BernoulliSampleFunc,
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
