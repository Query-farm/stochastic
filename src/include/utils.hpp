#pragma once
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include "duckdb/common/vector/array_vector.hpp"
#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/common/vector/vector_iterator.hpp"
#include "duckdb/common/vector/vector_writer.hpp"
#include "stochastic_policy.hpp"
#include <boost/math/distributions.hpp>
#include <cmath>
#include <optional>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include "rng_adapter.hpp"
#include "custom_random.hpp"
#include "distribution_traits.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Distribution specs
//===--------------------------------------------------------------------===//
// Every distribution is described by a spec struct deriving from DistributionSpecBase:
//
//   struct NormalSpec : DistributionSpecBase {
//       static constexpr const char *name = "normal";            // dist_normal_*
//       static constexpr const char *display_name = "normal";    // used in descriptions
//       using params = std::tuple<double, double>;
//       static constexpr std::array<const char *, 2> param_names = {"mean", "stddev"};
//       using boost_type = boost::math::normal_distribution<double, StochasticPolicy>;
//       using sampler_type = std::normal_distribution<double>;
//       using sample_t = double;
//       static constexpr const char *example_params = "0.0, 1.0";
//       static constexpr const char *example_x = "0.5";
//       static void Validate(double mean, double stddev);
//   };
//
// Both boost_type and sampler_type are constructed from the parameters in declaration order.

struct DistributionSpecBase {
	static constexpr bool discrete = false;
	using quantile_t = double;
	static constexpr const char *example_p = "0.95";
	//! Reject parameters whose integer samples could exceed 2^53 (requires boost_type mean and stddev)
	static constexpr bool check_sample_range = true;
};

template <class Spec>
constexpr idx_t SpecParamCount() {
	return std::tuple_size<typename Spec::params>::value;
}

constexpr idx_t MAX_SAMPLES_PER_ROW = 10000000;

//===--------------------------------------------------------------------===//
// Parameter validation helpers
//===--------------------------------------------------------------------===//
template <class T>
std::string FormatParameter(T value) {
	std::ostringstream formatted;
	formatted << std::setprecision(15) << value;
	return formatted.str();
}

template <class T>
void RequireFinite(const char *dist, const char *param, T value) {
	if constexpr (std::is_floating_point<T>::value) {
		if (!std::isfinite(value)) {
			throw InvalidInputException("%s: %s must be finite, got %s", dist, param, FormatParameter(value));
		}
	}
}

template <class T>
void RequirePositive(const char *dist, const char *param, T value) {
	RequireFinite(dist, param, value);
	if (!(value > 0)) {
		throw InvalidInputException("%s: %s must be > 0, got %s", dist, param, FormatParameter(value));
	}
}

template <class T>
void RequireNonNegative(const char *dist, const char *param, T value) {
	RequireFinite(dist, param, value);
	if (!(value >= 0)) {
		throw InvalidInputException("%s: %s must be >= 0, got %s", dist, param, FormatParameter(value));
	}
}

inline void RequireProbability(const char *dist, const char *param, double value) {
	RequireFinite(dist, param, value);
	if (!(value >= 0 && value <= 1)) {
		throw InvalidInputException("%s: %s must be between 0 and 1, got %s", dist, param, FormatParameter(value));
	}
}

template <class Spec, class Tuple, std::size_t... Is>
void RequireFiniteParams(const Tuple &params, std::index_sequence<Is...>) {
	(RequireFinite(Spec::name, Spec::param_names[Is], std::get<Is>(params)), ...);
}

template <class Spec>
void ValidateParams(const typename Spec::params &params) {
	std::apply(Spec::Validate, params);
	RequireFiniteParams<Spec>(params, std::make_index_sequence<SpecParamCount<Spec>()> {});
}

template <class Spec>
typename Spec::boost_type MakeDistribution(const typename Spec::params &params) {
	ValidateParams<Spec>(params);
	return std::make_from_tuple<typename Spec::boost_type>(params);
}

//! Integer samplers compute in DOUBLE and then convert, so reject parameters whose samples could leave the
//! range where DOUBLE represents every integer exactly (the conversion would overflow, and some samplers loop
//! forever near that limit).
template <class Spec>
void CheckSampleRange(const typename Spec::params &params) {
	using sample_t = typename Spec::sample_t;
	if constexpr (Spec::check_sample_range && std::is_integral<sample_t>::value &&
	              !std::is_same<sample_t, bool>::value) {
		const auto dist = std::make_from_tuple<typename Spec::boost_type>(params);
		const double limit = std::ldexp(1.0, 53);
		const double bound = boost::math::mean(dist) + 40 * boost::math::standard_deviation(dist);
		if (!(bound < limit)) {
			throw InvalidInputException("%s: parameters are too extreme to sample, samples could exceed 2^53",
			                            Spec::name);
		}
	}
}

template <class Spec>
typename Spec::sampler_type MakeSampler(const typename Spec::params &params) {
	ValidateParams<Spec>(params);
	CheckSampleRange<Spec>(params);
	return std::make_from_tuple<typename Spec::sampler_type>(params);
}

//===--------------------------------------------------------------------===//
// Result output
//===--------------------------------------------------------------------===//
// ResultOutput<T> writes rows of type T either into a flat result vector or, when every input is constant,
// into a single constant row. std::optional<T> writes NULL for nullopt, std::pair<T, T> fills an ARRAY(T, 2)
// and std::vector<T> fills a LIST(T).

template <class T>
struct ResultOutput {
	ResultOutput(Vector &result, idx_t count, bool constant) : result(result), constant(constant) {
		if (constant) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		} else {
			writer.emplace(FlatVector::Writer<T>(result, count));
		}
	}
	void Write(const T &value) {
		if (constant) {
			ConstantVector::GetData<T>(result)[0] = value;
		} else {
			writer->WriteValue(value);
		}
	}
	void WriteNull() {
		if (constant) {
			ConstantVector::SetNull(result, true);
		} else {
			writer->WriteNull();
		}
	}

	Vector &result;
	bool constant;
	std::optional<VectorWriter<T>> writer;
};

template <class T>
struct ResultOutput<std::optional<T>> {
	ResultOutput(Vector &result, idx_t count, bool constant) : inner(result, count, constant) {
	}
	void Write(const std::optional<T> &value) {
		if (value) {
			inner.Write(*value);
		} else {
			inner.WriteNull();
		}
	}
	void WriteNull() {
		inner.WriteNull();
	}

	ResultOutput<T> inner;
};

template <class T>
struct ResultOutput<std::pair<T, T>> {
	ResultOutput(Vector &result, idx_t count, bool constant) : result(result), constant(constant) {
		if (constant) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		} else {
			result.SetVectorType(VectorType::FLAT_VECTOR);
		}
		data = FlatVector::GetDataMutable<T>(ArrayVector::GetChildMutable(result));
	}
	void Write(const std::pair<T, T> &value) {
		data[row * 2] = value.first;
		data[row * 2 + 1] = value.second;
		row++;
	}
	void WriteNull() {
		if (constant) {
			ConstantVector::SetNull(result, true);
		} else {
			FlatVector::SetNull(result, row, true);
		}
		row++;
	}

	Vector &result;
	bool constant;
	T *data;
	idx_t row = 0;
};

template <class T>
struct ResultOutput<std::vector<T>> {
	ResultOutput(Vector &result, idx_t count, bool constant)
	    : writer(FlatVector::Writer<VectorListType<T>>(result, count)) {
		D_ASSERT(!constant);
	}
	void Write(const std::vector<T> &values) {
		auto it = values.begin();
		for (auto &child_writer : writer.WriteList(values.size())) {
			child_writer.WriteValue(*it++);
		}
	}
	void WriteNull() {
		writer.WriteNull();
	}

	VectorWriter<VectorListType<T>> writer;
};

//===--------------------------------------------------------------------===//
// Executor
//===--------------------------------------------------------------------===//
// ExecuteDistribution reads the distribution parameters (args[0..N)) and any call arguments (args[N..)),
// builds an Object from the parameters with `make`, and writes `op(object, call_args...)` for every row.
// A NULL in any input gives NULL. The object is built once when the parameters are constant. Unless
// VOLATILE, constant inputs produce a single constant result.

template <class T>
struct ArgIterators;

template <class... Ts>
struct ArgIterators<std::tuple<Ts...>> {
	template <std::size_t... Is>
	ArgIterators(DataChunk &args, idx_t offset, std::index_sequence<Is...>) : iterators(args.data[offset + Is]...) {
	}

	template <std::size_t... Is>
	bool RowIsValid(idx_t row, std::index_sequence<Is...>) const {
		return (std::get<Is>(iterators)[row].IsValid() && ...);
	}
	bool RowIsValid(idx_t row) const {
		return RowIsValid(row, std::index_sequence_for<Ts...> {});
	}

	template <std::size_t... Is>
	std::tuple<Ts...> Get(idx_t row, std::index_sequence<Is...>) const {
		return std::tuple<Ts...>(std::get<Is>(iterators)[row].GetValue()...);
	}
	std::tuple<Ts...> Get(idx_t row) const {
		return Get(row, std::index_sequence_for<Ts...> {});
	}

	std::tuple<VectorIterator<Ts>...> iterators;
};

template <class Spec, class Result, bool VOLATILE, class CallArgs, class Make, class Op>
void ExecuteDistribution(DataChunk &args, Vector &result, Make &&make, Op &&op) {
	using Params = typename Spec::params;
	constexpr idx_t param_count = SpecParamCount<Spec>();
	constexpr idx_t call_count = std::tuple_size<CallArgs>::value;
	D_ASSERT(args.ColumnCount() == param_count + call_count);

	const idx_t count = args.size();
	bool params_constant = true;
	for (idx_t i = 0; i < param_count; i++) {
		params_constant = params_constant && args.data[i].GetVectorType() == VectorType::CONSTANT_VECTOR;
	}
	bool all_constant = !VOLATILE && params_constant;
	for (idx_t i = param_count; i < param_count + call_count; i++) {
		all_constant = all_constant && args.data[i].GetVectorType() == VectorType::CONSTANT_VECTOR;
	}

	try {
		ArgIterators<Params> params(args, 0, std::make_index_sequence<param_count> {});
		ArgIterators<CallArgs> calls(args, param_count, std::make_index_sequence<call_count> {});
		const idx_t rows = all_constant ? 1 : count;
		ResultOutput<Result> output(result, rows, all_constant);

		using Object = std::decay_t<decltype(make(std::declval<const Params &>()))>;
		std::optional<Object> shared_object;
		if (params_constant && count > 0 && params.RowIsValid(0)) {
			shared_object.emplace(make(params.Get(0)));
		}
		for (idx_t row = 0; row < rows; row++) {
			if (!params.RowIsValid(row) || !calls.RowIsValid(row)) {
				output.WriteNull();
				continue;
			}
			auto invoke = [&](Object &object) {
				output.Write(std::apply([&](auto &&...call_args) { return op(object, call_args...); }, calls.Get(row)));
			};
			if (shared_object) {
				invoke(*shared_object);
			} else {
				auto object = make(params.Get(row));
				invoke(object);
			}
		}
	} catch (MathError &error) {
		throw InvalidInputException("%s: %s", Spec::name, error.message);
	}
}

//! Wraps a property computation so that an undefined property (e.g. the mean of a Cauchy distribution, or the
//! variance of Student's t with df <= 1) yields NULL rather than an error.
template <class Fn>
auto UndefinedAsNull(Fn fn) {
	return [fn](const auto &dist) -> std::optional<double> {
		try {
			const double value = fn(dist);
			if (std::isnan(value)) {
				return std::nullopt;
			}
			return value;
		} catch (MathError &error) {
			if (error.kind == MathError::Kind::INVALID_ARGUMENT) {
				return std::nullopt;
			}
			throw;
		}
	};
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
template <class Spec, std::size_t... Is>
vector<LogicalType> SpecLogicalTypes(std::index_sequence<Is...>) {
	return {logical_type_map<std::tuple_element_t<Is, typename Spec::params>>::Get()...};
}

template <class Spec>
string SpecFunctionName(const string &name) {
	return string("dist_") + Spec::name + "_" + name;
}

struct FunctionDoc {
	string description;
	//! Arguments for the documented example call, e.g. "0.0, 1.0, 0.5"
	string example_args;
	//! Function category, e.g. "density" or "sampling"
	string category;
};

template <class Spec>
void RegisterFunction(ExtensionLoader &loader, const string &name, FunctionStability stability,
                      const LogicalType &result_type, scalar_function_t func, const FunctionDoc &doc,
                      const vector<std::pair<string, LogicalType>> &additional_params = {}) {
	auto final_types = SpecLogicalTypes<Spec>(std::make_index_sequence<SpecParamCount<Spec>()> {});
	vector<string> final_names(Spec::param_names.begin(), Spec::param_names.end());
	for (auto &item : additional_params) {
		final_types.push_back(item.second);
		final_names.push_back(item.first);
	}
	const auto final_name = SpecFunctionName<Spec>(name);

	auto function =
	    ScalarFunction(Identifier(final_name), final_types, result_type, std::move(func), nullptr, nullptr, nullptr,
	                   LogicalTypeId::INVALID, stability, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr);
	// Every distribution function validates its parameters and can raise an error.
	function.SetFallible();
	// DuckDB 2.0 reports parameter names from the signature (duckdb_functions()), so name the
	// (positional-only) parameters there as well. Signature names must be unique: a call parameter that
	// shares its name with a distribution parameter gets a suffix.
	for (idx_t i = 0; i < final_names.size(); i++) {
		auto param_name = final_names[i];
		for (idx_t j = 0; j < i; j++) {
			if (final_names[j] == final_names[i]) {
				param_name = final_names[i] + "_" + std::to_string(i + 1);
				break;
			}
		}
		function.GetSignature().GetParameter(i).SetName(Identifier(param_name));
	}

	CreateScalarFunctionInfo info(function);
	FunctionDescription desc;
	desc.description = doc.description;
	desc.examples.push_back(final_name + "(" + doc.example_args + ")");
	desc.parameter_types = final_types;
	desc.parameter_names = final_names;
	desc.categories = {"statistics", doc.category, Spec::name};
	info.descriptions.push_back(desc);
	loader.RegisterFunction(info);
}

//! Registers a CONSISTENT function computed from the Boost distribution and the call arguments CallArgs.
template <class Spec, class Result, class CallArgs = std::tuple<>, class Op>
void RegisterEvaluation(ExtensionLoader &loader, const string &name, Op op, const FunctionDoc &doc,
                        const vector<std::pair<string, LogicalType>> &call_params = {}) {
	using Output = std::conditional_t<std::is_same<Result, std::optional<double>>::value, double, Result>;
	RegisterFunction<Spec>(
	    loader, name, FunctionStability::CONSISTENT, logical_type_map<Output>::Get(),
	    [op](DataChunk &args, ExpressionState &, Vector &result) {
		    ExecuteDistribution<Spec, Result, false, CallArgs>(args, result, MakeDistribution<Spec>, op);
	    },
	    doc, call_params);
}

template <class Spec>
void RegisterSampling(ExtensionLoader &loader) {
	using sample_t = typename Spec::sample_t;
	const string params = Spec::example_params;
	const string text = string(Spec::display_name) + " distribution";

	RegisterFunction<Spec>(
	    loader, "sample", FunctionStability::VOLATILE, logical_type_map<sample_t>::Get(),
	    [](DataChunk &args, ExpressionState &state, Vector &result) {
		    RngAdapter rng(RandomEngine::Get(state.GetContext()));
		    ExecuteDistribution<Spec, sample_t, true, std::tuple<>>(
		        args, result, MakeSampler<Spec>, [&rng](auto &sampler) -> sample_t { return sampler(rng); });
	    },
	    {"Generates a random sample from the " + text + " with the specified parameters.", params, "sampling"});

	RegisterFunction<Spec>(
	    loader, "samples", FunctionStability::VOLATILE, LogicalType::LIST(logical_type_map<sample_t>::Get()),
	    [](DataChunk &args, ExpressionState &state, Vector &result) {
		    RngAdapter rng(RandomEngine::Get(state.GetContext()));
		    ExecuteDistribution<Spec, std::vector<sample_t>, true, std::tuple<int64_t>>(
		        args, result, MakeSampler<Spec>, [&rng](auto &sampler, int64_t n) {
			        if (n < 0 || n > static_cast<int64_t>(MAX_SAMPLES_PER_ROW)) {
				        throw InvalidInputException("%s: n must be between 0 and %s, got %s", Spec::name,
				                                    std::to_string(MAX_SAMPLES_PER_ROW), std::to_string(n));
			        }
			        std::vector<sample_t> values(static_cast<size_t>(n));
			        for (idx_t i = 0; i < values.size(); i++) {
				        values[i] = sampler(rng);
			        }
			        return values;
		        });
	    },
	    {"Generates a list of n random samples from the " + text + " with the specified parameters.", params + ", 5",
	     "sampling"},
	    {{"n", LogicalType::BIGINT}});
}

//! For discrete distributions, returns the smallest integer x with P(X <= x) >= p (or P(X > x) <= p for the
//! complement). Boost rounds most discrete quantiles this way already but returns a continuous value for some
//! (e.g. geometric).
template <class Spec, class Dist>
double DiscreteQuantile(const Dist &dist, double p, bool complement, double quantile) {
	if (!Spec::discrete || !std::isfinite(quantile)) {
		return quantile;
	}
	double x = std::ceil(quantile);
	const double lower = boost::math::support(dist).first;
	while (x > lower) {
		const double previous =
		    complement ? boost::math::cdf(boost::math::complement(dist, x - 1)) : boost::math::cdf(dist, x - 1);
		if (complement ? previous > p : previous < p) {
			break;
		}
		x -= 1;
	}
	return x;
}

//! Registers sample, samples, the density, cumulative, quantile and hazard functions, and the distribution
//! properties. Properties that are undefined for the given parameters return NULL.
template <class Spec>
void RegisterStandardDistribution(ExtensionLoader &loader) {
	using quantile_t = typename Spec::quantile_t;
	using X = std::tuple<double>;
	const vector<std::pair<string, LogicalType>> x_param = {{"x", LogicalType::DOUBLE}};
	const vector<std::pair<string, LogicalType>> p_param = {{"p", LogicalType::DOUBLE}};
	const string params = Spec::example_params;
	const string with_x = params + ", " + Spec::example_x;
	const string with_p = params + ", " + Spec::example_p;
	const string text = string(Spec::display_name) + " distribution";
	const string density = Spec::discrete ? "probability mass function (PMF)" : "probability density function (PDF)";

	RegisterSampling<Spec>(loader);

	auto pdf = [](const auto &dist, double x) -> double {
		return boost::math::pdf(dist, x);
	};
	auto log_pdf = [](const auto &dist, double x) -> double {
		return boost::math::logpdf(dist, x);
	};
	RegisterEvaluation<Spec, double, X>(
	    loader, "pdf", pdf, {"Computes the " + density + " of the " + text + " at x.", with_x, "density"}, x_param);
	RegisterEvaluation<Spec, double, X>(
	    loader, "log_pdf", log_pdf,
	    {"Computes the natural logarithm of the " + density + " of the " + text +
	         " at x. Useful for numerical stability when dealing with very small probabilities.",
	     with_x, "density"},
	    x_param);
	if (Spec::discrete) {
		RegisterEvaluation<Spec, double, X>(
		    loader, "pmf", pdf,
		    {"Computes the probability mass function (PMF) of the " + text + " at x, P(X = x).", with_x, "density"},
		    x_param);
		RegisterEvaluation<Spec, double, X>(
		    loader, "log_pmf", log_pdf,
		    {"Computes the natural logarithm of the probability mass function (PMF) of the " + text + " at x.", with_x,
		     "density"},
		    x_param);
	}

	RegisterEvaluation<Spec, double, X>(
	    loader, "cdf", [](const auto &dist, double x) -> double { return boost::math::cdf(dist, x); },
	    {"Computes the cumulative distribution function (CDF) of the " + text + ", P(X <= x).", with_x, "cumulative"},
	    x_param);
	RegisterEvaluation<Spec, double, X>(
	    loader, "cdf_complement",
	    [](const auto &dist, double x) -> double { return boost::math::cdf(boost::math::complement(dist, x)); },
	    {"Computes the complementary cumulative distribution function (survival function) of the " + text +
	         ", P(X > x).",
	     with_x, "cumulative"},
	    x_param);
	RegisterEvaluation<Spec, double, X>(
	    loader, "log_cdf", [](const auto &dist, double x) -> double { return boost::math::logcdf(dist, x); },
	    {"Computes the natural logarithm of the cumulative distribution function (CDF) of the " + text + ".", with_x,
	     "cumulative"},
	    x_param);
	RegisterEvaluation<Spec, double, X>(
	    loader, "log_cdf_complement",
	    [](const auto &dist, double x) -> double { return boost::math::logcdf(boost::math::complement(dist, x)); },
	    {"Computes the natural logarithm of the survival function of the " + text + ", log P(X > x).", with_x,
	     "cumulative"},
	    x_param);

	RegisterEvaluation<Spec, quantile_t, X>(
	    loader, "quantile",
	    [](const auto &dist, double p) -> quantile_t {
		    return static_cast<quantile_t>(DiscreteQuantile<Spec>(dist, p, false, boost::math::quantile(dist, p)));
	    },
	    {"Computes the quantile function (inverse CDF) of the " + text +
	         ". Returns the smallest x such that P(X <= x) >= p.",
	     with_p, "quantile"},
	    p_param);
	RegisterEvaluation<Spec, quantile_t, X>(
	    loader, "quantile_complement",
	    [](const auto &dist, double p) -> quantile_t {
		    return static_cast<quantile_t>(
		        DiscreteQuantile<Spec>(dist, p, true, boost::math::quantile(boost::math::complement(dist, p))));
	    },
	    {"Computes the complementary quantile function of the " + text +
	         ". Returns x such that P(X > x) = p, useful for upper tail quantiles.",
	     with_p, "quantile"},
	    p_param);

	RegisterEvaluation<Spec, double, X>(
	    loader, "hazard", [](const auto &dist, double x) -> double { return boost::math::hazard(dist, x); },
	    {"Computes the hazard function of the " + text + " at x.", with_x, "hazard"}, x_param);
	RegisterEvaluation<Spec, double, X>(
	    loader, "chf", [](const auto &dist, double x) -> double { return boost::math::chf(dist, x); },
	    {"Computes the cumulative hazard function of the " + text + " at x.", with_x, "hazard"}, x_param);

	using Property = std::optional<double>;
	const string undefined = " Returns NULL when it is undefined for the given parameters.";
	RegisterEvaluation<Spec, Property>(
	    loader, "mean", UndefinedAsNull([](const auto &dist) -> double { return boost::math::mean(dist); }),
	    {"Returns the mean of the " + text + "." + undefined, params, "property"});
	RegisterEvaluation<Spec, Property>(
	    loader, "stddev",
	    UndefinedAsNull([](const auto &dist) -> double { return boost::math::standard_deviation(dist); }),
	    {"Returns the standard deviation of the " + text + "." + undefined, params, "property"});
	RegisterEvaluation<Spec, Property>(
	    loader, "variance", UndefinedAsNull([](const auto &dist) -> double { return boost::math::variance(dist); }),
	    {"Returns the variance of the " + text + "." + undefined, params, "property"});
	RegisterEvaluation<Spec, Property>(
	    loader, "mode", UndefinedAsNull([](const auto &dist) -> double { return boost::math::mode(dist); }),
	    {"Returns the mode (most likely value) of the " + text + "." + undefined, params, "property"});
	RegisterEvaluation<Spec, Property>(
	    loader, "median", UndefinedAsNull([](const auto &dist) -> double { return boost::math::median(dist); }),
	    {"Returns the median of the " + text + "." + undefined, params, "property"});
	RegisterEvaluation<Spec, Property>(
	    loader, "skewness", UndefinedAsNull([](const auto &dist) -> double { return boost::math::skewness(dist); }),
	    {"Returns the skewness of the " + text + "." + undefined, params, "property"});
	RegisterEvaluation<Spec, Property>(
	    loader, "kurtosis", UndefinedAsNull([](const auto &dist) -> double { return boost::math::kurtosis(dist); }),
	    {"Returns the kurtosis of the " + text + "." + undefined, params, "property"});
	RegisterEvaluation<Spec, Property>(
	    loader, "kurtosis_excess",
	    UndefinedAsNull([](const auto &dist) -> double { return boost::math::kurtosis_excess(dist); }),
	    {"Returns the excess kurtosis of the " + text + "." + undefined, params, "property"});

	using Interval = std::pair<double, double>;
	RegisterEvaluation<Spec, Interval>(loader, "range",
	                                   [](const auto &dist) -> Interval { return boost::math::range(dist); },
	                                   {"Returns the range of the " + text + " as [min, max].", params, "property"});
	RegisterEvaluation<Spec, Interval>(loader, "support",
	                                   [](const auto &dist) -> Interval { return boost::math::support(dist); },
	                                   {"Returns the support of the " + text + " as [min, max].", params, "property"});
}

} // namespace duckdb
