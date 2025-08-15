#pragma once
#include "duckdb.hpp"
#include <boost/random.hpp>
#include "callable_traits.hpp"

namespace duckdb {

template <typename T>
struct logical_type_map {
	static LogicalType Get() {
		using fail_type = T;
		static_assert(AlwaysFalse_v<fail_type>, "Unsupported return type");
		return LogicalType {};
	}
};

template <>
struct logical_type_map<double> {
	static LogicalType Get() {
		return LogicalType::DOUBLE;
	}
};

template <>
struct logical_type_map<std::pair<double, double>> {
	static LogicalType Get() {
		return LogicalType::ARRAY(LogicalType::DOUBLE, 2);
	}
};

// --- Distribution parameter traits ---
template <typename Distribution>
struct distribution_traits; // Primary template left undefined

// Specialization for boost::random::normal_distribution<double>
template <>
struct distribution_traits<boost::math::normal_distribution<double>> {
	using param1_t = double; // mean
	using param2_t = double; // standard deviation
	                         //	using return_t = double; // result type

	static constexpr std::array<const char *, 2> param_names = {"mean", "stddev"};
	static constexpr string prefix = "normal";

	static std::vector<LogicalType> LogicalParamTypes() {
		return {logical_type_map<param1_t>::Get(), logical_type_map<param2_t>::Get()};
	}
};

template <>
struct distribution_traits<boost::random::normal_distribution<double>> {
	using param1_t = double; // mean
	using param2_t = double; // standard deviation
	                         //	using return_t = double; // result type

	static constexpr std::array<const char *, 2> param_names = {"mean", "stddev"};

	static constexpr string prefix = "normal";

	static std::vector<LogicalType> LogicalParamTypes() {
		return {logical_type_map<param1_t>::Get(), logical_type_map<param2_t>::Get()};
	}
};

} // namespace duckdb