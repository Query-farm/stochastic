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
struct logical_type_map<int64_t> {
	static LogicalType Get() {
		return LogicalType::BIGINT;
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

} // namespace duckdb