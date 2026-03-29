#pragma once
#include "duckdb/common/random_engine.hpp"
#include "duckdb/common/mutex.hpp"
#include <cstdint>
#include <limits>
#include <random>

namespace duckdb {

// Seeds a local std::mt19937_64 from DuckDB's per-client RandomEngine.
// Acquires RandomEngine::lock to avoid data races (RandomEngine is shared
// across threads), then runs entirely on the local engine to avoid mutex
// contention in hot sampling loops.
struct RngAdapter {
	std::mt19937_64 local_rng;
	using result_type = uint64_t;

	explicit RngAdapter(RandomEngine &engine) : local_rng([&engine]() {
		lock_guard<mutex> guard(engine.lock);
		return engine.NextRandomInteger64();
	}()) {
	}

	static constexpr result_type min() {
		return std::mt19937_64::min();
	}
	static constexpr result_type max() {
		return std::mt19937_64::max();
	}
	result_type operator()() {
		return local_rng();
	}
};

} // namespace duckdb
