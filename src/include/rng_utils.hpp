#pragma once
#include <random>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <vector>
#include <boost/random/mersenne_twister.hpp>

namespace duckdb {
// Global seed for all RNG streams
constexpr unsigned int GLOBAL_SEED = 12345;

// Get unique fixed index for each thread
unsigned int get_thread_index();

// Thread-local RNG, seeded uniquely per thread
static thread_local boost::random::mt19937 rng = [] {
	unsigned int tidx = get_thread_index();
	std::seed_seq seq {GLOBAL_SEED, tidx};
	std::vector<uint32_t> seed_data(1);
	seq.generate(seed_data.begin(), seed_data.end());

	boost::random::mt19937 local_rng;
	local_rng.seed(seed_data[0]);
	return local_rng;
}();

}