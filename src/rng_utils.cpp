#include "rng_utils.hpp"

namespace duckdb {
static std::unordered_map<std::thread::id, unsigned int> thread_id_map;
static std::mutex thread_id_map_mutex;
static unsigned int next_thread_index = 0;

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

}