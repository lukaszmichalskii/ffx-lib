#pragma once

#include <thread>
#include <pthread.h>

namespace ffx::framework::concurrency {

  inline bool set_thread_affinity(std::jthread& thread, const std::size_t core_id) {
    const std::size_t core = core_id % std::thread::hardware_concurrency();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);

    return pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset) == 0;
  }

  inline bool set_current_thread_affinity(const std::size_t core_id) {
    const std::size_t core = core_id % std::thread::hardware_concurrency();

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);

    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
  }

}  // namespace ffx::framework::concurrency