#pragma once

#include "concurrent_lane_guard.h"
#include "thread_affinity.h"
#include "ffx/framework/concurrency/ring_buffer.h"
#include "ffx/core/detail/concepts.h"
#include "ffx/framework/concurrency/concurrent_lane_memory.h"
#include "ffx/framework/fw_core/scheduler.h"

namespace ffx::framework::concurrency {

  template <concepts::queue TQueue, std::size_t N = 8>
  class ConcurrentLane {
  public:
    using Device = alpaka::Dev<TQueue>;
    using Task = std::function<void(TQueue&, ConcurrentLaneMemory&, Scheduler<TQueue>&)>;

    ConcurrentLane(const Device& device, const std::size_t lane_id)
        : queue_(device), lane_id_(lane_id), lane_guard_(device), worker_([this](std::stop_token token) {
            set_current_thread_affinity(lane_id_);
            worker_loop(token);
          }) {
      set_thread_affinity(worker_, lane_id_);
    }

    ~ConcurrentLane() { shutdown(); }

    ConcurrentLane(const ConcurrentLane&) = delete;
    ConcurrentLane& operator=(const ConcurrentLane&) = delete;
    ConcurrentLane(ConcurrentLane&&) = delete;
    ConcurrentLane& operator=(ConcurrentLane&&) = delete;

    Scheduler<TQueue>& scheduler() noexcept { return scheduler_; }
    TQueue& queue() noexcept { return queue_; }
    ConcurrentLaneMemory& shared_memory() noexcept { return shared_memory_; }
    [[nodiscard]] std::size_t id() const noexcept { return lane_id_; }

    template <typename Fn>
    void submit(Fn&& fn) {
      const std::size_t slot = lane_guard_.acquire();
      active_tasks_.fetch_add(1, std::memory_order_relaxed);

      auto task = [this, slot, func = std::forward<Fn>(fn)](
                      TQueue& queue, ConcurrentLaneMemory& mem, Scheduler<TQueue>& scheduler) mutable {
        func(queue, mem, scheduler);

        auto& event = lane_guard_.get_event(slot);
        alpaka::enqueue(queue, event);
        lane_guard_.mark_enqueued(slot);
      };

      while (!task_queue_.emplace(std::move(task))) {
        lane_guard_.reclaim_completed();
        std::this_thread::yield();
      }
    }

    void wait_tasks() noexcept {
      while (active_tasks_.load(std::memory_order_acquire) > 0) {
        lane_guard_.reclaim_completed();
        std::this_thread::yield();
      }
    }

    void sync() {
      alpaka::wait(queue_);
      lane_guard_.reset();
    }

    void wait() {
      wait_tasks();
      sync();
    }

  private:
    void shutdown() {
      worker_.request_stop();
      task_queue_.unblock_waiters();
      if (worker_.joinable()) {
        worker_.join();
      }
      alpaka::wait(queue_);
      lane_guard_.reset();
    }

    void worker_loop(const std::stop_token& st) {
      while (auto task = task_queue_.pop_wait(st)) {
        (*task)(queue_, shared_memory_, scheduler_);
        active_tasks_.fetch_sub(1, std::memory_order_release);
        lane_guard_.reclaim_completed();
      }
    }

    TQueue queue_;
    const std::size_t lane_id_;
    ConcurrentLaneMemory shared_memory_;
    Scheduler<TQueue> scheduler_;
    ConcurrentLaneGuard<TQueue, N> lane_guard_;
    ring_buffer<Task, 2048> task_queue_;
    alignas(64) std::atomic<std::size_t> active_tasks_{0};
    std::jthread worker_;
  };

}  // namespace ffx::framework::concurrency