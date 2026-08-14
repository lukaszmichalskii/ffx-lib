#pragma once

#include <alpaka/alpaka.hpp>
#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>

#include "ffx/core/detail/concepts.h"

namespace ffx::framework::concurrency {

  template <concepts::queue TQueue, std::size_t TLaneCapacity = 8>
  class ConcurrentLaneGuard {
    static_assert(TLaneCapacity >= 2, "TLaneCapacity must be at least 2 for async double-buffering!");
    static_assert((TLaneCapacity & (TLaneCapacity - 1)) == 0, "TLaneCapacity must be a power of 2!");
    static constexpr std::size_t kIndexMask = TLaneCapacity - 1;

  public:
    using Device = alpaka::Dev<TQueue>;
    using Event = alpaka::Event<TQueue>;

    explicit ConcurrentLaneGuard(const Device& device) {
      for (auto index = 0zu; index < TLaneCapacity; ++index) {
        events_[index] = std::make_unique<Event>(device);
        slot_ready_[index].store(false, std::memory_order_relaxed);
      }
    }

    void reclaim_completed() noexcept {
      while (true) {
        auto tail = tail_.load(std::memory_order_acquire);
        const auto head = head_.load(std::memory_order_acquire);

        if (tail >= head) {
          break;  // All issued tickets have been reclaimed
        }

        const auto slot = tail & kIndexMask;

        // If the task hasn't been submitted to the GPU stream yet, stop
        if (!slot_ready_[slot].load(std::memory_order_acquire)) {
          break;
        }

        if (alpaka::isComplete(*events_[slot])) {
          // Atomically advance tail to ensure ONLY ONE thread retires this slot
          if (tail_.compare_exchange_weak(tail, tail + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            slot_ready_[slot].store(false, std::memory_order_release);
          }
        } else {
          // In-order hardware stream: older operations are still running
          break;
        }
      }
    }

    [[nodiscard]] std::size_t acquire() noexcept {
      while (true) {
        reclaim_completed();

        const auto head = head_.load(std::memory_order_acquire);
        const auto tail = tail_.load(std::memory_order_acquire);

        // In-flight count is purely derived from (head - tail)
        if (head - tail < TLaneCapacity) {
          if (head_.compare_exchange_weak(
                  const_cast<std::size_t&>(head), head + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            return head & kIndexMask;
          }
          continue;
        }

        // Window full: yield to allow worker/GPU to complete tasks
        std::this_thread::yield();
      }
    }

    void mark_enqueued(const std::size_t slot) noexcept { slot_ready_[slot].store(true, std::memory_order_release); }

    [[nodiscard]] Event& get_event(const std::size_t slot) noexcept { return *events_[slot]; }

    void reset() noexcept {
      for (auto& slot : slot_ready_) {
        slot.store(false, std::memory_order_relaxed);
      }
      head_.store(0, std::memory_order_release);
      tail_.store(0, std::memory_order_release);
    }

    [[nodiscard]] std::size_t in_flight() const noexcept {
      const auto head = head_.load(std::memory_order_acquire);
      const auto tail = tail_.load(std::memory_order_acquire);
      return (head >= tail) ? (head - tail) : 0;
    }

  private:
    std::array<std::unique_ptr<Event>, TLaneCapacity> events_;
    std::array<std::atomic<bool>, TLaneCapacity> slot_ready_;

    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
  };

}  // namespace ffx::framework::concurrency