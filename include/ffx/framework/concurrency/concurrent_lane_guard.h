#pragma once

#include <alpaka/alpaka.hpp>
#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>

#include "ffx/core/detail/concepts.h"

namespace ffx::framework::concurrency {

  template <concepts::queue TQueue, std::size_t TSlidingWindowSize = 8>
  class ConcurrentLaneGuard {
    static_assert((TSlidingWindowSize & (TSlidingWindowSize - 1)) == 0, "TSlidingWindowSize must be a power of 2!");
    static constexpr std::size_t kIndexMask = TSlidingWindowSize - 1;

  public:
    using Device = alpaka::Dev<TQueue>;
    using Event = alpaka::Event<TQueue>;

    explicit ConcurrentLaneGuard(const Device& device) {
      for (auto index = 0zu; index < TSlidingWindowSize; ++index) {
        events_[index] = std::make_unique<Event>(device);
        slot_ready_[index].store(false, std::memory_order_relaxed);
      }
    }

    void reclaim_completed() noexcept {
      std::size_t tail = tail_.load(std::memory_order_relaxed);
      const std::size_t head = head_.load(std::memory_order_acquire);

      while (tail < head) {
        const auto slot = tail & kIndexMask;

        // not enqueued on device, cannot test completion
        if (!slot_ready_[slot].load(std::memory_order_acquire)) {
          break;
        }

        if (alpaka::isComplete(*events_[slot])) {
          slot_ready_[slot].store(false, std::memory_order_release);

          ++tail;
          tail_.store(tail, std::memory_order_release);

          in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        } else {
          break;  // in-order queue: older operations are still running
        }
      }
    }

    [[nodiscard]] std::size_t acquire() noexcept {
      while (true) {
        reclaim_completed();

        auto current = in_flight_.load(std::memory_order_acquire);
        if (current < TSlidingWindowSize) {
          if (in_flight_.compare_exchange_weak(
                  current, current + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            const auto ticket = head_.fetch_add(1, std::memory_order_acq_rel);
            return ticket & kIndexMask;
          }
          continue;
        }

        std::this_thread::yield();  // window full: spin-yield and poll device
      }
    }

    void mark_enqueued(const std::size_t slot) noexcept { slot_ready_[slot].store(true, std::memory_order_release); }

    [[nodiscard]] Event& get_event(const std::size_t slot) noexcept { return *events_[slot]; }

    /// reset all state. Called ONLY after alpaka::wait(queue_) when device is guaranteed idle.
    void reset() noexcept {
      for (auto& slot : slot_ready_) {
        slot.store(false, std::memory_order_relaxed);
      }
      head_.store(0, std::memory_order_relaxed);
      tail_.store(0, std::memory_order_relaxed);
      in_flight_.store(0, std::memory_order_release);
    }

    [[nodiscard]] std::size_t in_flight() const noexcept { return in_flight_.load(std::memory_order_acquire); }

  private:
    std::array<std::unique_ptr<Event>, TSlidingWindowSize> events_;
    std::array<std::atomic<bool>, TSlidingWindowSize> slot_ready_;

    alignas(64) std::atomic<std::size_t> in_flight_{0};
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
  };

}  // namespace ffx::framework::concurrency