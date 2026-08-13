#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <stop_token>
#include <utility>

namespace ffx::framework::concurrency {

  template <typename T, std::size_t Capacity = 2048>
  class alignas(64) ring_buffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2!");

  public:
    ring_buffer() = default;

    ring_buffer(const ring_buffer&) = delete;
    ring_buffer& operator=(const ring_buffer&) = delete;
    ring_buffer(ring_buffer&&) = delete;
    ring_buffer& operator=(ring_buffer&&) = delete;

    template <typename... Args>
    bool emplace(Args&&... args) noexcept {
      const auto current_tail = tail_.load(std::memory_order_relaxed);
      if (current_tail - head_.load(std::memory_order_acquire) == Capacity)
        return false;  // full

      buffer_[current_tail & (Capacity - 1)] = T(std::forward<Args>(args)...);
      tail_.store(current_tail + 1, std::memory_order_release);

      wakeup_signal_.fetch_add(1, std::memory_order_release);
      wakeup_signal_.notify_one();
      return true;
    }

    std::optional<T> pop_wait(const std::stop_token& stop_tok) noexcept {
      while (true) {
        const auto current_head = head_.load(std::memory_order_relaxed);
        const auto current_tail = tail_.load(std::memory_order_acquire);

        // drain items
        if (current_head != current_tail) {
          T item = std::move(buffer_[current_head & (Capacity - 1)]);
          head_.store(current_head + 1, std::memory_order_release);
          return item;
        }

        // check stop if queue is empty
        if (stop_tok.stop_requested() || stopped_.load(std::memory_order_acquire)) {
          return std::nullopt;
        }

        // sleep until the producer signals a new item (or stop).
        const auto current_signal = wakeup_signal_.load(std::memory_order_relaxed);
        wakeup_signal_.wait(current_signal, std::memory_order_relaxed);
      }
    }

    /// signal all threads sleeping in pop_wait() to drain remaining items
    /// and then return std::nullopt.
    void unblock_waiters() noexcept {
      stopped_.store(true, std::memory_order_release);  // pairs with acquire in pop_wait
      wakeup_signal_.fetch_add(1, std::memory_order_release);
      wakeup_signal_.notify_all();
    }

    [[nodiscard]] bool empty() const noexcept {
      return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_acquire);
    }

  private:
    // each cursor on its own cache line to prevent false-sharing.
    alignas(64) std::atomic<std::size_t> head_{0};           // consumer cursor
    alignas(64) std::atomic<std::size_t> tail_{0};           // producer cursor
    alignas(64) std::atomic<std::size_t> wakeup_signal_{0};  // monotone wake counter
    alignas(64) std::atomic<bool> stopped_{false};           // shutdown latch
    alignas(64) std::array<T, Capacity> buffer_{};
  };

}  // namespace ffx::framework::concurrency
