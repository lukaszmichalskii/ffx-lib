#pragma once

/// @file include/core/ring_buffer.hpp
/// @brief Lock-free SPSC / MPSC circular ring buffer using C++20
///        std::atomic::wait / notify_one for kernel-free blocking.
///
/// ### Design: Vyukov sequence-counter protocol
///
/// Every slot carries an atomic `sequence` counter whose lifecycle is:
///
///  Phase          | sequence value
///  -------------- | -----------------------------------------------
///  Initial / free | natural slot index `i`  (i.e. pos - 0 wrap-arounds)
///  Written        | pos + 1  (release-stored by the producer)
///  Recycled       | pos + Capacity  (release-stored by the consumer)
///
/// This encoding lets both producers (checking "is this slot free?") and
/// the consumer (checking "is data ready?") operate with a single acquire
/// load — no separate head/tail cross-reads, no false-sharing between
/// producer cursors and consumer cursor.
///
/// ### Blocking (zero kernel overhead)
///
/// A monotone `wakeup_signal_` counter is incremented (release) and
/// notify_one() is called by each successful emplace().  The consumer
/// snapshots the counter BEFORE performing its final try_pop() so that an
/// emplace() that fires between try_pop() and wait() is never missed.
///
/// ### Thread-safety contract
///
///  Method               | Caller constraint
///  -------------------- | -------------------------------------------------
///  emplace()            | ONE thread when Policy == SingleProducer; any
///                       | number of threads when Policy == MultiProducer.
///  pop_wait() / try_pop | EXACTLY ONE consumer thread.
///  unblock_waiters()    | Any thread.
///
/// ### Memory-ordering
///
///  emplace:   slot.value write  →  slot.sequence.store(pos+1, release)
///  try_pop:   slot.sequence.load(acquire)  →  slot.value read
///  unblock:   stopped_.store(true, release)  →  consumer acquire load

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <stop_token>
#include <utility>

namespace ffx::core {

  // -------------------------------------------------------------------------
  // Cache-line size
  // -------------------------------------------------------------------------
  namespace detail {
#if defined(__cpp_lib_hardware_interference_size)
    inline constexpr std::size_t kCacheLine =
        std::hardware_destructive_interference_size;
#else
    inline constexpr std::size_t kCacheLine = 64u;
#endif
  }  // namespace detail

  // -------------------------------------------------------------------------
  // ProducerPolicy
  // -------------------------------------------------------------------------

  /// Selects the producer concurrency model for LockFreeRingBuffer.
  enum class ProducerPolicy : std::uint8_t {
    SingleProducer,  ///< SPSC — exactly one producer thread.  No CAS.
    MultiProducer,   ///< MPSC — any number of producer threads.  CAS on tail_.
  };

  // -------------------------------------------------------------------------
  // LockFreeRingBuffer
  // -------------------------------------------------------------------------

  /// @brief Allocation-free, lock-free circular ring buffer with a
  ///        blocking consumer and zero OS-kernel overhead when idle.
  ///
  /// @tparam T        Element type (must be move-constructible).
  /// @tparam Capacity Number of slots; MUST be a power of two.
  /// @tparam Policy   ProducerPolicy::SingleProducer (default, SPSC) or
  ///                  ProducerPolicy::MultiProducer (MPSC).
  template <typename T,
            std::size_t   Capacity = 2048,
            ProducerPolicy Policy  = ProducerPolicy::SingleProducer>
  class alignas(detail::kCacheLine) LockFreeRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");
    static constexpr std::size_t kMask = Capacity - 1;

    // -----------------------------------------------------------------------
    // Slot: each padded to its own cache line to prevent false-sharing
    // between adjacent elements when sizeof(T) is small.
    // -----------------------------------------------------------------------
    struct alignas(detail::kCacheLine) Slot {
      std::atomic<std::size_t> sequence{0};
      T                        value{};
    };

   public:
    /// Initialises slot sequence numbers to their natural indices so that
    /// `slot[i].sequence == i` ("slot i is free for the first write").
    LockFreeRingBuffer() noexcept {
      for (std::size_t i = 0; i < Capacity; ++i)
        buffer_[i].sequence.store(i, std::memory_order_relaxed);
    }

    // Non-copyable, non-movable: atomic members cannot be relocated.
    LockFreeRingBuffer(const LockFreeRingBuffer&)             = delete;
    LockFreeRingBuffer& operator=(const LockFreeRingBuffer&)  = delete;
    LockFreeRingBuffer(LockFreeRingBuffer&&)                  = delete;
    LockFreeRingBuffer& operator=(LockFreeRingBuffer&&)       = delete;

    // -----------------------------------------------------------------------
    // Producer API
    // -----------------------------------------------------------------------

    /// Enqueue an element constructed in-place from @p args.
    /// @return true on success; false when the buffer is full (never blocks).
    template <typename... Args>
    bool emplace(Args&&... args)
        noexcept(std::is_nothrow_constructible_v<T, Args...>) {
      if constexpr (Policy == ProducerPolicy::SingleProducer)
        return emplace_spsc(std::forward<Args>(args)...);
      else
        return emplace_mpsc(std::forward<Args>(args)...);
    }

    // -----------------------------------------------------------------------
    // Consumer API  (single consumer thread only)
    // -----------------------------------------------------------------------

    /// Block until an item is available or a stop is requested.
    ///
    /// All enqueued items are drained before a stop request is honoured so
    /// that in-flight data is never silently discarded.
    ///
    /// Uses std::atomic::wait so the thread yields its CPU timeslice when the
    /// queue is empty — CPU usage drops to ~0 % while waiting.
    ///
    /// @param stop  C++20 stop_token from the owning std::jthread.
    /// @return An item, or std::nullopt when empty AND stop was requested.
    [[nodiscard]] std::optional<T> pop_wait(
        const std::stop_token& stop) noexcept {
      while (true) {
        // Fast path: item already available.
        if (auto item = try_pop()) return item;

        // Check stop only after confirming the queue is empty.
        if (stop.stop_requested() ||
            stopped_.load(std::memory_order_acquire))
          return std::nullopt;

        // Snapshot the wakeup counter BEFORE the final try_pop so that an
        // emplace() racing between try_pop() and wait() is never missed:
        // the counter will already have changed, making wait() return
        // immediately instead of sleeping.
        const auto sig = wakeup_signal_.load(std::memory_order_relaxed);

        // One last opportunity to dequeue before sleeping.
        if (auto item = try_pop()) return item;

        // Sleep until a producer (or unblock_waiters()) nudges us.
        wakeup_signal_.wait(sig, std::memory_order_relaxed);
      }
    }

    /// Non-blocking dequeue.
    /// @return An item, or std::nullopt when the queue is empty.
    [[nodiscard]] std::optional<T> try_pop() noexcept {
      const auto   pos  = head_.load(std::memory_order_relaxed);
      auto&        slot = buffer_[pos & kMask];
      const auto   seq  = slot.sequence.load(std::memory_order_acquire);

      // seq == pos + 1  →  the producer has written this slot.
      const auto diff =
          static_cast<std::intptr_t>(seq) -
          static_cast<std::intptr_t>(pos + 1);
      if (diff != 0) return std::nullopt;

      T item = std::move(slot.value);
      head_.store(pos + 1, std::memory_order_relaxed);

      // Mark slot as free for the next producer writing at pos + Capacity.
      slot.sequence.store(pos + Capacity, std::memory_order_release);
      return item;
    }

    /// Wake all threads sleeping in pop_wait() and signal them to drain
    /// remaining items then return std::nullopt.
    ///
    /// Thread-safe; may be called from any thread.
    void unblock_waiters() noexcept {
      // release pairs with the acquire in pop_wait.
      stopped_.store(true, std::memory_order_release);
      wakeup_signal_.fetch_add(1, std::memory_order_release);
      wakeup_signal_.notify_all();
    }

    /// Snapshot emptiness check (may race with concurrent push/pop).
    [[nodiscard]] bool empty() const noexcept {
      const auto pos = head_.load(std::memory_order_relaxed);
      return buffer_[pos & kMask].sequence.load(std::memory_order_acquire) !=
             pos + 1;
    }

    /// Compile-time capacity.
    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
      return Capacity;
    }

   private:
    // -----------------------------------------------------------------------
    // SPSC emplace — single producer, no CAS, minimal overhead.
    // -----------------------------------------------------------------------
    template <typename... Args>
    bool emplace_spsc(Args&&... args)
        noexcept(std::is_nothrow_constructible_v<T, Args...>) {
      const auto pos  = tail_.load(std::memory_order_relaxed);
      auto&      slot = buffer_[pos & kMask];
      const auto seq  = slot.sequence.load(std::memory_order_acquire);

      // seq == pos  →  slot is free (initial or recycled by consumer).
      // diff < 0    →  producer has lapped the consumer; buffer is full.
      const auto diff =
          static_cast<std::intptr_t>(seq) -
          static_cast<std::intptr_t>(pos);
      if (diff != 0) return false;

      slot.value = T(std::forward<Args>(args)...);
      // release: slot.value is visible to consumer before seq becomes pos+1.
      slot.sequence.store(pos + 1, std::memory_order_release);
      tail_.store(pos + 1, std::memory_order_relaxed);

      // Notify the sleeping consumer.
      wakeup_signal_.fetch_add(1, std::memory_order_release);
      wakeup_signal_.notify_one();
      return true;
    }

    // -----------------------------------------------------------------------
    // MPSC emplace — CAS loop to atomically claim a slot index.
    //
    // Exception safety: if T construction throws after the CAS succeeds, the
    // slot's sequence is left at `pos` (free state) because we have not yet
    // written pos+1.  The buffer remains self-consistent.  Users are advised
    // to use nothrow-constructible types in MPSC mode.
    // -----------------------------------------------------------------------
    template <typename... Args>
    bool emplace_mpsc(Args&&... args)
        noexcept(std::is_nothrow_constructible_v<T, Args...>) {
      std::size_t pos = tail_.load(std::memory_order_relaxed);
      while (true) {
        auto&      slot = buffer_[pos & kMask];
        const auto seq  = slot.sequence.load(std::memory_order_acquire);
        const auto diff =
            static_cast<std::intptr_t>(seq) -
            static_cast<std::intptr_t>(pos);

        if (diff == 0) {
          // Slot appears free — compete to claim it with CAS.
          if (tail_.compare_exchange_weak(
                  pos, pos + 1, std::memory_order_relaxed)) {
            // We won: sole owner of slot at index pos.
            slot.value = T(std::forward<Args>(args)...);
            // release: makes slot.value visible before seq = pos+1.
            slot.sequence.store(pos + 1, std::memory_order_release);

            wakeup_signal_.fetch_add(1, std::memory_order_release);
            wakeup_signal_.notify_one();
            return true;
          }
          // CAS failed — another producer advanced tail_; `pos` now holds
          // the updated value from the failed CAS, retry immediately.
        } else if (diff < 0) {
          // seq < pos: consumer has not yet freed enough slots.
          return false;  // buffer full
        } else {
          // diff > 0: another producer already claimed this slot; reload.
          pos = tail_.load(std::memory_order_relaxed);
        }
      }
    }

    // -----------------------------------------------------------------------
    // Members — each on its own cache line to prevent false-sharing between
    // the consumer cursor (head_), producer cursor (tail_), wakeup signal,
    // and shutdown latch.
    // -----------------------------------------------------------------------
    alignas(detail::kCacheLine) std::atomic<std::size_t> head_{0};
    alignas(detail::kCacheLine) std::atomic<std::size_t> tail_{0};
    alignas(detail::kCacheLine) std::atomic<std::size_t> wakeup_signal_{0};
    alignas(detail::kCacheLine) std::atomic<bool>        stopped_{false};

    // Slot array — intentionally not separately aligned; each Slot is already
    // cache-line-aligned via its struct alignment.
    std::array<Slot, Capacity> buffer_{};
  };

}  // namespace ffx::core
