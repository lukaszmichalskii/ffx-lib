#pragma once

#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include <alpaka/alpaka.hpp>
#include <boost/core/demangle.hpp>

#include "ffx/core/alpaka/platform.h"

namespace ffx {

  namespace mem::detail {

    // bin growth factor (bin_growth in cub::CachingDeviceAllocator)
    constexpr unsigned int kBinGrowth = 2;
    // smallest bin, corresponds to kBinGrowth^kMinBin bytes (min_bin in cub::CachingDeviceAllocator
    constexpr unsigned int kMinBin = 8;  // 256 bytes
    // largest bin, corresponds to kBinGrowth^kMaxBin bytes (max_bin in cub::CachingDeviceAllocator).
    // Note that unlike in cub, allocations larger than binGrowth^kMaxBin are set to fail.
    constexpr unsigned int kMaxBin = 30;  // 1 GB
    // total storage for the allocator; 0 means no limit.
    constexpr std::size_t kMaxCachedBytes = 0;
    // fraction of total device memory taken for the allocator; 0 means no limit.
    constexpr double kMaxCachedFraction = 0.8;
    // if both kMaxCachedBytes and kMaxCachedFraction are non-zero, the smallest resulting value is used.

    constexpr unsigned int power(unsigned int base, unsigned int exponent) {
      unsigned int power = 1;
      while (exponent > 0) {
        if (exponent & 1)
          power = power * base;

        base = base * base;
        exponent = exponent >> 1;
      }
      return power;
    }

    // format a memory size in B/kB/MB/GB
    inline std::string as_bytes(const size_t value) {
      if (value == std::numeric_limits<size_t>::max())
        return "unlimited";
      if (value >= (1 << 30) and value % (1 << 30) == 0)
        return std::to_string(value >> 30) + " GB";
      if (value >= (1 << 20) and value % (1 << 20) == 0)
        return std::to_string(value >> 20) + " MB";
      if (value >= (1 << 10) and value % (1 << 10) == 0)
        return std::to_string(value >> 10) + " kB";
      return std::to_string(value) + "  B";
    }

  }  // namespace mem::detail

  /*
   * The "memory device" identifies the memory space, i.e. the device where the memory is allocated.
   * A caching allocator object is associated to a single memory `Device`, set at construction time, and unchanged for
   * the lifetime of the allocator.
   *
   * Each allocation is associated to an event on a queue, that identifies the "synchronisation device" according to
   * which the synchronisation occurs.
   * The `Event` type depends only on the synchronisation `Device` type.
   * The `Queue` type depends on the synchronisation `Device` type and the queue properties, either `Sync` or `Async`.
   *
   * **Note**: how to handle different queue and event types in a single allocator ?  store and access type-punned
   * queues and events ?  or template the internal structures on them, but with a common base class ?
   * alpaka does rely on the compile-time type for dispatch.
   *
   * Common use case #1: accelerator's memory allocations
   *   - the "memory device" is the accelerator device (e.g. a GPU);
   *   - the "synchronisation device" is the same accelerator device;
   *   - the `Queue` type is usually always the same (either `Sync` or `Async`).
   *
   * Common use case #2: pinned host memory allocations
   *    - the "memory device" is the host device (e.g. system memory);
   *    - the "synchronisation device" is the accelerator device (e.g. a GPU) whose work queue will access the host;
   *      memory (direct memory access from the accelerator, or scheduling `alpaka::memcpy`/`alpaka::memset`), and can
   *      be different for each allocation;
   *    - the synchronisation `Device` _type_ could potentially be different, but memory pinning is currently tied to
   *      the accelerator's platform (CUDA, HIP, etc.), so the device type needs to be fixed to benefit from caching;
   *    - the `Queue` type can be either `Sync` _or_ `Async` on any allocation.
   */
  template <typename TDevice, typename TQueue>
  class CachingAllocator {
  public:
    using Device = TDevice;              // the "memory device", where the memory will be allocated
    using Queue = TQueue;                // the queue used to submit the memory operations
    using Event = alpaka::Event<Queue>;  // the events used to synchronise the operations
    using Buffer = alpaka::Buf<Device, std::byte, alpaka::DimInt<1u>, size_t>;

    // The "memory device" type can either be the same as the "synchronisation device" type, or be the host CPU.
    static_assert(std::is_same_v<Device, alpaka::Dev<Queue>> or std::is_same_v<Device, alpaka::DevCpu>,
                  "The \"memory device\" type can either be the same as the "
                  "\"synchronisation device\" "
                  "type, or be the "
                  "host CPU.");

    struct CachedBytes {
      size_t free = 0;       // total bytes freed and cached on this device
      size_t live = 0;       // total bytes currently in use oin this device
      size_t requested = 0;  // total bytes requested and currently in use on this device
    };

    explicit CachingAllocator(
        Device const& device,
        unsigned int bin_growth,     // bin growth factor;
        unsigned int min_bin,        // smallest bin, corresponds to bin_growth^min_bin bytes;
                                     // smaller allocations are rounded to this value;
        unsigned int max_bin,        // largest bin, corresponds to bin_growth^kMaxBin bytes;
                                     // larger allocations will fail;
        size_t max_cached_bytes,     // total storage for the allocator (0 means no limit);
        double max_cached_fraction,  // fraction of total device memory taken for the allocator (0 means no limit);
        // if both kMaxCachedBytes and kMaxCachedFraction are non-zero,
        // the smallest resulting value is used.
        bool reuse_same_queue_allocations,  // reuse non-ready allocations if they are in the same queue as the new one;
        // this is safe only if all memory operations are scheduled in the same queue
        bool debug)
        : device_(device),
          bin_growth_(bin_growth),
          min_bin_(min_bin),
          max_bin_(max_bin),
          min_bin_bytes_(mem::detail::power(bin_growth, min_bin)),
          max_bin_bytes_(mem::detail::power(bin_growth, max_bin)),
          max_cached_bytes_(cache_size(max_cached_bytes, max_cached_fraction)),
          reuse_same_queue_allocations_(reuse_same_queue_allocations),
          debug_(debug) {
      if (debug_) {
        std::ostringstream out;
        out << "CachingAllocator settings\n"
            << "  bin growth " << bin_growth_ << "\n"
            << "  min bin    " << min_bin_ << "\n"
            << "  max bin    " << max_bin_ << "\n"
            << "  resulting bins:\n";
        for (auto bin = min_bin_; bin <= max_bin_; ++bin) {
          auto binSize = mem::detail::power(bin_growth, bin);
          out << "    " << std::right << std::setw(12) << mem::detail::as_bytes(binSize) << '\n';
        }
        out << "  maximum amount of cached memory: " << mem::detail::as_bytes(max_cached_bytes_);
        std::cout << out.str() << std::endl;
      }
    }

    ~CachingAllocator() {
      {
        // this should never be called while some memory blocks are still live
        std::scoped_lock lock(mutex_);
        assert(live_blocks_.empty());
        assert(cached_bytes_.live == 0);
      }

      free_all_cached();
    }

    // return a copy_if of the cache allocation status, for monitoring purposes
    CachedBytes cache_status() const {
      std::scoped_lock lock(mutex_);
      return cached_bytes_;
    }

    // Allocate given number of bytes on the current device associated to given queue
    void* allocate(size_t bytes, Queue& queue) {
      // create a block descriptor for the requested allocation
      BlockDescriptor block;
      block.queue = std::move(queue);
      block.requested = bytes;
      std::tie(block.bin, block.bytes) = findBin(bytes);

      // try to re-use a cached block, or allocate a new buffer
      if (not try_reuse_cached_block(block)) {
        allocate_new_block(block);
      }

      return block.buffer->data();
    }

    // frees an allocation
    void free(void* ptr) {
      std::scoped_lock lock(mutex_);

      auto iblock = live_blocks_.find(ptr);
      if (iblock == live_blocks_.end()) {
        std::stringstream ss;
        ss << "Trying to free a non-live block at " << ptr;
        throw std::runtime_error(ss.str());
      }
      // remove the block from the list of live blocks
      BlockDescriptor block = std::move(iblock->second);
      live_blocks_.erase(iblock);
      cached_bytes_.live -= block.bytes;
      cached_bytes_.requested -= block.requested;

      bool recache = (cached_bytes_.free + block.bytes <= max_cached_bytes_);
      if (recache) {
        alpaka::enqueue(*(block.queue), *(block.event));
        cached_bytes_.free += block.bytes;
        // after the call to insert(), cached_blocks_ shares ownership of the buffer
        // TODO use std::move ?
        cached_blocks_.insert(std::make_pair(block.bin, block));

        if (debug_) {
          std::ostringstream out;
          out << "\t" << device_type_ << " " << alpaka::getName(device_) << " returned " << block.bytes << " bytes at "
              << ptr << " from associated queue " << block.queue->m_spQueueImpl.get() << " , event "
              << block.event->m_spEventImpl.get() << " .\n\t\t " << cached_blocks_.size()
              << " available blocks cached (" << cached_bytes_.free << " bytes), " << live_blocks_.size()
              << " live blocks (" << cached_bytes_.live << " bytes) outstanding." << std::endl;
          std::cout << out.str() << std::endl;
        }
      } else {
        // if the buffer is not recached, it is automatically freed when block goes out of scope
        if (debug_) {
          std::ostringstream out;
          out << "\t" << device_type_ << " " << alpaka::getName(device_) << " freed " << block.bytes << " bytes at "
              << ptr << " from associated queue " << block.queue->m_spQueueImpl.get() << ", event "
              << block.event->m_spEventImpl.get() << " .\n\t\t " << cached_blocks_.size()
              << " available blocks cached (" << cached_bytes_.free << " bytes), " << live_blocks_.size()
              << " live blocks (" << cached_bytes_.live << " bytes) outstanding." << std::endl;
          std::cout << out.str() << std::endl;
        }
      }
    }

  private:
    struct BlockDescriptor {
      std::optional<Buffer> buffer;
      std::optional<Queue> queue;
      std::optional<Event> event;
      size_t bytes = 0;
      size_t requested = 0;  // for monitoring only
      unsigned int bin = 0;

      // the "synchronisation device" for this block
      auto device() { return alpaka::getDev(*queue); }
    };

    // return the maximum amount of memory that should be cached on this device
    size_t cache_size(size_t max_cached_bytes, double max_cached_fraction) const {
      // note that getMemBytes() returns 0 if the platform does not support querying the device memory
      size_t total_memory = alpaka::getMemBytes(device_);
      size_t memory_fraction = static_cast<size_t>(max_cached_fraction * total_memory);
      size_t size = std::numeric_limits<size_t>::max();
      if (max_cached_bytes > 0 and max_cached_bytes < size) {
        size = max_cached_bytes;
      }
      if (memory_fraction > 0 and memory_fraction < size) {
        size = memory_fraction;
      }
      return size;
    }

    // return (bin, bin size)
    std::tuple<unsigned int, size_t> findBin(const size_t bytes) const {
      if (bytes < min_bin_bytes_) {
        return std::make_tuple(min_bin_, min_bin_bytes_);
      }
      if (bytes > max_bin_bytes_) {
        throw std::runtime_error("Requested allocation size " + std::to_string(bytes) +
                                 " bytes is too large for the caching detail with maximum bin " +
                                 std::to_string(max_bin_bytes_) +
                                 " bytes. You might want to increase the maximum bin size");
      }
      unsigned int bin = min_bin_;
      size_t bin_bytes = min_bin_bytes_;
      while (bin_bytes < bytes) {
        ++bin;
        bin_bytes *= bin_growth_;
      }
      return std::make_tuple(bin, bin_bytes);
    }

    bool try_reuse_cached_block(BlockDescriptor& block) {
      std::scoped_lock lock(mutex_);

      // iterate through the range of cached blocks in the same bin
      const auto [begin, end] = cached_blocks_.equal_range(block.bin);
      for (auto iblock = begin; iblock != end; ++iblock) {
        if ((reuse_same_queue_allocations_ and (*block.queue == *(iblock->second.queue))) or
            alpaka::isComplete(*(iblock->second.event))) {
          // associate the cached buffer to the new queue
          auto queue = std::move(*(block.queue));
          // TODO cache (or remove) the debug information and use std::move()
          block = iblock->second;
          block.queue = std::move(queue);

          // if the new queue is on different device than the old event, create a new event
          if (block.device() != alpaka::getDev(*(block.event))) {
            block.event = Event{block.device()};
          }

          // insert the cached block into the live blocks
          // TODO cache (or remove) the debug information and use std::move()
          live_blocks_[block.buffer->data()] = block;

          // update the accounting information
          cached_bytes_.free -= block.bytes;
          cached_bytes_.live += block.bytes;
          cached_bytes_.requested += block.requested;

          if (debug_) {
            std::ostringstream out;
            out << "\t" << device_type_ << " " << alpaka::getName(device_) << " reused cached block at "
                << block.buffer->data() << " (" << block.bytes << " bytes) for queue "
                << block.queue->m_spQueueImpl.get() << ", event " << block.event->m_spEventImpl.get()
                << " (previously associated with stream " << iblock->second.queue->m_spQueueImpl.get() << " , event "
                << iblock->second.event->m_spEventImpl.get() << ")." << std::endl;
            std::cout << out.str() << std::endl;
          }

          // remove the reused block from the list of cached blocks
          cached_blocks_.erase(iblock);
          return true;
        }
      }

      return false;
    }

    Buffer allocate_buffer(size_t bytes, Queue const& queue) {
      if constexpr (std::is_same_v<Device, alpaka::Dev<Queue>>) {
        // allocate device memory
        return alpaka::allocBuf<std::byte, size_t>(device_, bytes);
      } else if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
        // allocate pinned host memory accessible by the queue's platform
        using TPlatform = alpaka::Platform<alpaka::Dev<Queue>>;
        return alpaka::allocMappedBuf<std::byte, size_t>(device_, ::ffx::platform<TPlatform>(), bytes);
      }
      // unsupported combination
      static_assert(std::is_same_v<Device, alpaka::Dev<Queue>> or std::is_same_v<Device, alpaka::DevCpu>,
                    "The \"memory device\" type can either be the same as the "
                    "\"synchronisation device\" "
                    "type, or be "
                    "the host CPU.");
    }

    void allocate_new_block(BlockDescriptor& block) {
      try {
        block.buffer = allocate_buffer(block.bytes, *block.queue);
      } catch (std::runtime_error const&) {
        // the allocation attempt failed: free all cached blocks on the device and retry
        if (debug_) {
          std::ostringstream out;
          out << "\t" << device_type_ << " " << alpaka::getName(device_) << " failed to allocate " << block.bytes
              << " bytes for queue " << block.queue->m_spQueueImpl.get()
              << ", retrying after freeing cached allocations" << std::endl;
          std::cout << out.str() << std::endl;
        }
        // TODO implement a method that frees only up to block.bytes bytes
        free_all_cached();

        // throw an exception if it fails again
        block.buffer = allocate_buffer(block.bytes, *block.queue);
      }

      // create a new event associated to the "synchronisation device"
      block.event = Event{block.device()};

      {
        std::scoped_lock lock(mutex_);
        cached_bytes_.live += block.bytes;
        cached_bytes_.requested += block.requested;
        // TODO use std::move() ?
        live_blocks_[block.buffer->data()] = block;
      }

      if (debug_) {
        std::ostringstream out;
        out << "\t" << device_type_ << " " << alpaka::getName(device_) << " allocated new block at "
            << block.buffer->data() << " (" << block.bytes << " bytes associated with queue "
            << block.queue->m_spQueueImpl.get() << ", event " << block.event->m_spEventImpl.get() << "." << std::endl;
        std::cout << out.str() << std::endl;
      }
    }

    void free_all_cached() {
      std::scoped_lock lock(mutex_);

      while (not cached_blocks_.empty()) {
        auto iblock = cached_blocks_.begin();
        cached_bytes_.free -= iblock->second.bytes;

        if (debug_) {
          std::ostringstream out;
          out << "\t" << device_type_ << " " << alpaka::getName(device_) << " freed " << iblock->second.bytes
              << " bytes.\n\t\t  " << (cached_blocks_.size() - 1) << " available blocks cached (" << cached_bytes_.free
              << " bytes), " << live_blocks_.size() << " live blocks (" << cached_bytes_.live << " bytes) outstanding."
              << std::endl;
          std::cout << out.str() << std::endl;
        }

        cached_blocks_.erase(iblock);
      }
    }

    using CachedBlocks = std::multimap<unsigned int, BlockDescriptor>;  // ordered by the allocation bin
    // TODO replace with a tbb::concurrent_map ?
    using BusyBlocks = std::map<void*, BlockDescriptor>;  // ordered by the address of the allocated memory

    inline static const std::string device_type_ = boost::core::demangle(typeid(Device).name());

    mutable std::mutex mutex_;
    Device device_;  // the device where the memory is allocated

    CachedBytes cached_bytes_;
    CachedBlocks cached_blocks_;  // Set of cached device allocations available for reuse
    BusyBlocks live_blocks_;      // map of pointers to the live device allocations currently in use

    const unsigned int bin_growth_;  // Geometric growth factor for bin-sizes
    const unsigned int min_bin_;
    const unsigned int max_bin_;

    const size_t min_bin_bytes_;
    const size_t max_bin_bytes_;
    const size_t max_cached_bytes_;  // Maximum aggregate cached bytes per device

    const bool reuse_same_queue_allocations_;
    const bool debug_;
  };

}  // namespace ffx
