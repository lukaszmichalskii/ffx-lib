#pragma once

#include <memory>

#include "ffx/core/detail/concepts.h"
#include "ffx/framework/concurrency/concurrent_lane_memory.h"
#include "ffx/framework/utilities/token.h"

namespace ffx::framework {

  struct Meta {
    const std::size_t batch_id;
    const std::size_t batch_size;

    constexpr Meta(const std::size_t batch_id, const std::size_t batch_size) noexcept
        : batch_id(batch_id), batch_size(batch_size) {}
  };

  template <concepts::queue TQueue>
  class Context {
  public:
    Context(TQueue& queue,
            concurrency::ConcurrentLaneMemory& memory_pool,
            const std::size_t batch_id,
            const std::size_t batch_size) noexcept
        : queue_(queue), memory_(memory_pool), meta_(batch_id, batch_size) {}

    template <typename T>
    std::shared_ptr<const T> get(Token<T> token) const {
      return memory_.get(meta_.batch_id, token);
    }

    template <typename T>
    void put(Token<T> token, T&& data) const {
      memory_.put(meta_.batch_id, token, std::forward<T>(data));
    }

    TQueue& queue() const { return queue_; }

    [[nodiscard]] const Meta& meta() const { return meta_; }

  private:
    TQueue& queue_;
    concurrency::ConcurrentLaneMemory& memory_;
    const Meta meta_;
  };

}  // namespace ffx::framework
