#pragma once

#include "ffx/framework/concurrency/concurrent_lane.h"
#include "ffx/core/detail/concepts.h"

namespace ffx::framework::concurrency {

  template <concepts::queue TQueue>
  class ConcurrentLaneDispatch {
  public:
    using concurrent_lanes_pool_t = std::vector<std::unique_ptr<ConcurrentLane<TQueue>>>;

    explicit ConcurrentLaneDispatch(concurrent_lanes_pool_t& pool) : pool_(pool) {
      assert(!pool_.empty() && "ConcurrentLane pool cannot be empty!");
    }

    ConcurrentLane<TQueue>& concurrent_lane(const std::size_t batch_id) noexcept {
      // affine mapping
      const std::size_t index = batch_id % pool_.size();
      return *pool_[index];
    }

  private:
    concurrent_lanes_pool_t& pool_;
  };

}  // namespace ffx::framework::concurrency