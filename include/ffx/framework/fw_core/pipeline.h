#pragma once

#include <alpaka/alpaka.hpp>
#include <cstddef>
#include <thread>
#include <vector>

#include "ffx/core/detail/concepts.h"
#include "ffx/core/mem/buf/device_buffer.h"
#include "ffx/core/mem/view/host_view.h"
#include "ffx/framework/concurrency/concurrent_lane.h"
#include "ffx/framework/concurrency/concurrent_lane_dispatch.h"
#include "ffx/framework/fw_core/data_stream.h"
#include "ffx/framework/fw_core/scheduler.h"

namespace ffx::framework {

  namespace detail {

    template <typename>
    struct extract_batch_element_type;

    template <typename T>
    struct extract_batch_element_type<std::optional<batch_t<T>>> {
      using type = T;
    };

    template <typename Provider>
    struct provider_data_type {
      using get_return_t = decltype(std::declval<Provider&>().get());
      using type = typename extract_batch_element_type<std::decay_t<get_return_t>>::type;
    };

    template <typename Provider>
    using provider_data_type_t = typename provider_data_type<std::decay_t<Provider>>::type;

  }  // namespace detail

  template <typename TDataProvider>
  concept data_provider = requires(TDataProvider& provider) {
    { provider.get() } -> std::same_as<std::optional<batch_t<detail::provider_data_type_t<TDataProvider>>>>;
  };

  template <typename T>
  concept direct_mem_access_compatible = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

  template <concepts::queue TQueue>
  class Pipeline {
  public:
    using Device = alpaka::Dev<TQueue>;

    explicit Pipeline(const Device& device,
                      const std::size_t number_of_concurrent_lanes = std::thread::hardware_concurrency())
        : number_of_concurrent_lanes_(concurrency::get_number_of_threads(number_of_concurrent_lanes)) {
      concurrency_pool_.reserve(number_of_concurrent_lanes_);
      for (auto index = 0zu; index < number_of_concurrent_lanes_; ++index)
        concurrency_pool_.push_back(std::make_unique<concurrency::ConcurrentLane<TQueue>>(device, index));
      dispatcher_ = std::make_unique<concurrency::ConcurrentLaneDispatch<TQueue>>(concurrency_pool_);
    }

    template <typename TModule, typename... Args>
    void add_module(Args&&... args) {
      for (auto& concurrent_lane : concurrency_pool_)
        concurrent_lane->scheduler().add_module(std::make_shared<TModule>(args...));
      is_pipeline_ready_ = false;
    }

    void build() {
      if (is_pipeline_ready_)
        return;

      for (auto& concurrent_lane : concurrency_pool_)
        concurrent_lane->scheduler().build_pipeline(concurrent_lane->queue());
      is_pipeline_ready_ = true;
    }

    template <data_provider TDataProvider>
    void dispatch(TDataProvider&& data_provider) {
      using T = detail::provider_data_type_t<TDataProvider>;
      static_assert(direct_mem_access_compatible<T>,
                    "Stream record type T must be trivially copyable and standard layout for DMA!");

      build();

      std::size_t batch_index = 0;
      while (auto batch = data_provider.get()) {
        const std::size_t batch_size = batch->size;
        const T* data = batch->data;

        if (batch_size == 0 || data == nullptr) {
          ++batch_index;
          continue;
        }

        auto& concurrent_lane = dispatcher_->concurrent_lane(batch_index);
        concurrent_lane.submit([batch_index, batch_size, data](TQueue& queue,
                                                               concurrency::ConcurrentLaneMemory& shared_memory,
                                                               Scheduler<TQueue>& scheduler) {
          auto device_data = ffx::make_device_buffer<T[]>(queue, batch_size);
          auto host_view = ffx::make_host_view(data, batch_size);
          alpaka::memcpy(queue, device_data, host_view);

          const auto context = Context(queue, shared_memory, batch_index, batch_size);
          auto token = Token<device_buffer<Device, T[]>>{"data"};
          context.put(token, std::move(device_data));

          scheduler.dispatch(context);
        });

        ++batch_index;
      }

      // drain all pending tasks across lanes
      for (auto& concurrent_lane : concurrency_pool_)
        concurrent_lane->wait_tasks();
      // sync queues
      for (auto& concurrent_lane : concurrency_pool_)
        concurrent_lane->sync();
      // clear shared memory
      for (auto& concurrent_lane : concurrency_pool_)
        concurrent_lane->shared_memory().clear();
    }

    template <typename T>
    void dispatch(const std::string_view filepath) {
      DataStream<T> file_stream(filepath);
      dispatch(std::move(file_stream));
    }

  private:
    using concurrent_lanes_pool_t = std::vector<std::unique_ptr<concurrency::ConcurrentLane<TQueue>>>;

    const std::size_t number_of_concurrent_lanes_;
    concurrent_lanes_pool_t concurrency_pool_;
    std::unique_ptr<concurrency::ConcurrentLaneDispatch<TQueue>> dispatcher_;

    bool is_pipeline_ready_{false};
  };

}  // namespace ffx::framework
