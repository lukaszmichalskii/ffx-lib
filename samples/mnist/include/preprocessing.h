#pragma once

#include <ffx/ffx.h>

#include "mnist_config.h"

namespace ffx_runtime::mnist {

  using namespace ffx_mnist;

  class Preprocessing : public ffx::framework::Module<Queue> {
  public:
    struct PreprocessingKernel {
      template <ffx::concepts::accelerator TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    const config::mnist_sample_t* input,
                                    float* output,
                                    const std::size_t size) const {
        for (const auto thread_index : alpaka::uniformElements(acc, size)) {
          const auto sample_idx = thread_index / 784zu;
          const auto pixel_idx = thread_index % 784zu;
          const auto scaled = static_cast<float>(input[sample_idx].pixels[pixel_idx]) / 255.0f;
          output[thread_index] = (scaled - 0.1307f) / 0.3081f;
        }
      }
    };

    Preprocessing()
        : input_token_(this->consumes<input_data_t>("data")),
          output_token_(this->produces<output_data_t>("features")) {}

    void dispatch(const ffx::framework::Context<Queue>& context) const override {
#ifdef DEBUG
      std::println("[Preprocessing] Batch {}, queue {}",
                   context.meta().batch_id,
                   static_cast<void*>(alpaka::getNativeHandle(context.queue())));
#endif
      nvtx_tools::ScopedRange range("Preprocessing", nvtx_tools::colors::Blue);
      const auto& input_data = context.get(input_token_);
      const auto size = context.meta().batch_size * input_data->data()->pixels.size();
      auto output_data = ffx::make_device_buffer<float[]>(context.queue(), size);

      const auto blocks_per_grid = ffx::divide_up_by(size, config::kThreadsPerBlock);
      const auto grid = ffx::make_workdiv<Acc1D>(blocks_per_grid, config::kThreadsPerBlock);
      alpaka::exec<Acc1D>(context.queue(), grid, PreprocessingKernel{}, input_data->data(), output_data.data(), size);

      context.put(output_token_, std::move(output_data));
    }

  private:
    using input_data_t = ffx::device_buffer<Device, config::mnist_sample_t[]>;
    using output_data_t = ffx::device_buffer<Device, float[]>;

    const ffx::framework::Token<input_data_t> input_token_;
    const ffx::framework::Token<output_data_t> output_token_;
  };

}  // namespace ffx_runtime::mnist
