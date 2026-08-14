#pragma once

#include <ffx/ffx.h>

#include "mnist_config.h"

namespace ffx_runtime::mnist {

  using namespace ffx_mnist;

  class Postprocessing : public ffx::framework::Module<Queue> {
  public:
    struct PostprocessingKernel {
      template <ffx::concepts::accelerator TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    const float* input,
                                    std::size_t* output,
                                    const std::size_t size) const {
        for (const auto thread_index : alpaka::uniformElements(acc, size)) {
          const auto local_thread_index = thread_index * kNumberOfClasses;
          std::size_t max_class_id = 0;
          float max_logit = input[local_thread_index];

          for (std::size_t class_id = 1; class_id < kNumberOfClasses; ++class_id) {
            const float logit = input[local_thread_index + class_id];
            if (logit > max_logit) {
              max_logit = logit;
              max_class_id = class_id;
            }
          }
          output[thread_index] = max_class_id;
        }
      }
    };

    Postprocessing()
        : dataset_token_(this->consumes<dataset_t>("data")),
          input_token_(this->consumes<input_data_t>("logits")),
          output_token_(this->produces<output_data_t>("predictions")) {}

    void dispatch(const ffx::framework::Context<Queue>& context) const override {
#ifdef DEBUG
      std::println("[Postprocessing] Batch {}, queue {}",
                   context.meta().batch_id,
                   static_cast<void*>(alpaka::getNativeHandle(context.queue())));
#endif
      nvtx_tools::ScopedRange range("Postprocessing", nvtx_tools::colors::Green);
      const auto& input_data = context.get(input_token_);
      const auto size = context.meta().batch_size;
      auto output_data = ffx::make_device_buffer<std::size_t[]>(context.queue(), size);

      const auto blocks_per_grid = ffx::divide_up_by(size, kThreadsPerBlock);
      const auto grid = ffx::make_workdiv<Acc1D>(blocks_per_grid, kThreadsPerBlock);
      alpaka::exec<Acc1D>(context.queue(), grid, PostprocessingKernel{}, input_data->data(), output_data.data(), size);

      context.put(output_token_, std::move(output_data));
    }

  private:
    using dataset_t = ffx::device_buffer<Device, mnist_sample_t[]>;
    using input_data_t = ffx::device_buffer<Device, float[]>;
    using output_data_t = ffx::device_buffer<Device, std::size_t[]>;

    const ffx::framework::Token<dataset_t> dataset_token_;
    const ffx::framework::Token<input_data_t> input_token_;
    const ffx::framework::Token<output_data_t> output_token_;
  };

}  // namespace ffx_runtime::mnist
