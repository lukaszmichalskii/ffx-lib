#pragma once

#include <ffx/ffx.h>

#include "model.h"
#include "mnist_config.h"

namespace ffx_runtime::mnist {

  using namespace ffx_mnist;

  class Inference : public ffx::framework::Module<Queue> {
  public:
    Inference()
        : input_token_(this->consumes<input_data_t>("features")),
          output_token_(this->produces<output_data_t>("logits")) {}

    void init(Queue& queue) override { model_.emplace(queue); }

    void dispatch(const ffx::framework::Context<Queue>& context) const override {
#ifdef DEBUG
      std::println("[Inference] Batch {}, queue {}",
                   context.meta().batch_id,
                   static_cast<void*>(alpaka::getNativeHandle(context.queue())));
#endif
      nvtx_tools::ScopedRange range("Inference", nvtx_tools::colors::Red);
      const auto input_data = context.get(input_token_);
      const auto size = context.meta().batch_size * kNumberOfClasses;
      auto output_data = ffx::make_device_buffer<float[]>(context.queue(), size);

      model_->infer(context.queue(), *input_data, output_data);

      context.put(output_token_, std::move(output_data));
    }

  private:
    using input_data_t = ffx::device_buffer<Device, float[]>;
    using output_data_t = ffx::device_buffer<Device, float[]>;

    const ffx::framework::Token<input_data_t> input_token_;
    const ffx::framework::Token<output_data_t> output_token_;

    mutable std::optional<Model> model_;
  };

}  // namespace ffx_runtime::mnist
