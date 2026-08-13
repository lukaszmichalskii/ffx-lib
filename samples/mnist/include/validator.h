#pragma once

#include <cassert>
#include <format>
#include <iterator>
#include <print>
#include <string>

#include <ffx/ffx.h>
#include "mnist_config.h"

namespace ffx_runtime::mnist {

  using namespace ffx_mnist::config;

  class Validator : public ffx::framework::Module<Queue> {
  public:
    Validator()
        : expected_token_(this->consumes<dataset_t>("data")),
          actual_token_(this->consumes<output_data_t>("predictions")) {}

    void dispatch(const ffx::framework::Context<Queue>& context) const override {
      nvtx_tools::ScopedRange range("Validator", nvtx_tools::colors::Amber);
      const auto& expected_data = context.get(expected_token_);
      const auto& actual_data = context.get(actual_token_);

      auto expected = ffx::make_host_buffer<mnist_sample_t[]>(context.queue(), expected_data->size());
      auto actual = ffx::make_host_buffer<std::size_t[]>(context.queue(), actual_data->size());

      alpaka::memcpy(context.queue(), expected, *expected_data);
      alpaka::memcpy(context.queue(), actual, *actual_data);
      alpaka::wait(context.queue());

      std::size_t correct = 0;
      for (auto index = 0zu; index < actual.size(); ++index) {
        if (actual.data()[index] == expected.data()[index].label) {
          ++correct;
        }
      }

      const float accuracy = (static_cast<float>(correct) / actual.size()) * 100.0f;

      std::string expected_predictions_str;
      for (auto index = 0zu; index < expected.size(); ++index) {
        std::format_to(std::back_inserter(expected_predictions_str),
                       "{}{}",
                       (index > 0 ? ", " : ""),
                       expected.data()[index].label);
      }

      std::string predictions_str;
      for (auto index = 0zu; index < actual.size(); ++index) {
        std::format_to(std::back_inserter(predictions_str), "{}{}", (index > 0 ? ", " : ""), actual.data()[index]);
      }

      std::println("[Validator] Batch {} -> Accuracy: {:.1f}% ({}/{}) | [{}] vs ({})",
                   context.meta().batch_id,
                   accuracy,
                   correct,
                   actual.size(),
                   predictions_str,
                   expected_predictions_str);

      // Soft threshold: Assert only if accuracy drops below expected baseline (e.g., 80%)
      assert(accuracy >= 80.0f && "Batch accuracy fell below tolerance threshold!");
    }

  private:
    using dataset_t = ffx::device_buffer<Device, mnist_sample_t[]>;
    using output_data_t = ffx::device_buffer<Device, std::size_t[]>;

    const ffx::framework::Token<dataset_t> expected_token_;
    const ffx::framework::Token<output_data_t> actual_token_;
  };

}  // namespace ffx_runtime::mnist