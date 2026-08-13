#pragma once

#include <ffx/ffx.h>

#include "mnist_config.h"

namespace ffx_runtime {

  using namespace ffx_mnist::config;

  extern "C" {
  extern const unsigned char _binary_model_data_start[];
  }

  class Model {
  public:
    explicit Model(Queue& queue)
        : device_node_conv_25_relu_output_(ffx::make_device_buffer<float[]>(queue, kBatchSize * 25088)),
          device_node_conv_25_relu_weight_(ffx::make_device_buffer<float[]>(queue, 288)),
          device_node_conv_25_relu_bias_(ffx::make_device_buffer<float[]>(queue, 32)),
          device_node_conv_26_relu_output_(ffx::make_device_buffer<float[]>(queue, kBatchSize * 50176)),
          device_node_conv_26_relu_weight_(ffx::make_device_buffer<float[]>(queue, 18432)),
          device_node_conv_26_relu_bias_(ffx::make_device_buffer<float[]>(queue, 64)),
          device_node_max_pool2d_output_(ffx::make_device_buffer<float[]>(queue, kBatchSize * 12544)),
          device_node_conv_27_relu_output_(ffx::make_device_buffer<float[]>(queue, kBatchSize * 25088)),
          device_node_conv_27_relu_weight_(ffx::make_device_buffer<float[]>(queue, 73728)),
          device_node_conv_27_relu_bias_(ffx::make_device_buffer<float[]>(queue, 128)),
          device_node_max_pool2d_1_output_(ffx::make_device_buffer<float[]>(queue, kBatchSize * 6272)),
          device_node_linear_relu_output_(ffx::make_device_buffer<float[]>(queue, kBatchSize * 128)),
          device_node_linear_relu_weight_(ffx::make_device_buffer<float[]>(queue, 802816)),
          device_node_linear_relu_bias_(ffx::make_device_buffer<float[]>(queue, 128)),
          device_node_linear_1_output_(ffx::make_device_buffer<float[]>(queue, kBatchSize * 10)),
          device_node_linear_1_weight_(ffx::make_device_buffer<float[]>(queue, 1280)),
          device_node_linear_1_bias_(ffx::make_device_buffer<float[]>(queue, 10)) {
      {
        const auto node_conv_25_relu_weight_view = alpaka::createView(
            ffx::host(), reinterpret_cast<const float*>(_binary_model_data_start + 0), static_cast<Extent>(288));
        alpaka::memcpy(queue, device_node_conv_25_relu_weight_, node_conv_25_relu_weight_view);
      }
      {
        const auto node_conv_25_relu_bias_view = alpaka::createView(
            ffx::host(), reinterpret_cast<const float*>(_binary_model_data_start + 1152), static_cast<Extent>(32));
        alpaka::memcpy(queue, device_node_conv_25_relu_bias_, node_conv_25_relu_bias_view);
      }
      {
        const auto node_conv_26_relu_weight_view = alpaka::createView(
            ffx::host(), reinterpret_cast<const float*>(_binary_model_data_start + 1280), static_cast<Extent>(18432));
        alpaka::memcpy(queue, device_node_conv_26_relu_weight_, node_conv_26_relu_weight_view);
      }
      {
        const auto node_conv_26_relu_bias_view = alpaka::createView(
            ffx::host(), reinterpret_cast<const float*>(_binary_model_data_start + 75008), static_cast<Extent>(64));
        alpaka::memcpy(queue, device_node_conv_26_relu_bias_, node_conv_26_relu_bias_view);
      }
      {
        const auto node_conv_27_relu_weight_view = alpaka::createView(
            ffx::host(), reinterpret_cast<const float*>(_binary_model_data_start + 75264), static_cast<Extent>(73728));
        alpaka::memcpy(queue, device_node_conv_27_relu_weight_, node_conv_27_relu_weight_view);
      }
      {
        const auto node_conv_27_relu_bias_view = alpaka::createView(
            ffx::host(), reinterpret_cast<const float*>(_binary_model_data_start + 370176), static_cast<Extent>(128));
        alpaka::memcpy(queue, device_node_conv_27_relu_bias_, node_conv_27_relu_bias_view);
      }
      {
        const auto node_linear_relu_weight_view =
            alpaka::createView(ffx::host(),
                               reinterpret_cast<const float*>(_binary_model_data_start + 370688),
                               static_cast<Extent>(802816));
        alpaka::memcpy(queue, device_node_linear_relu_weight_, node_linear_relu_weight_view);
      }
      {
        const auto node_linear_relu_bias_view = alpaka::createView(
            ffx::host(), reinterpret_cast<const float*>(_binary_model_data_start + 3581952), static_cast<Extent>(128));
        alpaka::memcpy(queue, device_node_linear_relu_bias_, node_linear_relu_bias_view);
      }
      {
        const auto node_linear_1_weight_view = alpaka::createView(
            ffx::host(), reinterpret_cast<const float*>(_binary_model_data_start + 3582464), static_cast<Extent>(1280));
        alpaka::memcpy(queue, device_node_linear_1_weight_, node_linear_1_weight_view);
      }
      {
        const auto node_linear_1_bias_view = alpaka::createView(
            ffx::host(), reinterpret_cast<const float*>(_binary_model_data_start + 3587584), static_cast<Extent>(10));
        alpaka::memcpy(queue, device_node_linear_1_bias_, node_linear_1_bias_view);
      }
    }

    void infer(Queue& queue,
               const ffx::device_buffer<Device, float[]>& device_input,
               ffx::device_buffer<Device, float[]>& device_output) {
      {
        // node_conv_25_relu
        constexpr auto blocks_node_conv_25_relu = ffx::divide_up_by(kBatchSize * 25088, kThreadsPerBlock);
        const auto grid_node_conv_25_relu = ffx::make_workdiv<Acc1D>(blocks_node_conv_25_relu, kThreadsPerBlock);
        alpaka::exec<Acc1D>(queue,
                            grid_node_conv_25_relu,
                            node_conv_25_relu_,
                            device_input.data(),
                            device_node_conv_25_relu_output_.data(),
                            device_node_conv_25_relu_weight_.data(),
                            device_node_conv_25_relu_bias_.data());
      }
      {
        // node_conv_26_relu
        constexpr auto blocks_node_conv_26_relu = ffx::divide_up_by(kBatchSize * 50176, kThreadsPerBlock);
        const auto grid_node_conv_26_relu = ffx::make_workdiv<Acc1D>(blocks_node_conv_26_relu, kThreadsPerBlock);
        alpaka::exec<Acc1D>(queue,
                            grid_node_conv_26_relu,
                            node_conv_26_relu_,
                            device_node_conv_25_relu_output_.data(),
                            device_node_conv_26_relu_output_.data(),
                            device_node_conv_26_relu_weight_.data(),
                            device_node_conv_26_relu_bias_.data());
      }
      {
        // node_max_pool2d
        constexpr auto blocks_node_max_pool2d = ffx::divide_up_by(kBatchSize * 12544, kThreadsPerBlock);
        const auto grid_node_max_pool2d = ffx::make_workdiv<Acc1D>(blocks_node_max_pool2d, kThreadsPerBlock);
        alpaka::exec<Acc1D>(queue,
                            grid_node_max_pool2d,
                            node_max_pool2d_,
                            device_node_conv_26_relu_output_.data(),
                            device_node_max_pool2d_output_.data());
      }
      {
        // node_conv_27_relu
        constexpr auto blocks_node_conv_27_relu = ffx::divide_up_by(kBatchSize * 25088, kThreadsPerBlock);
        const auto grid_node_conv_27_relu = ffx::make_workdiv<Acc1D>(blocks_node_conv_27_relu, kThreadsPerBlock);
        alpaka::exec<Acc1D>(queue,
                            grid_node_conv_27_relu,
                            node_conv_27_relu_,
                            device_node_max_pool2d_output_.data(),
                            device_node_conv_27_relu_output_.data(),
                            device_node_conv_27_relu_weight_.data(),
                            device_node_conv_27_relu_bias_.data());
      }
      {
        // node_max_pool2d_1
        constexpr auto blocks_node_max_pool2d_1 = ffx::divide_up_by(kBatchSize * 6272, kThreadsPerBlock);
        const auto grid_node_max_pool2d_1 = ffx::make_workdiv<Acc1D>(blocks_node_max_pool2d_1, kThreadsPerBlock);
        alpaka::exec<Acc1D>(queue,
                            grid_node_max_pool2d_1,
                            node_max_pool2d_1_,
                            device_node_conv_27_relu_output_.data(),
                            device_node_max_pool2d_1_output_.data());
      }
      {
        // node_linear_relu
        constexpr auto blocks_node_linear_relu = ffx::divide_up_by(kBatchSize * 128, kThreadsPerBlock);
        const auto grid_node_linear_relu = ffx::make_workdiv<Acc1D>(blocks_node_linear_relu, kThreadsPerBlock);
        alpaka::exec<Acc1D>(queue,
                            grid_node_linear_relu,
                            node_linear_relu_,
                            device_node_max_pool2d_1_output_.data(),
                            device_node_linear_relu_output_.data(),
                            device_node_linear_relu_weight_.data(),
                            device_node_linear_relu_bias_.data());
      }
      {
        // node_linear_1
        constexpr auto blocks_node_linear_1 = ffx::divide_up_by(kBatchSize * 10, kThreadsPerBlock);
        const auto grid_node_linear_1 = ffx::make_workdiv<Acc1D>(blocks_node_linear_1, kThreadsPerBlock);
        alpaka::exec<Acc1D>(queue,
                            grid_node_linear_1,
                            node_linear_1_,
                            device_node_linear_relu_output_.data(),
                            device_node_linear_1_output_.data(),
                            device_node_linear_1_weight_.data(),
                            device_node_linear_1_bias_.data());
      }

      alpaka::memcpy(queue, device_output, device_node_linear_1_output_);
    }

  private:
    // kernels
    ffx::nn::Conv2dReLU<kBatchSize, 28, 28, 1, 32, 3, 3, 1, 1, 1, 1> node_conv_25_relu_{};
    ffx::nn::Conv2dReLU<kBatchSize, 28, 28, 32, 64, 3, 3, 1, 1, 1, 1> node_conv_26_relu_{};
    ffx::nn::MaxPool2d<kBatchSize, 28, 28, 64, 2, 2, 2, 2, 0, 0> node_max_pool2d_{};
    ffx::nn::Conv2dReLU<kBatchSize, 14, 14, 64, 128, 3, 3, 1, 1, 1, 1> node_conv_27_relu_{};
    ffx::nn::MaxPool2d<kBatchSize, 14, 14, 128, 2, 2, 2, 2, 0, 0> node_max_pool2d_1_{};
    ffx::nn::LinearReLU<kBatchSize, 6272, 128> node_linear_relu_{};
    ffx::nn::Linear<kBatchSize, 128, 10> node_linear_1_{};

    // params & intermediate buffers
    ffx::device_buffer<Device, float[]> device_node_conv_25_relu_output_;
    ffx::device_buffer<Device, float[]> device_node_conv_25_relu_weight_;
    ffx::device_buffer<Device, float[]> device_node_conv_25_relu_bias_;
    ffx::device_buffer<Device, float[]> device_node_conv_26_relu_output_;
    ffx::device_buffer<Device, float[]> device_node_conv_26_relu_weight_;
    ffx::device_buffer<Device, float[]> device_node_conv_26_relu_bias_;
    ffx::device_buffer<Device, float[]> device_node_max_pool2d_output_;
    ffx::device_buffer<Device, float[]> device_node_conv_27_relu_output_;
    ffx::device_buffer<Device, float[]> device_node_conv_27_relu_weight_;
    ffx::device_buffer<Device, float[]> device_node_conv_27_relu_bias_;
    ffx::device_buffer<Device, float[]> device_node_max_pool2d_1_output_;
    ffx::device_buffer<Device, float[]> device_node_linear_relu_output_;
    ffx::device_buffer<Device, float[]> device_node_linear_relu_weight_;
    ffx::device_buffer<Device, float[]> device_node_linear_relu_bias_;
    ffx::device_buffer<Device, float[]> device_node_linear_1_output_;
    ffx::device_buffer<Device, float[]> device_node_linear_1_weight_;
    ffx::device_buffer<Device, float[]> device_node_linear_1_bias_;
  };

}  // namespace ffx_runtime