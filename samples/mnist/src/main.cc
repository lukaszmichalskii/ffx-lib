#include <ffx/ffx.h>

#include "mnist_config.h"
#include "preprocessing.h"
#include "inference.h"
#include "postprocessing.h"
#ifdef DEBUG
#include "validator.h"
#endif  // DEBUG

namespace ffx_runtime {

  using namespace mnist::config;

  int main() {
    nvtx_tools::ScopedRange range("MNIST", nvtx_tools::colors::White);
    const auto& device = ffx::devices<Platform>()[0];
    auto pipeline = ffx::framework::Pipeline<Queue, kConcurrentLaneCapacity>(device, kNumberOfThreads);

    pipeline.add_module<mnist::Preprocessing>();
    pipeline.add_module<mnist::Inference>();
    pipeline.add_module<mnist::Postprocessing>();
#ifdef DEBUG
    pipeline.add_module<mnist::Validator>();
#endif  // DEBUG

    auto stream = ffx::framework::DataStream<mnist_sample_t>(kFilepath);
    nvtx_tools::ScopedRange range1("Dispatch", nvtx_tools::colors::White);
    pipeline.dispatch(stream);

    return 0;
  }

}  // namespace ffx_runtime

int main() { return ffx_runtime::main(); }
