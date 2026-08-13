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
    const auto& device = ffx::devices<Platform>()[0];
    // num_lanes = kNumberOfThreads enforces the 1:1 Thread↔Queue lane binding.
    // (kNumberOfQueues == kNumberOfThreads per mnist_config.h, so semantics are unchanged.)
    auto pipeline = ffx::framework::Pipeline<Queue>(device, kNumberOfThreads);

    pipeline.add_module<mnist::Preprocessing>();
    pipeline.add_module<mnist::Inference>();
    pipeline.add_module<mnist::Postprocessing>();
#ifdef DEBUG
    pipeline.add_module<mnist::Validator>();
#endif  // DEBUG

    pipeline.build();

    auto stream = ffx::framework::DataStream<mnist_sample_t>(kFilepath);
    pipeline.dispatch(stream);

    return 0;
  }

}  // namespace ffx_runtime

int main() { return ffx_runtime::main(); }
