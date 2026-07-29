#pragma once

#include "ffx/core/ffx_core.h"

#ifdef FFX_ALGORITHM_ENABLED
#include "ffx/algorithm/copy_if.h"
#include "ffx/algorithm/count_if.h"
#include "ffx/algorithm/extrema.h"
#include "ffx/algorithm/for_each.h"
#include "ffx/algorithm/reduce.h"
#include "ffx/algorithm/scan.h"
#include "ffx/algorithm/sort.h"
#include "ffx/algorithm/transform.h"
#endif  // FFX_ALGORITHM_ENABLED

#ifdef FFX_FRAMEWORK_ENABLED
#include "ffx/framework/fw_core/environment.h"
#endif  // FFX_FRAMEWORK_ENABLED

#ifdef FFX_NN_ENABLED
#include "ffx/nn/functional/add.h"
#include "ffx/nn/functional/div.h"
#include "ffx/nn/functional/gelu.h"
#include "ffx/nn/functional/hardswish.h"
#include "ffx/nn/functional/identity.h"
#include "ffx/nn/functional/leaky_relu.h"
#include "ffx/nn/functional/mul.h"
#include "ffx/nn/functional/relu.h"
#include "ffx/nn/functional/relu6.h"
#include "ffx/nn/functional/sigmoid.h"
#include "ffx/nn/functional/silu.h"
#include "ffx/nn/functional/sub.h"
#include "ffx/nn/functional/tanh.h"
#include "ffx/nn/adaptive_avg_pool2d.h"
#include "ffx/nn/adaptive_max_pool2d.h"
#include "ffx/nn/add.h"
#include "ffx/nn/avg_pool2d.h"
#include "ffx/nn/batch_norm2d.h"
#include "ffx/nn/conv2d.h"
#include "ffx/nn/div.h"
#include "ffx/nn/element_wise.h"
#include "ffx/nn/gelu.h"
#include "ffx/nn/hardswish.h"
#include "ffx/nn/layer_norm.h"
#include "ffx/nn/leaky_relu.h"
#include "ffx/nn/linear.h"
#include "ffx/nn/mat_mul.h"
#include "ffx/nn/max_pool2d.h"
#include "ffx/nn/mul.h"
#include "ffx/nn/relu.h"
#include "ffx/nn/relu6.h"
#include "ffx/nn/rms_norm.h"
#include "ffx/nn/sigmoid.h"
#include "ffx/nn/silu.h"
#include "ffx/nn/softmax.h"
#include "ffx/nn/sub.h"
#include "ffx/nn/tanh.h"
#endif  // FFX_NN_ENABLED
