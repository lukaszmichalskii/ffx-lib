#include "ffx/ffx.h"

namespace ffx_runtime {

  int main() {
    ffx::framework::get_environment_info<Platform>();
    return 0;
  }

}  // namespace ffx_runtime

int main() { return ffx_runtime::main(); }