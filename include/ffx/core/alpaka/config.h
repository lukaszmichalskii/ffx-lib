#pragma once

#include <alpaka/alpaka.hpp>

namespace ffx_alpaka {

  using Idx = uint32_t;
  using Extent = uint32_t;
  using Offsets = Extent;

  using Dim0D = alpaka::DimInt<0u>;
  using Dim1D = alpaka::DimInt<1u>;
  using Dim2D = alpaka::DimInt<2u>;
  using Dim3D = alpaka::DimInt<3u>;

  template <typename TDim>
  using Vec = alpaka::Vec<TDim, Idx>;
  using Vec1D = Vec<Dim1D>;
  using Vec2D = Vec<Dim2D>;
  using Vec3D = Vec<Dim3D>;
  using Scalar = Vec<Dim0D>;

  template <typename TDim>
  using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;
  using WorkDiv1D = WorkDiv<Dim1D>;
  using WorkDiv2D = WorkDiv<Dim2D>;
  using WorkDiv3D = WorkDiv<Dim3D>;

  using DevHost = alpaka::DevCpu;
  using PlatformHost = alpaka::PlatformCpu;

}  // namespace ffx_alpaka

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

namespace ffx_cuda_async {

  using namespace ffx_alpaka;

  using Platform = alpaka::PlatformCudaRt;
  using Device = alpaka::DevCudaRt;
  using Queue = alpaka::QueueCudaRtNonBlocking;
  using Event = alpaka::EventCudaRt;

  template <typename TDim>
  using Acc = alpaka::AccGpuCudaRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace ffx_cuda_async

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

namespace ffx_hip_async {

  using namespace ffx_alpaka;

  using Platform = alpaka::PlatformHipRt;
  using Device = alpaka::DevHipRt;
  using Queue = alpaka::QueueHipRtNonBlocking;
  using Event = alpaka::EventHipRt;

  template <typename TDim>
  using Acc = alpaka::AccGpuHipRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace ffx_hip_async

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

namespace ffx_tbb_async {

  using namespace ffx_alpaka;

  using Platform = alpaka::PlatformCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuNonBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuTbbBlocks<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace ffx_tbb_async

#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

namespace ffx_omp2_async {

  using namespace ffx_alpaka;

  using Platform = alpaka::PlatformCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuOmp2Blocks<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace ffx_omp2_async

#endif  // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

namespace ffx_serial_sync {

  using namespace ffx_alpaka;

  using Platform = alpaka::PlatformCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuSerial<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace ffx_serial_sync

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

// Set ffx to the highest-priority enabled backend exactly once.
// Alpaka always enables ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED alongside every
// other CPU backend, so using a single elif chain avoids redefinition warnings.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#define ffx_runtime ffx_cuda_async
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#define ffx_runtime ffx_hip_async
#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
#define ffx_runtime ffx_tbb_async
#elif defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#define ffx_runtime ffx_omp2_async
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
#define ffx_runtime ffx_serial_sync
#endif
