#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#define USE_GRACE_HOPPER
#include "copy_sm80.h"
#include "hopper_matmul.h"
#include "pipeline.h"
#include "tma_copy.h"
using namespace cute;
static constexpr  int NUM_WARPS_PER_GROUP = 4;

template <class TMA_10000003, class TMA_10000004>
__global__ void  __launch_bounds__(256) custom_kernel_0(CUTE_GRID_CONSTANT TMA_10000003 const tma_10000003, CUTE_GRID_CONSTANT TMA_10000004 const tma_10000004,  bfloat16_t* dtensor10000005_ptr, bfloat16_t const* dtensor10000003_ptr, bfloat16_t const* dtensor10000004_ptr) {
int thread_idx = threadIdx.x;
static constexpr int NUM_THREADS = 128;
static constexpr int CONSUMER_NUM_THREADS = 128;
// STensors
extern __shared__ char buf[];
bfloat16_t *stensor20000015_ptr = (bfloat16_t*)(buf + 128);
bfloat16_t *stensor20000013_ptr = (bfloat16_t*)(buf + 32896);
bfloat16_t *stensor20000012_ptr = (bfloat16_t*)(buf + 128);
// G->S copy atoms
// Copy for G->S: dtensor 10000003 -> stensor 20000012
using DTensor10000003TileLayout = Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<4096>>>;
tb::HopperAsyncPipeline<2> hopper_async_pipeline_20000012((void *) (buf + 49280), (tb::warpgroup_id() == 1 && tb::warp_id() % NUM_WARPS_PER_GROUP == 0), tb::warpgroup_id() < 1, 16384, 1);
using STensor20000012InputAtom = tb::TMA_COPY<bfloat16_t, decltype(composition(Swizzle<3, 4, 3>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<64>, Int<1>>>{})), Layout<Shape<Int<128>, Int<64>, Int<1>, Int<1>, Int<1>, Int<64>>, Stride<Int<4096>, Int<1>, Int<1>, Int<4096>, Int<1>, Int<64>>>, decltype(tma_10000003), decltype(hopper_async_pipeline_20000012), true, 64>;
// Copy for G->S: dtensor 10000004 -> stensor 20000013
using DTensor10000004TileLayout = Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<4096>>>;
tb::HopperAsyncPipeline<2> hopper_async_pipeline_20000013((void *) (buf + 49312), (tb::warpgroup_id() == 1 && tb::warp_id() % NUM_WARPS_PER_GROUP == 0), tb::warpgroup_id() < 1, 8192, 1);
using STensor20000013InputAtom = tb::TMA_COPY<bfloat16_t, decltype(composition(Swizzle<3, 4, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{})), Layout<Shape<Int<64>, Int<64>, Int<64>, Int<1>, Int<1>, Int<64>>, Stride<Int<1>, Int<4096>, Int<64>, Int<1>, Int<1>, Int<262144>>>, decltype(tma_10000004), decltype(hopper_async_pipeline_20000013), false, 64>;

__syncthreads();
  *((uint128_t*)buf) = 0ul;
  
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000015 -> dtensor 10000005
  bfloat16_t *dtensor10000005_tile_ptr = dtensor10000005_ptr  + blockIdx.x*64*1;
  using DTensor10000005TileLayout = Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<4096>>>;
  using STensor20000015OutputAtom = tb::OutputChunkedSyncCopy<bfloat16_t, DTensor10000005TileLayout, decltype(composition(Swizzle<3, 4, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{})), NUM_THREADS>;
  
  
  using Matmul20000015LayoutA = decltype(composition(Swizzle<3, 4, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000015LayoutB = decltype(composition(Swizzle<3, 4, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000015LayoutC = decltype(composition(Swizzle<3, 4, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000015Kernel = tb::Hopper_Matmul<bfloat16_t, true, false, Matmul20000015LayoutA, Matmul20000015LayoutB, Matmul20000015LayoutC, NUM_THREADS, 0, false, false, true, true, 2>;
  auto matmul_20000015_accum = Matmul20000015Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  int warpgroup_id = tb::warpgroup_id();
  if (warpgroup_id == 1) {
    if (tb::warp_id_in_wg() == 0) {
      for (uint32_t for_idx = 0; for_idx < 64; for_idx++) {
        STensor20000012InputAtom::run(tma_10000003, stensor20000012_ptr,  -1, 0, -1, for_idx, hopper_async_pipeline_20000012);
        STensor20000013InputAtom::run(tma_10000004, stensor20000013_ptr,  1, -1, -1, for_idx, hopper_async_pipeline_20000013);
      }
    }
  }
  else {
    // Consumer main loop
    for (uint32_t for_idx = 0; for_idx < 64; for_idx++) {
      {
        // OP type: tb_matmul_op
        int read_idx_20000012 = hopper_async_pipeline_20000012.consumer_wait();
        int read_idx_20000013 = hopper_async_pipeline_20000013.consumer_wait();
        Matmul20000015Kernel::run(matmul_20000015_accum, stensor20000012_ptr, stensor20000013_ptr, (char*)(buf+0), thread_idx, read_idx_20000012);
        tb::wg_sync<CONSUMER_NUM_THREADS>(8);
      }
      hopper_async_pipeline_20000012.consumer_release();
      hopper_async_pipeline_20000013.consumer_release();
    }
    // Write back in-register accumulators
    tb::wg_sync<CONSUMER_NUM_THREADS>(8);
    Matmul20000015Kernel::write_back_mma_rC(stensor20000015_ptr, matmul_20000015_accum, thread_idx);
    // The epilogue (kernels outside the loop)
    tb::wg_sync<CONSUMER_NUM_THREADS>(8);
    {
      // OP type: tb_output_op
      STensor20000015OutputAtom::run(dtensor10000005_tile_ptr, stensor20000015_ptr, thread_idx);
    }
  }
}




 void gemm_kernel_bf16(std::vector<void const *> input_tensors, std::vector<void*> output_tensors){
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  
  {
    // OP type: kn_customized_op
    bfloat16_t *dtensor10000005 = (bfloat16_t*)output_tensors.at(0);
    bfloat16_t *dtensor10000003 = (bfloat16_t*)input_tensors.at(0);
    bfloat16_t *dtensor10000004 = (bfloat16_t*)input_tensors.at(1);
    dim3 grid_dim(64, 1, 1);
    dim3 block_dim(256, 1, 1);
    size_t smem_size = 49344;
    
    // define tmas
    std::vector<bool> minputs = {true, false};
    static constexpr cute::GMMA::Major GmmaMajor_10000003 = GMMA::Major::K;
    using DstMNKLayout_10000003 = decltype(composition(Swizzle<3, 4, 3>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
    using SrcMNKLayout_10000003 = Layout<Shape<Int<128>, Int<64>, Int<1>, Int<1>, Int<1>, Int<64>>, Stride<Int<4096>, Int<1>, Int<1>, Int<4096>, Int<1>, Int<64>>>;
    using SmemLayoutAtom_10000003 = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GmmaMajor_10000003, bfloat16_t, decltype(get<0>(DstMNKLayout_10000003{})), decltype(get<1>(DstMNKLayout_10000003{}))>());
    using DstPipeLayout_10000003 = decltype(tile_to_shape(SmemLayoutAtom_10000003{}, make_shape(shape<0>(DstMNKLayout_10000003{}), shape<1>(DstMNKLayout_10000003{}), Int<2>{}), Step<_1, _2, _3>{}));
    auto g_tensor_10000003 = make_tensor(make_gmem_ptr<bfloat16_t>(dtensor10000003), SrcMNKLayout_10000003{});
    auto tma_10000003 = make_tma_copy(SM90_TMA_LOAD{}, g_tensor_10000003, DstPipeLayout_10000003{}(_, _, Int<0>{}));
    
    static constexpr cute::GMMA::Major GmmaMajor_10000004 = GMMA::Major::MN;
    using DstMNKLayout_10000004 = decltype(composition(Swizzle<3, 4, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
    using SrcMNKLayout_10000004 = Layout<Shape<Int<64>, Int<64>, Int<64>, Int<1>, Int<1>, Int<64>>, Stride<Int<1>, Int<4096>, Int<64>, Int<1>, Int<1>, Int<262144>>>;
    using SmemLayoutAtom_10000004 = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GmmaMajor_10000004, bfloat16_t, decltype(get<0>(DstMNKLayout_10000004{})), decltype(get<1>(DstMNKLayout_10000004{}))>());
    using DstPipeLayout_10000004 = decltype(tile_to_shape(SmemLayoutAtom_10000004{}, make_shape(shape<0>(DstMNKLayout_10000004{}), shape<1>(DstMNKLayout_10000004{}), Int<2>{}), Step<_1, _2, _3>{}));
    auto g_tensor_10000004 = make_tensor(make_gmem_ptr<bfloat16_t>(dtensor10000004), SrcMNKLayout_10000004{});
    auto tma_10000004 = make_tma_copy(SM90_TMA_LOAD{}, g_tensor_10000004, DstPipeLayout_10000004{}(_, _, Int<0>{}));
    
    cudaFuncSetAttribute(custom_kernel_0<decltype(tma_10000003), decltype(tma_10000004)>, cudaFuncAttributeMaxDynamicSharedMemorySize, 49344);
    custom_kernel_0<<<grid_dim, block_dim, smem_size>>>(tma_10000003, tma_10000004,  dtensor10000005, dtensor10000003, dtensor10000004);
    cudaDeviceSynchronize();
  }
  {
    // OP type: kn_output_op
  }
}