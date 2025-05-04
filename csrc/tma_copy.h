//reuse some code from https://github.com/mirage-project/mirage.git
#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include <cstdint>
#include <cute/layout.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include "utils.h"
using namespace cute;

namespace tb {
template <typename T,
          class DstLayout,
          class SrcLayout,
          class TMA,
          // class MainloopPipeline,
          // class PipelineState,
          class HopperAsyncPipeline,
          bool MInput,
          int K_ITER>
class TMA_COPY {
public:
  using CTA_TILER =
      decltype(make_shape(shape<0>(DstLayout{}), shape<1>(DstLayout{})));

  static constexpr cute::GMMA::Major GmmaMajor = GMMA::Major::MN;
  using DstMNKLayout = DstLayout;
  using SmemLayoutAtom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GmmaMajor,
               half_t,
               decltype(get<0>(DstMNKLayout{})),
               decltype(get<1>(DstMNKLayout{}))>());

  // A, B, X, Y, Z, Stage
  using DstPipeLayout =
      decltype(tile_to_shape(SmemLayoutAtom{},
                             make_shape(shape<0>(DstMNKLayout{}),
                                        shape<1>(DstMNKLayout{}),
                                        Int<HopperAsyncPipeline::Stage>{})));

  static constexpr int tmaTransactionBytes =
      sizeof(T) * size(DstPipeLayout{}) / HopperAsyncPipeline::Stage;

  static __device__ __forceinline__ void prefetch(TMA const &tma) {
    cute::prefetch_tma_descriptor(tma.get_tma_descriptor());
  }

  template <int SrcLayoutSize>
  static __device__ auto
      make_coord_runtime(int imapx_a, int imapy_a, int imapz_a) {
    if constexpr (SrcLayoutSize == 6) {
      return make_coord(_,
                        _,
                        imapx_a >= 0 ? blockIdx.x : 0,
                        imapy_a >= 0 ? blockIdx.y : 0,
                        imapz_a >= 0 ? blockIdx.z : 0,
                        _);
    } else if constexpr (SrcLayoutSize == 7) {
      return make_coord(_,
                        _,
                        _,
                        imapx_a >= 0 ? blockIdx.x : 0,
                        imapy_a >= 0 ? blockIdx.y : 0,
                        imapz_a >= 0 ? blockIdx.z : 0,
                        _);
    } else {
      static_assert(SrcLayoutSize == 6 || SrcLayoutSize == 7,
                    "Unsupported layout size");
    }
  }

  static __device__ __forceinline__ void run(TMA const &tma_a,
                                             T *dst_a,
                                             int imapx_a,
                                             int imapy_a,
                                             int imapz_a,
                                             int k_tile_iter,
                                             HopperAsyncPipeline &pipeline) {
    if (tb::lane_id() == 0) {
      Tensor mA = tma_a.get_tma_tensor(shape(SrcLayout{}));
      // （CTA_M, CTA_K, X, Y, Z, FORLOOP）
      auto blkCoordA = make_coord_runtime<decltype(rank(SrcLayout{}))::value>(
          imapx_a, imapy_a, imapz_a);
      // auto blkCoordA = make_coord(_,
      //                             _,
      //                             imapx_a >= 0 ? blockIdx.x : 0,
      //                             imapy_a >= 0 ? blockIdx.y : 0,
      //                             imapz_a >= 0 ? blockIdx.z : 0,
      //                             _);

      Tensor gA = mA(blkCoordA);

      Tensor sA = make_tensor(make_smem_ptr(dst_a), DstPipeLayout{});
      auto cta_tma_a = tma_a.get_slice(Int<0>{}); // CTA slice

      Tensor tAgA = cta_tma_a.partition_S(gA);
      Tensor tAsA = cta_tma_a.partition_D(sA);
      Tensor tAgAX = group_modes<0, rank(tAgA) - 1>(tAgA); // REST, Forloop
      Tensor tAsAX = group_modes<0, rank(tAsA) - 1>(tAsA);

      auto [tma_barrier, write_stage] = pipeline.producer_acquire();
      copy(tma_a.with(*tma_barrier),
           tAgAX(_, k_tile_iter),
           tAsAX(_, write_stage));
      pipeline.producer_advance();
    }
  }
};



template <typename T,
          class DstLayout,
          class SrcLayout,
          class TMA,
          // class MainloopPipeline,
          // class PipelineState,
          class HopperAsyncPipeline,
          bool MInput,
          int K_ITER>
class TMA_COPY_FP8 {
public:
  using CTA_TILER =
      decltype(make_shape(shape<0>(DstLayout{}), shape<1>(DstLayout{})));

  static constexpr cute::GMMA::Major GmmaMajor = GMMA::Major::MN;
  using DstMNKLayout = DstLayout;
  using SmemLayoutAtom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GmmaMajor,
               half_t,
               decltype(get<0>(DstMNKLayout{})),
               decltype(get<1>(DstMNKLayout{}))>());

  // A, B, X, Y, Z, Stage
  using DstPipeLayout =
      decltype(tile_to_shape(SmemLayoutAtom{},
                             make_shape(shape<0>(DstMNKLayout{}),
                                        shape<1>(DstMNKLayout{}),
                                        Int<HopperAsyncPipeline::Stage>{})));

  static constexpr int tmaTransactionBytes =
      sizeof(T) * size(DstPipeLayout{}) / HopperAsyncPipeline::Stage;

  static __device__ __forceinline__ void prefetch(TMA const &tma) {
    cute::prefetch_tma_descriptor(tma.get_tma_descriptor());
  }

  template <int SrcLayoutSize>
  static __device__ auto
      make_coord_runtime(int imapx_a, int imapy_a, int imapz_a) {
    if constexpr (SrcLayoutSize == 6) {
      return make_coord(_,
                        _,
                        imapx_a >= 0 ? blockIdx.x : 0,
                        imapy_a >= 0 ? blockIdx.y : 0,
                        imapz_a >= 0 ? blockIdx.z : 0,
                        _);
    } else if constexpr (SrcLayoutSize == 7) {
      return make_coord(_,
                        _,
                        _,
                        imapx_a >= 0 ? blockIdx.x : 0,
                        imapy_a >= 0 ? blockIdx.y : 0,
                        imapz_a >= 0 ? blockIdx.z : 0,
                        _);
    } else {
      static_assert(SrcLayoutSize == 6 || SrcLayoutSize == 7,
                    "Unsupported layout size");
    }
  }

  static __device__ __forceinline__ void run(TMA const &tma_a,
                                             T *dst_a,
                                             int imapx_a,
                                             int imapy_a,
                                             int imapz_a,
                                             int k_tile_iter,
                                             HopperAsyncPipeline &pipeline) {
    if (tb::lane_id() == 0) {
      Tensor mA = tma_a.get_tma_tensor(shape(SrcLayout{}));
      // （CTA_M, CTA_K, X, Y, Z, FORLOOP）
      auto blkCoordA = make_coord_runtime<decltype(rank(SrcLayout{}))::value>(
          imapx_a, imapy_a, imapz_a);
      // auto blkCoordA = make_coord(_,
      //                             _,
      //                             imapx_a >= 0 ? blockIdx.x : 0,
      //                             imapy_a >= 0 ? blockIdx.y : 0,
      //                             imapz_a >= 0 ? blockIdx.z : 0,
      //                             _);

      Tensor gA = mA(blkCoordA);

      Tensor sA = make_tensor(make_smem_ptr(dst_a), DstLayout{});
      auto cta_tma_a = tma_a.get_slice(Int<0>{}); // CTA slice

      Tensor tAgA = cta_tma_a.partition_S(gA);
      Tensor tAsA = cta_tma_a.partition_D(sA);
      Tensor tAgAX = group_modes<0, rank(tAgA) - 1>(tAgA); // REST, Forloop
      Tensor tAsAX = group_modes<0, rank(tAsA) - 1>(tAsA);

      auto [tma_barrier, write_stage] = pipeline.producer_acquire();
      copy(tma_a.with(*tma_barrier),
           tAgAX(_, k_tile_iter),
           tAsAX(_, write_stage));
      pipeline.producer_advance();
    }
  }
};
}