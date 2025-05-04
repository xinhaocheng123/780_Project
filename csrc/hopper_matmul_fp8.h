//reuse some code from https://github.com/mirage-project/mirage.git
#pragma once

#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include "copy_sm80.h"
#include "tma_copy.h"
#include "cutlass/gemm/collective/fp8_accumulation.hpp"
using namespace cute;

namespace tb {

    template <class InputLayout>
    class Dim01Swapper {
      CUTE_STATIC_ASSERT_V(rank(InputLayout{}) == _2{});
    
      using A0 = decltype(get<0>(shape(InputLayout{})));
      using A1 = decltype(get<1>(shape(InputLayout{})));
      using TransposeCoordLayout = Layout<Shape<A1, A0>, Stride<A0, _1>>;
      using Result_ = decltype(composition(InputLayout{}, TransposeCoordLayout{}));
    
    public:
      using Result =
          decltype(coalesce(Result_{}, Step<_1, _1>{})); // By-mode coalescing
    };
    enum class R2STiledCopyType { UNIVERSAL, STMATRIX_N, STMATRIX_T };
    // Select the R2S (register -> shared) copy atom
    template <typename T, bool IS_STMATRIX_AVAIL, class Layout>
    class R2STiledCopySelector {
      static_assert(!IS_STMATRIX_AVAIL);
    
    public:
      using Result = Copy_Atom<UniversalCopy<uint16_t>, T>;
      static constexpr R2STiledCopyType TYPE = R2STiledCopyType::UNIVERSAL;
    };
    template <class T,
    class M,
    class N,
    int NUM_EXPS_BEFORE_STORE,
    bool IS_STORE_ACCUM,
    class TiledCopy,
    class SrcEngine,
    class SrcLayout,
    class DstEngine,
    class DstLayout>
CUTE_HOST_DEVICE void r2s_copy_with_oob_protection(
TiledCopy const &tiled_copy,
Tensor<SrcEngine, SrcLayout> const &src, // [R2S, R2S_M, R2S_N]
Tensor<DstEngine, DstLayout> &dst,       // The same as src
int thread_idx) {
static_assert(SrcLayout::rank == 3);
static_assert(DstLayout::rank == 3);

using TiledMN = typename TiledCopy::Tiler_MN;
using TileM = decltype(get<0>(TiledMN{}));
using TileN = decltype(get<1>(TiledMN{}));
if constexpr ((M::value % TileM::value == 0) &&
          (N::value % TileN::value == 0)) {
CUTE_UNROLL
for (int i = 0; i < size(src); ++i) {
float x = src(i);
if constexpr (NUM_EXPS_BEFORE_STORE > 0) {
  CUTE_UNROLL
  for (int i = 0; i < NUM_EXPS_BEFORE_STORE; ++i) {
    x = perform_exp_op<float>(x);
  }
}
if constexpr (IS_STORE_ACCUM) {
  dst(i) += T(x);
} else {
  dst(i) = T(x);
}
}
} else {
using MIndicatorLayout = Layout<Shape<M, N>, Stride<_1, _0>>;
using NIndicatorLayout = Layout<Shape<M, N>, Stride<_0, _1>>;
auto m_indicator_thrIdx_r2s_r2sM_r2sN =
  tiled_copy.tidfrg_D(MIndicatorLayout{});
auto n_indicator_thrIdx_r2s_r2sM_r2sN =
  tiled_copy.tidfrg_D(NIndicatorLayout{});
static_assert(is_static_v<decltype(m_indicator_thrIdx_r2s_r2sM_r2sN)>);
static_assert(is_static_v<decltype(n_indicator_thrIdx_r2s_r2sM_r2sN)>);
int offset_m = m_indicator_thrIdx_r2s_r2sM_r2sN(thread_idx, _0{}, _0{});
int offset_n = n_indicator_thrIdx_r2s_r2sM_r2sN(thread_idx, _0{}, _0{});
auto m_indicator_frag = m_indicator_thrIdx_r2s_r2sM_r2sN(
  thread_idx, _, make_tuple(_, _)); // [R2S, R2S_M, R2S_N]
auto n_indicator_frag = n_indicator_thrIdx_r2s_r2sM_r2sN(
  thread_idx, _, make_tuple(_, _)); // Same as above

CUTE_UNROLL
for (int i = 0; i < size(src); ++i) {
auto coord_m = offset_m + m_indicator_frag(i);
auto coord_n = offset_n + n_indicator_frag(i);
bool valid = coord_m < M{} && coord_n < N{};
if (valid) {
  float x = src(i);
  if constexpr (NUM_EXPS_BEFORE_STORE > 0) {
    CUTE_UNROLL
    for (int i = 0; i < NUM_EXPS_BEFORE_STORE; ++i) {
      x = perform_exp_op<float>(x);
    }
  }
  if constexpr (IS_STORE_ACCUM) {
    dst(i) += T(x);
  } else {
    dst(i) = T(x);
  }
}
}
}
}

template <typename T,
typename T_ACCUM,
          bool IS_LDMATRIX_AVAIL,
          bool IS_STMATRIX_AVAIL,
          class SmemLayoutA_, // [K, M]
          class SmemLayoutB_, // [N, K]
          class SmemLayoutC_, // [N, M]
          int NUM_THREADS,
          int NUM_EXPS_BEFORE_STORE, // Since matmul may use some advanced
                                     // instructions (like stmatrix) to store
                                     // data, it does not use the standard
                                     // "epilogue" semantic
          bool IS_STORE_ACCUM,
          bool IS_COORPERATIVE,
          bool IS_PIPELINE_A,
          bool IS_PIPELINE_B,
          int PIPELINE_STAGES>
class Hopper_Matmul_FP8 {
public:
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutA_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutB_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutC_{}) == _2{});

  using SmemLayoutA = typename Dim01Swapper<SmemLayoutA_>::Result; // [M, K]
  using SmemLayoutB = SmemLayoutB_;                                // [N, K]
  using SmemLayoutC = typename Dim01Swapper<SmemLayoutC_>::Result; // [M, N]

  // NT	M-major	(M,K):(1,ldA)	N-major	(N,K):(1,ldB)
  // TN	K-major	(M,K):(ldA,1)	K-major	(N,K):(ldB,1)
  // NN	M-major	(M,K):(1,ldA)	K-major	(N,K):(ldB,1)
  // TT	K-major	(M,K):(ldA,1)	N-major	(N,K):(1,ldB)

  static constexpr GMMA::Major GmmaMajorA = GMMA::Major::K;
  static constexpr GMMA::Major GmmaMajorB = GMMA::Major::K;
  using TileALayout =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GmmaMajorA,
               T,
               decltype(get<0>(SmemLayoutA{})),
               decltype(get<1>(SmemLayoutA{}))>());
  using TileBLayout =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GmmaMajorB,
               T,
               decltype(get<0>(SmemLayoutB{})),
               decltype(get<1>(SmemLayoutB{}))>());
  // Shape checking
  // Expect A have a shape of [M, K], B have a shape of [N, K], and
  // C have a shape of [M, N]
  using M = decltype(get<0>(shape(SmemLayoutA{})));
  using K = decltype(get<1>(shape(SmemLayoutA{})));
  using N = decltype(get<0>(shape(SmemLayoutB{})));
  CUTE_STATIC_ASSERT_V(K{} == get<1>(shape(SmemLayoutB{})));
  CUTE_STATIC_ASSERT_V(M{} == get<0>(shape(SmemLayoutC{})));
  CUTE_STATIC_ASSERT_V(N{} == get<1>(shape(SmemLayoutC{})));

  static constexpr int PIPELINE_STAGE_A = IS_PIPELINE_A ? PIPELINE_STAGES : 1;
  static constexpr int PIPELINE_STAGE_B = IS_PIPELINE_B ? PIPELINE_STAGES : 1;

  using AtomLayoutMNK = cute::conditional_t<IS_COORPERATIVE,
                                            Layout<Shape<_2, _1, _1>>,
                                            Layout<Shape<_1, _1, _1>>>;

  using TiledMMA = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<
          T,
          T,
          T_ACCUM,
          decltype(make_shape(
              cute::Int<(M::value < 64 ? 64 : M::value)>{}, N{}, K{})),
          GmmaMajorA,
          GmmaMajorB>(),
      AtomLayoutMNK{}));

  static constexpr int TILED_MMA_NUM_THREADS = thr_size(TiledMMA{});
  static_assert(TILED_MMA_NUM_THREADS ==
                NUM_THREADS * (IS_COORPERATIVE ? 2 : 1));

  using R2STiledCopyCSelector =
      R2STiledCopySelector<T, IS_STMATRIX_AVAIL, SmemLayoutC>;
  using R2STiledCopyCAtom = typename R2STiledCopyCSelector::Result;
  static constexpr R2STiledCopyType R2S_TILED_COPY_C_TYPE =
      R2STiledCopyCSelector::TYPE;
  using R2STiledCopyC =
      decltype(make_tiled_copy_C(R2STiledCopyCAtom{}, TiledMMA{}));

  static __device__ __forceinline__ auto get_mma_rC(int thread_idx) {
    // Make a fake tensor

    Tensor sC_fake = make_tensor(make_smem_ptr((T_ACCUM *)nullptr), SmemLayoutC{});

    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(thread_idx % 128);
    Tensor mma_rC =
        thr_mma.partition_fragment_C(sC_fake); // (MMA, MMA_M, MMA_N)

    clear(mma_rC);
    return mma_rC;
  }


  

  template <class AccumRegFrag>
  static __device__ __forceinline__ void write_back_mma_rC(
    T_ACCUM *__restrict__ c_ptr, AccumRegFrag const &mma_rC, int thread_idx) {
    if (thread_idx >= TILED_MMA_NUM_THREADS) {
      return;
    }

    Tensor sC = make_tensor(make_smem_ptr(c_ptr), SmemLayoutC{}); // [M, N]
    R2STiledCopyC r2s_tiled_copy_C;
    ThrCopy r2s_tiled_copy_C_thr = r2s_tiled_copy_C.get_slice(thread_idx);
    Tensor r2s_rC =
        r2s_tiled_copy_C_thr.retile_S(mma_rC);            // (R2S, R2S_M, R2S_N)
    Tensor r2s_sC = r2s_tiled_copy_C_thr.partition_D(sC); // (R2S, R2S_M, R2S_N)

    r2s_copy_with_oob_protection<T_ACCUM,
                                 M,
                                 N,
                                 NUM_EXPS_BEFORE_STORE,
                                 IS_STORE_ACCUM>(
        r2s_tiled_copy_C, r2s_rC, r2s_sC, thread_idx);
  }



  template <typename MMARc>
  static __device__ __forceinline__ void
      run(MMARc &mma_rC,
         float_e4m3_t *__restrict__ a_ptr,
          float_e4m3_t *__restrict__ b_ptr,
          float *sclae_A,
          float *scale_B,
          char const *__restrict__ smem_allzero_ptr,
          int thread_idx,
          int read_stage) {
    TiledMMA tiled_mma;
    auto sA_l = tile_to_shape(TileALayout{},
                              make_shape(shape<0>(SmemLayoutA{}),
                                         shape<1>(SmemLayoutA{}),
                                         Int<PIPELINE_STAGE_A>{}),
                              Step<_1, _2, _3>{});

    auto sB_l = tile_to_shape(TileBLayout{},
                              make_shape(shape<0>(SmemLayoutB{}),
                                         shape<1>(SmemLayoutB{}),
                                         Int<PIPELINE_STAGE_B>{}),
                              Step<_1, _2, _3>{});

    Tensor sA = make_tensor(make_smem_ptr(a_ptr), sA_l); // [M, K]
    Tensor sB = make_tensor(make_smem_ptr(b_ptr), sB_l); // [N, K]

    // Tensor gA_scale = make_tensor(make_gmem_ptr(sclae_A), sA_l); // [M, K]
    // Tensor gB_scale = make_tensor(make_gmem_ptr(scale_B), sB_l); // [N, K]


    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    // Tensor tCsAScale = thr_mma.partition_A(sA_scale); // (MMA,MMA_M,MMA_K,PIPE)
    // Tensor tCsBScale = thr_mma.partition_B(sB_scale); // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

    // Tensor tCrAScale = thr_mma.make_fragment_A(tCsAScale); // (MMA,MMA_M,MMA_K,PIPE)
    // Tensor tCrBScale = thr_mma.make_fragment_B(tCsBScale); // (MMA,MMA_N,MMA_K,PIPE)
    // int read_stage = smem_pipe_read.index();

    warpgroup_fence_operand(mma_rC);
    cute::warpgroup_arrive();
    gemm(tiled_mma,
         tCrA(_, _, _, IS_PIPELINE_A ? read_stage : 0),
         tCrB(_, _, _, IS_PIPELINE_B ? read_stage : 0),
         mma_rC);
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(mma_rC); ++i) {
        mma_rC(i) = (T_ACCUM)(sclae_A[i]/128) * (T_ACCUM)(scale_B[i]/64);
    }
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
    warpgroup_fence_operand(mma_rC);
  }
};

} // namespace tb
