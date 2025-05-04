//reuse some code from https://github.com/mirage-project/mirage.git
namespace tb {

    static constexpr int NUM_THREADS_PER_WARP = 32;
    static constexpr  int NUM_WARPS_PER_GROUP = 4;
    static constexpr int NUM_THREADS_PER_GROUP = NUM_WARPS_PER_GROUP * NUM_THREADS_PER_WARP;
static __device__ __forceinline__ int lane_id() {
    return threadIdx.x & 0x1f;
  }
  
  static __device__ __forceinline__ int warp_id_in_wg() {
    return __shfl_sync(0xffffffff,
                       (threadIdx.x / NUM_THREADS_PER_WARP) %
                           NUM_WARPS_PER_GROUP,
                       0);
  }
  
  static __device__ __forceinline__ int warp_id() {
    return __shfl_sync(
        0xffffffff, threadIdx.x / NUM_THREADS_PER_WARP, 0);
  }
  
  static __device__ __forceinline__ int warpgroup_id() {
    return __shfl_sync(
        0xffffffff, threadIdx.x / NUM_THREADS_PER_GROUP, 0);
  }


template <typename T>
static __device__ __forceinline__ T
    perform_exp_op(T a, float scalar = 0.0f) {
  if constexpr (!(std::is_same_v<T, cutlass::half_t> ||
                  std::is_same_v<T, cutlass::bfloat16_t> ||
                  std::is_same_v<T, float> || std::is_same_v<T, __half>)) {
    assert(0 && "unsupport datatype in tb elementunary");
  }
  return (T)expf((float)a);

  return (T)0.0;
}

template <int GROUP_THREADS>
static __device__ __forceinline__ void wg_sync(uint32_t barrier_id) {
#ifdef USE_GRACE_HOPPER
  asm volatile("bar.sync %0, %1;\n" ::"r"(barrier_id), "n"(GROUP_THREADS));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

}