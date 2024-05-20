#include "cuda_runtime.h"
#include <stdint.h>

typedef struct {
    uint16_t n;
} __half;
void launch_simple_gemm_tt_half(size_t m, size_t n, size_t k,
                                            __half const* alpha,
                                            __half const* A, size_t lda,
                                            __half const* B, size_t ldb,
                                            __half const* beta, __half* C,
                                            size_t ldc, cudaStream_t stream);
