#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdfloat>

namespace cpp_matrix::backend {

template <typename T>
__global__ void vectorAdd(const T* a, const T* b, T* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = a[i] + b[i];
    }
}

template <typename T>
__global__ void vectorAdd(const T* a, T b, T* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = a[i] + b;
    }
}

template <typename T, typename R>
cudaError_t DoCudaAdd(const T* a, R b, T* out, size_t numElements)
{
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, out, numElements);
    return cudaGetLastError();
}

half* CudaUnderlineType(std::float16_t* t)
{
    return (half*)t;
}

const half* CudaUnderlineType(const std::float16_t* t)
{
    return (const half*)t;
}

half CudaUnderlineType(std::float16_t t)
{
    return *(half*)&t;
}

float* CudaUnderlineType(std::float32_t* t)
{
    return (float*)t;
}

const float* CudaUnderlineType(const std::float32_t* t)
{
    return (const float*)t;
}

float CudaUnderlineType(std::float32_t t)
{
    return t;
}

#define DefineCudaFunc(Name, TypeA, TypeB, TypeOut)                                                                    \
    cudaError_t Name(TypeA cudaBufferA, TypeB cudaBufferB, TypeOut cudaBufferOut, size_t numElements)                  \
    {                                                                                                                  \
        return Do##Name(CudaUnderlineType(cudaBufferA), CudaUnderlineType(cudaBufferB),                                \
            CudaUnderlineType(cudaBufferOut), numElements);                                                            \
    }

DefineCudaFunc(CudaAdd, const std::float16_t*, const std::float16_t*, std::float16_t*);
DefineCudaFunc(CudaAdd, const std::float16_t*, std::float16_t, std::float16_t*);
DefineCudaFunc(CudaAdd, const std::float32_t*, const std::float32_t*, std::float32_t*);
DefineCudaFunc(CudaAdd, const std::float32_t*, std::float32_t, std::float32_t*);

}