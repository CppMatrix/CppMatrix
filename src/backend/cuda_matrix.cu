#include <cassert>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
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

template <typename T>
__global__ void vectorSub(const T* a, const T* b, T* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = a[i] - b[i];
    }
}

template <typename T>
__global__ void vectorSub(const T* a, T b, T* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = a[i] - b;
    }
}

template <typename T, typename R>
cudaError_t CudaBinaryOp(const T* a, R b, T* out, size_t numElements, char op)
{
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    switch (op) {
    case '+':
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, out, numElements);
        break;
    case '-':
        vectorSub<<<blocksPerGrid, threadsPerBlock>>>(a, b, out, numElements);
        break;

    default:
        assert(false);
        throw std::runtime_error { "Unsupported op" };
    }
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

#define DefineCudaBinaryFunc(Name, Op, TypeA, TypeB, TypeOut)                                                          \
    cudaError_t Name(TypeA cudaBufferA, TypeB cudaBufferB, TypeOut cudaBufferOut, size_t numElements)                  \
    {                                                                                                                  \
        return CudaBinaryOp(CudaUnderlineType(cudaBufferA), CudaUnderlineType(cudaBufferB),                            \
            CudaUnderlineType(cudaBufferOut), numElements, Op);                                                        \
    }

DefineCudaBinaryFunc(CudaAdd, '+', const std::float16_t*, const std::float16_t*, std::float16_t*);
DefineCudaBinaryFunc(CudaAdd, '+', const std::float16_t*, std::float16_t, std::float16_t*);
DefineCudaBinaryFunc(CudaAdd, '+', const std::float32_t*, const std::float32_t*, std::float32_t*);
DefineCudaBinaryFunc(CudaAdd, '+', const std::float32_t*, std::float32_t, std::float32_t*);

DefineCudaBinaryFunc(CudaSub, '-', const std::float16_t*, const std::float16_t*, std::float16_t*);
DefineCudaBinaryFunc(CudaSub, '-', const std::float32_t*, const std::float32_t*, std::float32_t*);

}