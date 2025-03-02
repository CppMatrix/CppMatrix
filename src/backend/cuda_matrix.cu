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
cudaError_t DoCudaAdd(const T* cudaBufferA, const T* cudaBufferB, T* cudaBufferOut, size_t numElements)
{
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(cudaBufferA, cudaBufferB, cudaBufferOut, numElements);
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

float* CudaUnderlineType(std::float32_t* t)
{
    return (float*)t;
}

const float* CudaUnderlineType(const std::float32_t* t)
{
    return (const float*)t;
}

#define DefineCudaFunc(Name, Type)                                                                                     \
    cudaError_t Name(const Type* cudaBufferA, const Type* cudaBufferB, Type* cudaBufferOut, size_t numElements)        \
    {                                                                                                                  \
        return Do##Name(CudaUnderlineType(cudaBufferA), CudaUnderlineType(cudaBufferB),                                \
            CudaUnderlineType(cudaBufferOut), numElements);                                                            \
    }

DefineCudaFunc(CudaAdd, std::float16_t);
DefineCudaFunc(CudaAdd, std::float32_t);

}