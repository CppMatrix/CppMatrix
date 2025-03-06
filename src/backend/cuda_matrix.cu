#include <cassert>
#include <cuda_fp16.h>
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
__global__ void vectorAdd(T a, const T* b, T* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = a + b[i];
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

template <typename T>
__global__ void vectorSub(T a, const T* b, T* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = a - b[i];
    }
}

template <typename T>
__global__ void vectorProduct(const T* a, const T* b, T* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = a[i] * b[i];
    }
}

template <typename T>
__global__ void vectorProduct(const T* a, T b, T* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = a[i] * b;
    }
}

template <typename T>
__global__ void vectorProduct(T a, const T* b, T* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = a * b[i];
    }
}

template <typename T>
__global__ void matrixSumByRow(const T* in, T* out, size_t row, size_t column)
{
    auto rowIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (rowIndex < row) {
        auto sum = T {};
        in += rowIndex * column;
        for (auto c = 0u; c < column; ++c) {
            sum += *in++;
        }
        out[rowIndex] = sum;
    }
}

template <typename T>
__global__ void matrixSumByColumn(const T* in, T* out, size_t row, size_t column)
{
    auto columnIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (columnIndex < column) {
        auto sum = T {};
        in += columnIndex;
        for (auto r = 0u; r < row; ++r) {
            sum += *in;
            in += column;
        }
        out[columnIndex] = sum;
    }
}

template <typename T>
__global__ void matrixDotMul(const T* a, const T* b, T* out, size_t aRow, size_t aColumn, size_t bColumn)
{
    auto n = blockDim.x * blockIdx.x + threadIdx.x;
    auto row = n / bColumn;
    auto column = n % bColumn;
    if (row < aRow) {
        auto sum = T {};
        for (auto i = 0; i < aColumn; ++i) {
            sum += a[row * aColumn + i] * b[i * bColumn + column];
        }
        out[row * bColumn + column] = sum;
    }
}

__global__ void matrixSigmoid(const half* in, half* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = (half)1.0 / ((half)1.0 + hexp(-in[i]));
    }
}

__global__ void matrixSigmoid(const float* in, float* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = 1 / (1 + expf(-in[i]));
    }
}

__global__ void matrixRelu(const half* in, half* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = (in[i] + __habs(in[i])) / (half)2;
    }
}

__global__ void matrixRelu(const float* in, float* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = (in[i] + fabsf(in[i])) / 2;
    }
}

template <typename T>
__global__ void matrixTranspose(const T* a, T* out, size_t aRow, size_t aColumn)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    auto row = i / aColumn;
    auto column = i % aColumn;
    if (row < aRow) {
        out[column * aRow + row] = a[row * aColumn + column];
    }
}

__global__ void vectorPow(const float* a, float e, float* out, size_t numElements)
{
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        out[i] = powf(a[i], e);
    }
}

template <typename TA, typename TB, typename TOut>
cudaError_t CudaBinaryOp(const TA a, const TB b, TOut out, size_t numElements, char op)
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
    case '*':
        vectorProduct<<<blocksPerGrid, threadsPerBlock>>>(a, b, out, numElements);
        break;
    default:
        assert(false);
        throw std::runtime_error { "Unsupported op" };
    }
    return cudaGetLastError();
}

enum class UnaryOp {
    Sigmoid,
    Relu,
};

template <typename T>
cudaError_t CudaUnaryOp(const T* in, T* out, size_t numElements, UnaryOp op)
{
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    switch (op) {
    case UnaryOp::Sigmoid:
        matrixSigmoid<<<blocksPerGrid, threadsPerBlock>>>(in, out, numElements);
        break;
    case UnaryOp::Relu:
        matrixRelu<<<blocksPerGrid, threadsPerBlock>>>(in, out, numElements);
        break;
    default:
        assert(false);
        throw std::runtime_error { "Unsupported op" };
    }
    return cudaGetLastError();
}

template <typename T>
cudaError_t CudaMatrixDotMul(const T* a, const T* b, T* c, size_t aRow, size_t aColumn, size_t bColumn)
{
    size_t n = aRow * bColumn;
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    matrixDotMul<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, aRow, aColumn, bColumn);
    return cudaGetLastError();
}

template <typename T>
cudaError_t CudaVectorPow(const T* in, T e, T* out, size_t numElements)
{
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorPow<<<blocksPerGrid, threadsPerBlock>>>(in, e, out, numElements);
    return cudaGetLastError();
}

template <typename T>
cudaError_t CudaMatrixSum(const T* in, T* out, bool byRow, size_t row, size_t column)
{
    auto group = byRow ? row : column;
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (group + threadsPerBlock - 1) / threadsPerBlock;
    if (byRow) {
        matrixSumByRow<<<blocksPerGrid, threadsPerBlock>>>(in, out, row, column);
    } else {
        matrixSumByColumn<<<blocksPerGrid, threadsPerBlock>>>(in, out, row, column);
    }
    return cudaGetLastError();
}

template <typename T>
cudaError_t CudaMatrixTranspose(const T* in, T* out, size_t inRow, size_t inColumn)
{
    size_t n = inRow * inColumn;
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(in, out, inRow, inColumn);
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
DefineCudaBinaryFunc(CudaSub, '-', std::float16_t, const std::float16_t*, std::float16_t*);
DefineCudaBinaryFunc(CudaSub, '-', std::float32_t, const std::float32_t*, std::float32_t*);

DefineCudaBinaryFunc(CudaProduct, '*', const std::float16_t*, const std::float16_t*, std::float16_t*);
DefineCudaBinaryFunc(CudaProduct, '*', const std::float32_t*, const std::float32_t*, std::float32_t*);
DefineCudaBinaryFunc(CudaProduct, '*', std::float16_t, const std::float16_t*, std::float16_t*);
DefineCudaBinaryFunc(CudaProduct, '*', std::float32_t, const std::float32_t*, std::float32_t*);

#define DefineCudaDotProductFunc(Name, Type)                                                                           \
    cudaError_t Name(const Type* cudaBufferA, const Type* cudaBufferB, Type* cudaBufferOut, size_t aRow,               \
        size_t aColumn, size_t bColumn)                                                                                \
    {                                                                                                                  \
        return CudaMatrixDotMul(CudaUnderlineType(cudaBufferA), CudaUnderlineType(cudaBufferB),                        \
            CudaUnderlineType(cudaBufferOut), aRow, aColumn, bColumn);                                                 \
    }

DefineCudaDotProductFunc(CudaDotProduct, std::float16_t);
DefineCudaDotProductFunc(CudaDotProduct, std::float32_t);

#define DefineCudaUnaryFunc(Name, Op, Type)                                                                            \
    cudaError_t Name(const Type* cudaBufferIn, Type* cudaBufferOut, size_t numElements)                                \
    {                                                                                                                  \
        return CudaUnaryOp(CudaUnderlineType(cudaBufferIn), CudaUnderlineType(cudaBufferOut), numElements, Op);        \
    }

DefineCudaUnaryFunc(CudaSigmoid, UnaryOp::Sigmoid, std::float16_t);
DefineCudaUnaryFunc(CudaSigmoid, UnaryOp::Sigmoid, std::float32_t);

DefineCudaUnaryFunc(CudaRelu, UnaryOp::Relu, std::float16_t);
DefineCudaUnaryFunc(CudaRelu, UnaryOp::Relu, std::float32_t);

#define DefineCudaTransposeFunc(Name, Type)                                                                            \
    cudaError_t Name(const Type* cudaBufferIn, Type* cudaBufferOut, size_t inRow, size_t inColumn)                     \
    {                                                                                                                  \
        return CudaMatrixTranspose(                                                                                    \
            CudaUnderlineType(cudaBufferIn), CudaUnderlineType(cudaBufferOut), inRow, inColumn);                       \
    }

DefineCudaTransposeFunc(CudaTranspose, std::float16_t);
DefineCudaTransposeFunc(CudaTranspose, std::float32_t);

#define DefineCudaPowFunc(Name, Type)                                                                                  \
    cudaError_t Name(const Type* cudaBufferIn, Type e, Type* cudaBufferOut, size_t numElements)                        \
    {                                                                                                                  \
        return CudaVectorPow(                                                                                          \
            CudaUnderlineType(cudaBufferIn), CudaUnderlineType(e), CudaUnderlineType(cudaBufferOut), numElements);     \
    }

DefineCudaPowFunc(CudaPow, std::float32_t);

#define DefineCudaSumFunc(Name, Type)                                                                                  \
    cudaError_t Name(const Type* cudaBufferIn, Type* cudaBufferOut, bool byRow, size_t row, size_t column)             \
    {                                                                                                                  \
        return CudaMatrixSum(CudaUnderlineType(cudaBufferIn), CudaUnderlineType(cudaBufferOut), byRow, row, column);   \
    }

DefineCudaSumFunc(CudaSum, std::float16_t);
DefineCudaSumFunc(CudaSum, std::float32_t);

}