module;

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <format>
#include <memory>
#include <span>
#include <stdfloat>
#include <vector>

namespace cpp_matrix::backend {

cudaError_t CudaAdd(const std::float16_t* cudaBufferA, const std::float16_t* cudaBufferB, std::float16_t* cudaBufferOut,
    size_t numElements);
cudaError_t CudaAdd(const std::float32_t* cudaBufferA, const std::float32_t* cudaBufferB, std::float32_t* cudaBufferOut,
    size_t numElements);
cudaError_t CudaAdd(
    const std::float16_t* cudaBufferA, std::float16_t cudaBufferB, std::float16_t* cudaBufferOut, size_t numElements);
cudaError_t CudaAdd(
    const std::float32_t* cudaBufferA, std::float32_t cudaBufferB, std::float32_t* cudaBufferOut, size_t numElements);

cudaError_t CudaSub(const std::float16_t* cudaBufferA, const std::float16_t* cudaBufferB, std::float16_t* cudaBufferOut,
    size_t numElements);
cudaError_t CudaSub(const std::float32_t* cudaBufferA, const std::float32_t* cudaBufferB, std::float32_t* cudaBufferOut,
    size_t numElements);
cudaError_t CudaSub(
    std::float16_t a, const std::float16_t* cudaBufferB, std::float16_t* cudaBufferOut, size_t numElements);
cudaError_t CudaSub(
    std::float32_t a, const std::float32_t* cudaBufferB, std::float32_t* cudaBufferOut, size_t numElements);

cudaError_t CudaDiv(const std::float16_t* cudaBufferA, const std::float16_t* cudaBufferB, std::float16_t* cudaBufferOut,
    size_t numElements);
cudaError_t CudaDiv(const std::float32_t* cudaBufferA, const std::float32_t* cudaBufferB, std::float32_t* cudaBufferOut,
    size_t numElements);
cudaError_t CudaDiv(
    const std::float16_t* cudaBufferA, std::float16_t cudaBufferB, std::float16_t* cudaBufferOut, size_t numElements);
cudaError_t CudaDiv(
    const std::float32_t* cudaBufferA, std::float32_t cudaBufferB, std::float32_t* cudaBufferOut, size_t numElements);
cudaError_t CudaDiv(
    std::float16_t cudaBufferA, const std::float16_t* cudaBufferB, std::float16_t* cudaBufferOut, size_t numElements);
cudaError_t CudaDiv(
    std::float32_t cudaBufferA, const std::float32_t* cudaBufferB, std::float32_t* cudaBufferOut, size_t numElements);

cudaError_t CudaProduct(const std::float16_t* cudaBufferA, const std::float16_t* cudaBufferB,
    std::float16_t* cudaBufferOut, size_t numElements);
cudaError_t CudaProduct(const std::float32_t* cudaBufferA, const std::float32_t* cudaBufferB,
    std::float32_t* cudaBufferOut, size_t numElements);
cudaError_t CudaProduct(
    std::float16_t a, const std::float16_t* cudaBufferB, std::float16_t* cudaBufferOut, size_t numElements);
cudaError_t CudaProduct(
    std::float32_t a, const std::float32_t* cudaBufferB, std::float32_t* cudaBufferOut, size_t numElements);

cudaError_t CudaDotProduct(const std::float16_t* cudaBufferA, const std::float16_t* cudaBufferB,
    std::float16_t* cudaBufferOut, size_t aRow, size_t aColumn, size_t bColumn);
cudaError_t CudaDotProduct(const std::float32_t* cudaBufferA, const std::float32_t* cudaBufferB,
    std::float32_t* cudaBufferOut, size_t aRow, size_t aColumn, size_t bColumn);

cudaError_t CudaTranspose(
    const std::float16_t* cudaBufferIn, std::float16_t* cudaBufferOut, size_t inRow, size_t inColumn);
cudaError_t CudaTranspose(
    const std::float32_t* cudaBufferIn, std::float32_t* cudaBufferOut, size_t inRow, size_t inColumn);

cudaError_t CudaRelu(const std::float16_t* cudaBufferIn, std::float16_t* cudaBufferOut, size_t numElements);
cudaError_t CudaRelu(const std::float32_t* cudaBufferIn, std::float32_t* cudaBufferOut, size_t numElements);

cudaError_t CudaExp(const std::float16_t* cudaBufferA, std::float16_t* cudaBufferOut, size_t numElements);
cudaError_t CudaExp(const std::float32_t* cudaBufferA, std::float32_t* cudaBufferOut, size_t numElements);

cudaError_t CudaPow(
    const std::float32_t* cudaBufferA, std::float32_t e, std::float32_t* cudaBufferOut, size_t numElements);

cudaError_t CudaSum(
    const std::float16_t* cudaBufferIn, std::float16_t* cudaBufferOut, bool byRow, size_t row, size_t column);
cudaError_t CudaSum(
    const std::float32_t* cudaBufferIn, std::float32_t* cudaBufferOut, bool byRow, size_t row, size_t column);

}

export module cpp_matrix:cuda_matrix;
import :matrix_type;

namespace cpp_matrix::backend {

#define Cuda(Func, ...)                                                                                                \
    do {                                                                                                               \
        auto err = Func(__VA_ARGS__);                                                                                  \
        if (err != cudaSuccess) {                                                                                      \
            throw std::runtime_error { std::format("Invoke cuda function " #Func " failed: {}", (int)err) };           \
        }                                                                                                              \
    } while (0)

template <typename T>
std::unique_ptr<T, decltype(&cudaFree)> CudaMalloc(size_t numElements)
{
    T* p {};
    if (numElements) {
        Cuda(cudaMalloc, (void**)&p, sizeof(T) * numElements);
    }
    return { p, &cudaFree };
}

export template <MatrixElementType T>
class CudaMatrix {
    template <MatrixElementType R>
    friend CudaMatrix<R> operator-(R v, const CudaMatrix<R>& m);

    template <MatrixElementType R>
    friend CudaMatrix<R> operator/(R v, const CudaMatrix<R>& m);

public:
    using ElementType = T;

    static bool IsAvaliable()
    {
        auto deviceCount = int {};
        auto err = cudaGetDeviceCount(&deviceCount);
        return err == cudaSuccess && deviceCount > 0;
    }

    CudaMatrix() = default;

    CudaMatrix(size_t row, size_t column)
        : m_row { row }
        , m_column { column }
        , m_cudaBuffer { CudaMalloc<ElementType>(row * column) }
    {
    }

    CudaMatrix(const CudaMatrix& other)
    {
        *this = other;
    }

    CudaMatrix(CudaMatrix&& other)
    {
        *this = std::move(other);
        assert(!other.m_row && !other.m_column && !other.m_cudaBuffer);
    }

    size_t Row() const
    {
        return m_row;
    }

    size_t Column() const
    {
        return m_column;
    }

    CudaMatrix& operator=(const CudaMatrix& other)
    {
        if (this != &other) {
            auto originalNumElement = m_row * m_column;
            auto newNumElement = (m_row = other.m_row) * (m_column = other.m_column);
            if (originalNumElement != newNumElement) {
                m_cudaBuffer = CudaMalloc<ElementType>(newNumElement);
            }
            if (m_cudaBuffer) {
                Cuda(cudaMemcpy, m_cudaBuffer.get(), other.m_cudaBuffer.get(), BufferSize(), cudaMemcpyDeviceToDevice);
            }
        }
        return *this;
    }

    CudaMatrix& operator=(CudaMatrix&& other)
    {
        if (this != &other) {
            m_row = other.m_row;
            m_column = other.m_column;
            m_cudaBuffer = std::move(other.m_cudaBuffer);

            other.m_row = 0;
            other.m_column = 0;
            assert(!other.m_cudaBuffer);
        }
        return *this;
    }

    CudaMatrix& operator=(std::vector<ElementType> data)
    {
        if (m_row * m_column == data.size()) {
            m_row = 1;
            m_column = data.size();
        } else {
            m_row = 1;
            m_column = data.size();
            m_cudaBuffer = CudaMalloc<ElementType>(m_column);
        }
        Cuda(cudaMemcpy, m_cudaBuffer.get(), data.data(), BufferSize(), cudaMemcpyHostToDevice);
        return *this;
    }

    void Write(std::span<ElementType> data)
    {
        if (Row() * Column() != data.size()) {
            throw std::runtime_error { "Elements size is not the same." };
        }

        Cuda(cudaMemcpy, m_cudaBuffer.get(), data.data(), BufferSize(), cudaMemcpyHostToDevice);
    }

    void Write(size_t row, size_t column, const CudaMatrix& m)
    {
        assert(m_row >= row + m.m_row && m_column >= column + m.m_column);

        for (auto r = 0u; r < m.m_row; ++r) {
            Cuda(cudaMemcpy, m_cudaBuffer.get() + (r + row) * m_column + column, m.m_cudaBuffer.get() + r * m.m_column,
                sizeof(ElementType) * m.m_column, cudaMemcpyHostToDevice);
        }
    }

    std::vector<ElementType> Read() const
    {
        std::vector<ElementType> res(m_row * m_column);
        Cuda(cudaMemcpy, res.data(), m_cudaBuffer.get(), BufferSize(), cudaMemcpyDeviceToHost);
        return res;
    }

    CudaMatrix operator+(const CudaMatrix& other) const
    {
        MakeSureShapeIsSame(other);

        CudaMatrix res { m_row, m_column };
        Cuda(CudaAdd, m_cudaBuffer.get(), other.m_cudaBuffer.get(), res.m_cudaBuffer.get(), m_row * m_column);
        return res;
    }

    CudaMatrix& operator+=(const CudaMatrix& other)
    {
        MakeSureShapeIsSame(other);

        return *this = *this + other;
    }

    CudaMatrix operator+(ElementType v) const
    {
        CudaMatrix res { m_row, m_column };
        Cuda(CudaAdd, m_cudaBuffer.get(), v, res.m_cudaBuffer.get(), m_row * m_column);
        return res;
    }

    CudaMatrix operator-(const CudaMatrix& other) const
    {
        MakeSureShapeIsSame(other);

        CudaMatrix res { m_row, m_column };
        Cuda(CudaSub, m_cudaBuffer.get(), other.m_cudaBuffer.get(), res.m_cudaBuffer.get(), m_row * m_column);
        return res;
    }

    CudaMatrix operator*(ElementType v) const
    {
        CudaMatrix<T> res { m_row, m_column };
        Cuda(CudaProduct, v, m_cudaBuffer.get(), res.m_cudaBuffer.get(), m_row * m_column);
        return res;
    }

    CudaMatrix operator*(const CudaMatrix& other) const
    {
        if (m_column != other.m_row) {
            throw std::runtime_error { "Can't dot two matrixs" };
        }

        CudaMatrix res { m_row, other.m_column };
        Cuda(CudaDotProduct, m_cudaBuffer.get(), other.m_cudaBuffer.get(), res.m_cudaBuffer.get(), m_row, m_column,
            other.m_column);
        return res;
    }

    CudaMatrix operator/(ElementType v) const
    {
        CudaMatrix res { m_row, m_column };
        Cuda(CudaDiv, m_cudaBuffer.get(), v, res.m_cudaBuffer.get(), m_row * m_column);
        return res;
    }

    CudaMatrix Transpose() const
    {
        CudaMatrix res { m_column, m_row };
        Cuda(CudaTranspose, m_cudaBuffer.get(), res.m_cudaBuffer.get(), m_row, m_column);
        return res;
    }

    CudaMatrix ElementProduct(const CudaMatrix& other) const
    {
        MakeSureShapeIsSame(other);

        CudaMatrix res { m_row, m_column };
        Cuda(CudaProduct, m_cudaBuffer.get(), other.m_cudaBuffer.get(), res.m_cudaBuffer.get(), m_row * m_column);
        return res;
    }

    CudaMatrix Relu() const
    {
        CudaMatrix res { m_row, m_column };
        Cuda(CudaRelu, m_cudaBuffer.get(), res.m_cudaBuffer.get(), m_row * m_column);
        return res;
    }

    CudaMatrix Exp() const
    {
        CudaMatrix res { m_row, m_column };
        Cuda(CudaExp, m_cudaBuffer.get(), res.m_cudaBuffer.get(), m_row * m_column);
        return res;
    }

    CudaMatrix Pow(T e) const;

    T operator[](size_t row, size_t column) const
    {
        // TODO: need optimization.
        auto vec = Read();
        return vec[row * m_column + column];
    }

    CudaMatrix Sum(bool byRow, bool byColumn) const
    {
        assert(byRow || byColumn);

        if (byRow && byColumn) {
            auto byRowRes = CudaMatrix { m_row, 1 };
            Cuda(CudaSum, m_cudaBuffer.get(), byRowRes.m_cudaBuffer.get(), /*byRow=*/true, m_row, m_column);

            auto res = CudaMatrix { 1, 1 };
            Cuda(CudaSum, byRowRes.m_cudaBuffer.get(), res.m_cudaBuffer.get(), /*byRow=*/false, m_row, /*column=*/1);
            return res;
        } else if (byRow) {
            auto byRowRes = CudaMatrix { m_row, 1 };
            Cuda(CudaSum, m_cudaBuffer.get(), byRowRes.m_cudaBuffer.get(), /*byRow=*/true, m_row, m_column);
            return byRowRes;
        } else if (byColumn) {
            auto byColumnRes = CudaMatrix { 1, m_column };
            Cuda(CudaSum, m_cudaBuffer.get(), byColumnRes.m_cudaBuffer.get(), /*byRow=*/false, m_row, m_column);
            return byColumnRes;
        } else {
            throw std::runtime_error { "'byRow' and 'byColumn' can't be false at the same time" };
        }
    }

    CudaMatrix AddToRow(const CudaMatrix& m) const
    {
        assert(m_column == m.m_column && m.m_row == 1);

        auto res = CudaMatrix { m_row, m_column };
        for (auto r = 0u; r < m_row; ++r) {
            Cuda(CudaAdd, m_cudaBuffer.get() + r * m_column, m.m_cudaBuffer.get(),
                res.m_cudaBuffer.get() + r * m_column, m_column);
        }
        return res;
    }

private:
    void MakeSureShapeIsSame(const CudaMatrix& m) const
    {
        if (m_row != m.m_row || m_column != m.m_column) {
            throw std::runtime_error { std::format(
                "Shape of matrix is not the same: ({}x{} vs {}x{})", m_row, m_column, m.m_row, m.m_column) };
        }
    }

    size_t BufferSize() const
    {
        return sizeof(T) * m_row * m_column;
    }

    size_t m_row {};
    size_t m_column {};
    std::unique_ptr<T, decltype(&cudaFree)> m_cudaBuffer { nullptr, &cudaFree };
};

template <>
CudaMatrix<std::float32_t> CudaMatrix<std::float32_t>::Pow(std::float32_t e) const
{
    CudaMatrix<std::float32_t> res { m_row, m_column };
    Cuda(CudaPow, m_cudaBuffer.get(), e, res.m_cudaBuffer.get(), m_row * m_column);
    return res;
}

template <>
CudaMatrix<std::float16_t> CudaMatrix<std::float16_t>::Pow(std::float16_t e) const
{
    // Seems cuda doesn't supply pow() for float16_t, so convert to cpu to caculate.
    auto data = Read();
    for (auto& d : data) {
        d = pow(d, e);
    }

    CudaMatrix<std::float16_t> res { m_row, m_column };
    res.Write(std::span<std::float16_t> { data.begin(), data.end() });
    return res;
}

export template <MatrixElementType T>
CudaMatrix<T> operator-(T v, const CudaMatrix<T>& m)
{
    CudaMatrix<T> res { m.m_row, m.m_column };
    Cuda(CudaSub, v, m.m_cudaBuffer.get(), res.m_cudaBuffer.get(), m.m_row * m.m_column);
    return res;
}

export template <MatrixElementType T>
CudaMatrix<T> operator/(T v, const CudaMatrix<T>& m)
{
    CudaMatrix<T> res { m.m_row, m.m_column };
    Cuda(CudaDiv, v, m.m_cudaBuffer.get(), res.m_cudaBuffer.get(), m.m_row * m.m_column);
    return res;
}

}