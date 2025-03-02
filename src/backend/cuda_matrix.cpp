module;

#include <cassert>
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

    CudaMatrix operator+(T v) const
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

    CudaMatrix operator*(const CudaMatrix& other) const
    {
        return {};
    }

    CudaMatrix Sigmoid() const
    {
        return {};
    }

    CudaMatrix Transpose() const
    {
        return {};
    }

    CudaMatrix ElementProduct(const CudaMatrix& other) const
    {
        return {};
    }

    CudaMatrix Relu() const
    {
        return {};
    }

    T operator[](size_t row, size_t column) const
    {
        // TODO: need optimization.
        auto vec = Read();
        return vec[row * m_column + column];
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

}