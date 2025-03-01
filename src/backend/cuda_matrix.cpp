module;

#include <cstddef>
#include <cuda_runtime.h>
#include <span>
#include <vector>

export module cpp_matrix:cuda_matrix;
import :matrix_type;

namespace cpp_matrix::backend {

export template <MatrixElementType T>
class CudaMatrix {
public:
    using ElementType = T;

    CudaMatrix() = default;

    CudaMatrix(size_t row, size_t column) { }

    static bool IsAvaliable()
    {
        auto deviceCount = int {};
        auto err = cudaGetDeviceCount(&deviceCount);
        return err == cudaSuccess && deviceCount > 0;
    }

    size_t Row() const
    {
        return m_row;
    }

    size_t Column() const
    {
        return m_column;
    }

    CudaMatrix& operator=(std::vector<T> data)
    {
        return *this;
    }

    void Write(std::span<T> data) { }

    std::vector<T> Read() const
    {
        return {};
    }

    CudaMatrix operator+(const CudaMatrix& other) const
    {
        return {};
    }

    CudaMatrix& operator+=(const CudaMatrix& other)
    {
        return *this;
    }

    CudaMatrix operator+(T v) const
    {
        return {};
    }

    CudaMatrix operator-(const CudaMatrix& other) const
    {
        return {};
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
        return {};
    }

private:
    size_t m_row {};
    size_t m_column {};
};

}