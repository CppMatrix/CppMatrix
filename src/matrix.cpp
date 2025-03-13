/// @file
module;

#include <format>
#include <span>
#include <stdexcept>
#include <stdfloat>
#include <variant>
#include <vector>

export module cpp_matrix:matrix;
import :cpu_matrix;
import :cuda_matrix;
import :matrix_type;

namespace cpp_matrix {

template <typename T>
concept MatrixBackend
    = std::is_same_v<T, backend::CpuMatrix<std::float16_t>> || std::is_same_v<T, backend::CpuMatrix<std::float32_t>>
    || std::is_same_v<T, backend::CudaMatrix<std::float16_t>> || std::is_same_v<T, backend::CudaMatrix<std::float32_t>>;

template <MatrixBackend M>
class Matrix {
public:
    using ElementType = M::ElementType;

    friend Matrix operator+(ElementType v, const Matrix& m);
    friend Matrix operator-(ElementType v, const Matrix& m);
    friend Matrix operator*(ElementType v, const Matrix& m);
    friend Matrix operator/(ElementType v, const Matrix& m);

    /// @brief Create a matrix with random value (value will be between 0 and 1).
    static Matrix Random(size_t row, size_t column)
    {
        auto matrix = Matrix { row, column };
        std::vector<ElementType> initData(row * column);
        for (auto& v : initData) {
            v = std::min(std::rand() / (float)RAND_MAX, 1.f);
        }
        matrix.Write(std::span<ElementType> { initData });
        return matrix;
    }

    static bool IsAvaliable()
    {
        return M::IsAvaliable();
    }

    Matrix()
        : Matrix { 0, 0 }
    {
    }

    Matrix(size_t row, size_t column)
        : m_matrix { row, column }
    {
    }

    /// @brief Create a \c row x \c column matrix. All elements are \c v.
    Matrix(size_t row, size_t column, ElementType v)
        : m_matrix { row, column }
    {
        std::vector<ElementType> data(row * column, v);
        m_matrix.Write(std::span<ElementType> { data });
    }

    Matrix(size_t row, size_t column, std::span<ElementType> initData)
        : m_matrix { row, column }
    {
        Write(initData);
    }

    Matrix(std::initializer_list<std::initializer_list<ElementType>> initData)
    {
        // Find the max column size.
        size_t maxColumn {};
        for (const auto& r : initData) {
            maxColumn = std::max(maxColumn, r.size());
        }

        // Prepare fill data.
        std::vector<ElementType> data(initData.size() * maxColumn);
        auto r = 0u;
        for (const auto& rd : initData) {
            auto c = 0u;
            for (const auto& cd : rd) {
                data[r * maxColumn + c] = cd;
                ++c;
            }
            ++r;
        }

        m_matrix = M { initData.size(), maxColumn };
        m_matrix.Write(std::span<ElementType> { data });
    }

    /** @brief Stack matrixs in sequence horizontally (column wise).
        @exception std::runtime_error If matrix has different number of rows.

        This constructor will create a matrix by stack other matrixs in sequence
        horizontally.

        For example, there are two matrixs:
        \code{.cpp}
        auto a = Matrix {
            {1.0, 1.1},
            {2.0, 2.1}
        };

        auto b = Matrix {
            {1.2, 1.3, 1.4},
            {2.2, 2.3, 2.4}
        };
        \endcode

        Now, you can create the third matrix like this:
        \code{.cpp}
        auto c = Matrix { a, b };
        \endcode

        The matrix \c c will be:
        \verbatim
        c = {
            {1.0, 1.1, 1.2, 1.3, 1.4},
            {2.0, 2.1, 2.2, 2.3, 2.4}
        };
        \endverbatim
    **/
    Matrix(std::initializer_list<Matrix> matrixs)
    {
        if (matrixs.size() == 0) {
            return;
        }

        size_t row = matrixs.begin()->Row();
        size_t column {};
        for (const auto& m : matrixs) {
            if (row != m.Row()) {
                throw std::runtime_error { std::format(
                    "Matrix must have the same number of rows: {} vs {}.", row, m.Row()) };
            }

            column += m.Column();
        }

        m_matrix = M { row, column };
        column = 0;
        for (const auto& m : matrixs) {
            m_matrix.Write(0, column, m.m_matrix);
            column += m.Column();
        }
    }

    Matrix(const Matrix& m)
        : m_matrix { m.m_matrix }
    {
    }

    Matrix(Matrix&& m)
        : m_matrix { std::move(m.m_matrix) }
    {
    }

    template <size_t N>
    void Write(std::span<ElementType, N> data)
    {
        m_matrix.Write(data);
    }

    std::vector<ElementType> Read() const
    {
        return m_matrix.Read();
    }

    Matrix operator-() const
    {
        return m_matrix * -1;
    }

    Matrix operator+(const Matrix& other) const
    {
        return m_matrix + other.m_matrix;
    }

    Matrix& operator+=(const Matrix& other)
    {
        m_matrix += other.m_matrix;
        return *this;
    }

    Matrix operator-(const Matrix& other) const
    {
        return m_matrix - other.m_matrix;
    }

    Matrix operator+(ElementType v) const
    {
        return m_matrix + v;
    }

    Matrix operator/(ElementType v) const
    {
        return m_matrix / v;
    }

    Matrix operator-(ElementType v) const
    {
        return *this + (-v);
    }

    Matrix operator*(ElementType v) const
    {
        return m_matrix * v;
    }

    Matrix operator*(const Matrix& other) const
    {
        return m_matrix * other.m_matrix;
    }

    size_t Row() const
    {
        return m_matrix.Row();
    }

    size_t Column() const
    {
        return m_matrix.Column();
    }

    Matrix& operator=(std::vector<ElementType> data)
    {
        m_matrix = std::move(data);
        return *this;
    }

    Matrix& operator=(ElementType f)
    {
        return operator=(std::vector<ElementType> { f });
    }

    Matrix& operator=(std::span<ElementType> data)
    {
        return operator=(std::vector<ElementType> { data.begin(), data.end() });
    }

    Matrix& operator=(Matrix&& m)
    {
        m_matrix = std::move(m.m_matrix);
        return *this;
    }

    Matrix Transpose() const
    {
        return m_matrix.Transpose();
    }

    Matrix ElementProduct(const Matrix& other) const
    {
        return m_matrix.ElementProduct(other.m_matrix);
    }

    Matrix Relu() const
    {
        return m_matrix.Relu();
    }

    /// @brief Caculate exp() element-wise.
    Matrix Exp() const
    {
        return m_matrix.Exp();
    }

    Matrix Pow(ElementType e) const
    {
        return m_matrix.Pow(e);
    }

    float operator[](size_t row, size_t column) const
    {
        return m_matrix[row, column];
    }

    /// @brief Add value each row. The input should be a 1xC matrix.
    Matrix AddToRow(const Matrix& m) const
    {
        if (m_matrix.Column() != m.m_matrix.Column() || m.m_matrix.Row() != 1) {
            throw std::runtime_error { std::format("Input matrix shoudl be a 1x{} matrix", m_matrix.Column()) };
        }

        return m_matrix.AddToRow(m.m_matrix);
    }

    /// @brief Returns the sum of all elements in the matrix.
    ///        The value will be a 1 x 1 matrix.
    Matrix Sum() const
    {
        return m_matrix.Sum(/*byRow=*/true, /*byColumn=*/true);
    }

    /// @brief Returns the sum of all elements in the same row.
    ///        If this is a R x C matrix, the result will be a R x 1 matrix.
    Matrix SumByRow() const
    {
        return m_matrix.Sum(/*byRow=*/true, /*byColumn=*/false);
    }

    /// @brief Returns the sum of all elements in the same column.
    ///        If this is a R x C matrix, the result will be a  1 x C matrix.
    Matrix SumByColumn() const
    {
        return m_matrix.Sum(/*byRow=*/false, /*byColumn=*/true);
    }

private:
    Matrix(M m)
        : m_matrix { std::move(m) }
    {
    }

    M m_matrix {};
};

export template <MatrixElementType T>
using CpuMatrix = Matrix<backend::CpuMatrix<T>>;

export template <MatrixElementType T>
using CudaMatrix = Matrix<backend::CudaMatrix<T>>;

#define OpOperators(M, op)                                                                                             \
    M operator op(typename M::ElementType v, const M& m)                                                               \
    {                                                                                                                  \
        return operator op(v, m.m_matrix);                                                                             \
    }

#define ReverseOpOperators(M, op)                                                                                      \
    M operator op(typename M::ElementType v, const M& m)                                                               \
    {                                                                                                                  \
        return m.operator op(v);                                                                                       \
    }

// clang-format off
#define OperatorsElementType(M, T)  \
    ReverseOpOperators(M<T>, +)     \
    OpOperators(M<T>, -)            \
    ReverseOpOperators(M<T>, *)     \
    OpOperators(M<T>, /)            \

#define Operators(M)                        \
    OperatorsElementType(M, std::float16_t) \
    OperatorsElementType(M, std::float32_t)
// clang-format on

Operators(CpuMatrix);
Operators(CudaMatrix);

}