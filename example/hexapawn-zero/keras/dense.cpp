module;

#include <memory>
#include <string>

import cpp_matrix;

export module keras:dense;
import :layer;

namespace keras {

template <typename Matrix>
class DenseImpl : public ILayer<Matrix> {
    Matrix& GetData() override
    {
        return m_matrix;
    }

private:
    Matrix m_matrix {};
};

export template <typename Matrix>
class Dense : public Layer<Matrix> {
public:
    Dense(size_t units, std::string activation) { }

    Dense(size_t units, std::string name, std::string activation) { }

    Dense& operator()(Layer<Matrix> input)
    {
        return *this;
    }

private:
    DenseImpl<Matrix>& impl()
    {
        return *static_cast<DenseImpl<Matrix>*>(this->m_p.get());
    }
};

}