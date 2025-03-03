module;

#include <utility>

export module keras:input;
import :layer;
import :shape;

namespace keras {

template <typename Matrix>
class InputImpl : public ILayer<Matrix> {
    Matrix& GetData() override
    {
        return m_matrix;
    }

private:
    Matrix m_matrix {};
};

export template <typename Matrix>
class Input : public Layer<Matrix> {
public:
    Input(Shape shape) { }
};

}