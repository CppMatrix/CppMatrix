/// @file
module;
#include <memory>

export module cpp_matrix.neural_network:modules_module;
export import :backend;

namespace cpp_matrix::neural_network {

export template <Backend Matrix>
class IModule {
public:
    virtual ~IModule() = default;
    virtual Matrix Forward(const Matrix& input) = 0;
};

export template <typename Matrix>
class Module {
public:
    Module() = default;

    Module(std::shared_ptr<IModule<Matrix>> pModule)
        : m_pModule { std::move(pModule) }
    {
    }

    Matrix operator()(const Matrix& input)
    {
        return m_pModule->Forward(input);
    }

protected:
    std::shared_ptr<IModule<Matrix>> m_pModule {};
};

}