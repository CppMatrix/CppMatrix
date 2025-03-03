module;

#include <functional>
#include <map>
#include <string>
#include <variant>

export module keras:model;
import :layer;
import :losses;

namespace keras {

export template <typename Matrix>
class Model {
    using HeadLabel = std::string;
    using Layer = keras::Layer<Matrix>;
    using LossFuncNameOrLossFunc = std::variant<std::string, losses::Loss<Matrix>>;

public:
    struct CompileArgs {
        std::string optimizer { "rmsprop" };
        std::map<HeadLabel, LossFuncNameOrLossFunc> loss {};
    };

    Model(Layer input, std::initializer_list<Layer> outputs) { }

    void compile(CompileArgs args)
    {
        std::string head;
        if (const auto* lossFuncName = std::get_if<std::string>(&args.loss[head])) {
            auto loss = losses::GetLossByName<Matrix>(*lossFuncName);
        }
    }
};

}