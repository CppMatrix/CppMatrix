add_library(cpp_matrix.neural_network)
target_sources(cpp_matrix.neural_network PUBLIC FILE_SET CXX_MODULES FILES
    backend.cpp
    loss_functions.cpp
    module.cpp
)
target_link_libraries(cpp_matrix.neural_network PUBLIC
    cpp_matrix
)