add_library(cpp_matrix.nn)
target_sources(cpp_matrix.nn PUBLIC FILE_SET CXX_MODULES FILES
    functional/module.cpp
    functional/mse_loss.cpp
    functional/sigmoid.cpp
    modules/linear.cpp
    modules/module.cpp
    backend.cpp
    module.cpp
    tensor.cpp
)
target_link_libraries(cpp_matrix.nn PUBLIC
    cpp_matrix
)