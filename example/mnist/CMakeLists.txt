add_executable(mnist
    main.cpp
)
target_sources(mnist PUBLIC FILE_SET CXX_MODULES FILES
    neural_network.cpp
)
target_link_libraries(mnist PRIVATE
    cpp_matrix
)