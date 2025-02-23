CppMatrix
==
Supported backed:

* Cpu
* Cuda
* WebGpu (experiment)

Supported type:

* std::float16_t
* std::float32_t

## Install Dependencies
### Install Cuda-Toolkit 12.8

    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb

The above instructions will install cuda-toolkit for WSL Ubuntu. If you need install for other linux distributions,
you can follow the instruction on https://developer.nvidia.com/cuda-downloads

After install, add the path to PATH environment variable:

    export PATH=/usr/local/cuda-12.8/bin:$PATH

You can also add this line to your `~/.bashrc` file for convenience.  

### Install other dependencies
    sudo apt install \
        clang \
        clang-tools \
        cmake \
        libgtest-dev \
        ninja-build \
        nlohmann-json3-dev\

## Build

    mkdir build
    cd build
    CXX=clang++ cmake .. -GNinja
    ninja

## Example

### Mnist
This example is rewriting from https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part2_neural_network_mnist_data.ipynb.
Download the traning data from https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/refs/heads/master/mnist_dataset/mnist_train_100.csv,
and the test data from https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/refs/heads/master/mnist_dataset/mnist_test_10.csv

then run it like this:

    $ ./build/example/mnist/mnist mnist_train_100.csv mnist_test_10.csv

or using cuda backend:

    $ ./build/example/mnist/mnist mnist_train_100.csv mnist_test_10.csv --use-cuda

or using webgpu backend:

    $ ./build/example/mnist/mnist mnist_train_100.csv mnist_test_10.csv --use-webgpu

output:

    prediction result: 7, actual result: 7 o
    prediction result: 1, actual result: 2 x
    prediction result: 1, actual result: 1 o
    prediction result: 0, actual result: 0 o
    prediction result: 4, actual result: 4 o
    prediction result: 1, actual result: 1 o
    prediction result: 4, actual result: 4 o
    prediction result: 4, actual result: 9 x
    prediction result: 4, actual result: 5 x
    prediction result: 9, actual result: 9 o
    performance = 0.7
