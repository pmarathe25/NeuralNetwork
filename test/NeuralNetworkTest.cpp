#include "NeuralNetwork/NeuralNetwork.hpp"
#include <random>
#include <math.h>

int main() {
    math::Matrix<float> input0 = math::Matrix<float>({1, 1, 1});
    math::Matrix<float> input1 = math::Matrix<float>({0, 0, 0});

    math::Matrix<float> desiredOutput = math::Matrix<float>({{1, 1, 1}, {0, 0, 0}});

    // input.display();
    ai::NeuralNetwork<float> net = ai::NeuralNetwork<float>({3, 3});
    std::cout << "Created network." << std::endl;
    std::cout << "Saving weights..." << std::endl;
    net.saveWeights("test/weights");
    std::cout << "Weights saved." << std::endl;
    std::cout << std::endl;
    net.feedForward(input0).display();
    std::cout << std::endl;
    for (int i = 0; i < 2000; ++i) {
        // net.train(input, desiredOutput, 0.01);
        // net.train(input2, desiredOutput2, 0.01);
        net.train(desiredOutput, desiredOutput, 0.1);
    }
    net.feedForward(input0).display();
    net.feedForward(input1).display();
    // net.feedForward({0.7, 0.7, 0.7}).display();
    ai::NeuralNetwork<float> net2;
    std::cout << "Created network 2." << std::endl;
    std::cout << "Reading weights..." << std::endl;
    net2.loadWeights("test/weights");
    std::cout << "Weights loaded." << std::endl;
}
