#include "NeuralNetwork/NeuralNetwork.hpp"
#include "NeuralNetwork/Layer/FullyConnectedLayer.hpp"
#include <random>
#include <math.h>

int main() {
    math::Matrix<double> input0({0.90, 0.90, 0.90});
    math::Matrix<double> input1({0.5, 0.5, 0.5});

    math::Matrix<double> desiredOutput({{1, 1, 1}, {0.75, 0.75, 0.75}, {0.5, 0.5, 0.5}, {0.25, 0.25, 0.25}, {0, 0, 0}});
    ai::FullyConnectedLayer<double> testLayer(3, 3);
    testLayer.feedForward(desiredOutput).display();

    // input.display();
    ai::NeuralNetwork<double> net({3, 3}, ai::NeuralNetwork<double>::RELU);
    std::cout << "Created network." << std::endl;
    std::cout << "Saving weights..." << std::endl;
    net.saveWeights("test/weights");
    std::cout << "Weights saved." << std::endl;
    std::cout << std::endl;
    net.feedForward(desiredOutput).display();
    std::cout << std::endl;
    for (int i = 0; i < 2000; ++i) {
        // net.train(input, desiredOutput, 0.01);
        // net.train(input2, desiredOutput2, 0.01);
        net.train(desiredOutput, desiredOutput, 0.01);
    }
    net.feedForward(desiredOutput).display();
    // net.feedForward(input0).display();
    // net.feedForward(input1).display();
    // net.feedForward({0.7, 0.7, 0.7}).display();
    ai::NeuralNetwork<double> net2;
    std::cout << "Created network 2." << std::endl;
    std::cout << "Reading weights..." << std::endl;
    net2.loadWeights("test/weights");
    std::cout << "Weights loaded." << std::endl;
}
