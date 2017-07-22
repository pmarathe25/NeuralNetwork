#include "NeuralNetwork/NeuralNetwork.hpp"
#include "NeuralNetwork/Layer/FullyConnectedLayer.hpp"
#include <random>
#include <math.h>

int main() {
    math::Matrix<double> input({{1, 1, 1}, {0.75, 0.75, 0.75}, {0.5, 0.5, 0.5}, {0.25, 0.25, 0.25}, {0, 0, 0}});
    math::Matrix<double> expectedOutput = input.applyFunction<ai::sigmoid>();

    // input.display();
    ai::NeuralNetwork<double> net({3, 3}, ai::NeuralNetwork<double>::RELU);
    std::cout << "Created network." << std::endl;
    std::cout << "Initial Output" << std::endl;
    net.feedForward(input).display();
    // Train
    for (int i = 0; i < 2000; ++i) {
        net.train(input, expectedOutput, 0.01);
    }
    std::cout << "Actual Output" << std::endl;
    net.feedForward(input).display();

    std::cout << "Expected Output" << std::endl;
    expectedOutput.display();

    // Test Layer functionality
    std::cout << "Testing Fully Connected Layer" << std::endl;
    ai::FullyConnectedLayer<double, ai::relu> testLayer(3, 3);
    testLayer.feedForward(input).display();

    // // Test saving current network
    // std::cout << "Saving weights..." << std::endl;
    // net.saveWeights("test/weights");
    // std::cout << "Weights saved." << std::endl;
    // std::cout << std::endl;
    // // Test loading second network.
    // ai::NeuralNetwork<double> net2;
    // std::cout << "Created network 2." << std::endl;
    // std::cout << "Reading weights..." << std::endl;
    // net2.loadWeights("test/weights");
    // std::cout << "Weights loaded." << std::endl;
}
