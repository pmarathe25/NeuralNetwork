#include "Matrix.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
// #include <random>
// #include <math.h>

typedef SigmoidFCL<Matrix_F> SigmoidFCL_F;
typedef ReLUFCL<Matrix_F> ReLUFCL_F;
// Define a layer manager using our matrix library.
template <typename... Layers>
using NeuralNetwork_MSE_F = NeuralNetwork_MSE<Matrix_F, Layers...>;

int main() {
    math::Matrix<float> input({1, 7.5, 5, 2.5, 0, -2.5, -7.5, -10}, 8);
    math::Matrix<float> expectedOutput = input.applyFunction<ai::sigmoid>();

    // Test Layer functionality
    SigmoidFCL_F testLayer(1, 10);
    ReLUFCL_F testLayer2(10, 1);

    NeuralNetwork_MSE_F<SigmoidFCL_F, ReLUFCL_F> layerTest(testLayer, testLayer2);
    std::cout << "Testing Layer Manager feedForward" << std::endl;
    layerTest.feedForward(input).display();

    std::cout << "Testing Layer Manager backpropagate" << std::endl;
    std::cout << "Expected" << std::endl;
    expectedOutput.display();
    std::cout << "Actual" << std::endl;
    layerTest.backpropagate(input, expectedOutput);
    layerTest.feedForward(input).display();

    std::cout << "Testing Layer Manager training" << std::endl;
    std::cout << "Expected" << std::endl;
    expectedOutput.display();
    std::cout << "Actual" << std::endl;
    layerTest.train(input, expectedOutput, 0.01);
    layerTest.feedForward(input).display();
}
