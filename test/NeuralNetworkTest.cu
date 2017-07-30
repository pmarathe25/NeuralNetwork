#include "Matrix.hpp"
#include "NeuralNetwork/NeuralNetwork.hpp"
#include "NeuralNetwork/Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork/Layer/LayerManager.hpp"
#include <random>
#include <math.h>

typedef SigmoidFCL<Matrix_F> SigmoidFCL_F;
typedef ReLUFCL<Matrix_F> ReLUFCL_F;
// Define a layer manager using our matrix library.
template <typename... Layers>
using NeuralNetwork_MSE_F = NeuralNetwork_MSE<Matrix_F, Layers...>;

int main() {
    math::Matrix<float> input({10, 7.5, 5, 2.5, 0, -2.5, -7.5, -10}, 7, 1);
    math::Matrix<float> expectedOutput = input.applyFunction<ai::sigmoid>();

    input.display();
    ai::NeuralNetwork<float> net({1, 100, 100, 1}, ai::NeuralNetwork<float>::ANALYTIC);
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
    //
    // Test Layer functionality
    std::cout << "Testing Fully Connected Layer" << std::endl;
    SigmoidFCL_F testLayer(1, 10);
    ReLUFCL_F testLayer2(10, 1);
    testLayer.feedForward(input).display();


    // LayerManager<SigmoidFCL, ReLUFCL> layerTest(testLayer, testLayer2);
    // LayerManager<Matrix_F, mse_prime<Matrix_F>, SigmoidFCL, ReLUFCL> layerTest(testLayer, testLayer2);
    NeuralNetwork_MSE_F<SigmoidFCL_F, ReLUFCL_F> layerTest(testLayer, testLayer2);
    std::cout << "Testing Layer Manager feedForward" << std::endl;
    std::cout << "Expected" << std::endl;
    testLayer2.feedForward(testLayer.feedForward(input)).display();
    std::cout << "Actual" << std::endl;
    layerTest.feedForward(input).display();

    std::cout << "Testing Layer Manager getLayerOutput" << std::endl;
    std::cout << "Expected" << std::endl;
    testLayer.feedForward(input).display();
    std::cout << "Actual" << std::endl;
    layerTest.getLayerOutput(input).display();

    // // Test saving current network
    // std::cout << "Saving weights..." << std::endl;
    // net.saveWeights("test/weights");
    // std::cout << "Weights saved." << std::endl;
    // std::cout << std::endl;
    // // Test loading second network.
    // ai::NeuralNetwork<float> net2;
    // std::cout << "Created network 2." << std::endl;
    // std::cout << "Reading weights..." << std::endl;
    // net2.loadWeights("test/weights");
    // std::cout << "Weights loaded." << std::endl;
}
