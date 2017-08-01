#include "Matrix.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkOptimizer.hpp"
// #include <random>
// #include <math.h>

typedef SigmoidFCL<Matrix_F> SigmoidFCL_F;
typedef ReLUFCL<Matrix_F> ReLUFCL_F;
template <typename... Layers>
using NeuralNetwork = ai::NeuralNetwork<Matrix_F, Layers...>;

int main() {
    Matrix_F input({10, 7.5, 5, 2.5, 0, -2.5, -7.5, -10}, 8);
    Matrix_F expectedOutput = input.applyFunction<ai::sigmoid>();

    // Test Layer functionality
    SigmoidFCL_F testLayer1(1, 10);
    ReLUFCL_F testLayer2(10, 1);

    NeuralNetwork<SigmoidFCL_F, ReLUFCL_F> layerTest(testLayer1, testLayer2);
    // Let's create an optimizer!
    ai::NeuralNetworkOptimizer<Matrix_F, ai::mse_prime<Matrix_F>, SigmoidFCL_F, ReLUFCL_F> optimizer(layerTest);
    std::cout << "Testing Layer Manager feedForward" << std::endl;
    layerTest.feedForward(input).display();

    // std::cout << "Testing Layer Manager backpropagate" << std::endl;
    // std::cout << "Expected" << std::endl;
    // expectedOutput.display();
    // std::cout << "Actual" << std::endl;
    // layerTest.backpropagate(input, expectedOutput);
    // layerTest.feedForward(input).display();
    //
    std::cout << "Testing Layer Manager training" << std::endl;
    std::cout << "Expected" << std::endl;
    expectedOutput.display();
    std::cout << "Actual" << std::endl;
    optimizer.train(input, expectedOutput);
    // layerTest.train(input, expectedOutput, 0.001);
    layerTest.feedForward(input).display();
    //
    // SigmoidFCL_F perfectSigmoid(1, 1);
    // NeuralNetwork_MSE_F<SigmoidFCL_F> sigmoidNetwork(perfectSigmoid);
    // std::cout << "Testing perfect sigmoid network before training" << std::endl;
    // sigmoidNetwork.feedForward(input).display();
    // std::cout << "Weights" << std::endl;
    // perfectSigmoid.getWeights().display();
    // std::cout << "Biases" << std::endl;
    // perfectSigmoid.getBiases().display();
    // std::cout << "Expected" << std::endl;
    // expectedOutput.display();
    // std::cout << "Actual (after training)" << std::endl;
    // sigmoidNetwork.train(input, expectedOutput);
    // sigmoidNetwork.feedForward(input).display();
    // std::cout << "Weights" << std::endl;
    // perfectSigmoid.getWeights().display();
    // std::cout << "Biases" << std::endl;
    // perfectSigmoid.getBiases().display();

}
