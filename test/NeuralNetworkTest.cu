#include "Matrix.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkOptimizer.hpp"
#include "NeuralNetworkSaver.hpp"

typedef SigmoidFCL<Matrix_F> SigmoidFCL_F;
typedef ReLUFCL<Matrix_F> ReLUFCL_F;
typedef LeakyReLUFCL<Matrix_F> LeakyReLUFCL_F;
// Define a network using a custom matrix class.
template <typename... Layers>
using NeuralNetwork_F = ai::NeuralNetwork<Matrix_F, Layers...>;

template <typename Matrix, typename... Layers>
void testFunc(ai::NeuralNetwork<Matrix, Layers...>& network) {
    std::cout << sizeof...(Layers) << '\n';
    std::cout << network.getDepth() << '\n';
    Matrix_F input({10, 7.5, 5, 2.5, 0, -2.5, -7.5, -10}, 8);
    network.feedForward(input).display();
}

int main() {
    Matrix_F input({10, 7.5, 5, 2.5, 0, -2.5, -7.5, -10}, 8);
    Matrix_F expectedOutput = input.applyFunction<ai::sigmoid>();

    SigmoidFCL_F inputLayer(1, 25);
    LeakyReLUFCL_F hiddenLayer1(25, 25);
    LeakyReLUFCL_F outputLayer(25, 1);

    NeuralNetwork_F<SigmoidFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F> layerTest(inputLayer, hiddenLayer1, hiddenLayer1, outputLayer);

    testFunc(layerTest);

    // Let's create an optimizer!
    ai::NeuralNetworkOptimizer<Matrix_F, ai::mse_prime<Matrix_F>, SigmoidFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F> optimizer(layerTest);
    std::cout << "Testing Layer Manager feedForward" << std::endl;
    layerTest.feedForward(input).display();

    std::cout << "Testing Layer Manager training" << std::endl;
    std::cout << "Expected" << std::endl;
    expectedOutput.display();
    std::cout << "Actual" << std::endl;
    optimizer.train(input, expectedOutput, 0.01);
    layerTest.feedForward(input).display();

    // Let's do weight saving!
    ai::NeuralNetworkSaver saver;
    saver.save(layerTest, "./test/networkWeights.bin");

    // Let's do weights loading!
    NeuralNetwork_F<SigmoidFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F> loadingTest(inputLayer, hiddenLayer1, hiddenLayer1, outputLayer);
    ai::NeuralNetworkSaver loader;
    loader.load(loadingTest, "./test/networkWeights.bin");
    std::cout << "Loaded Network" << '\n';
    loadingTest.feedForward(input).display();

}
