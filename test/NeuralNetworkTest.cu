#include "Matrix.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkOptimizer.hpp"
#include "NeuralNetworkSaver.hpp"

// Define layers using a custom matrix class.
typedef LinearFCL<Matrix_F> LinearFCL_F;
typedef SigmoidFCL<Matrix_F> SigmoidFCL_F;
typedef ReLUFCL<Matrix_F> ReLUFCL_F;
typedef LeakyReLUFCL<Matrix_F> LeakyReLUFCL_F;
// Define a network using a custom matrix class.
template <typename... Layers>
using NeuralNetwork_F = ai::NeuralNetwork<Matrix_F, Layers...>;

int main() {
    Matrix_F input({100, 10, 7.5, 5, 2.5, 0, -2.5, -7.5, -10, -100}, 10);
    Matrix_F expectedOutput = input.applyFunction<ai::sigmoid>();

    SigmoidFCL_F inputLayer(1, 25);
    LeakyReLUFCL_F hiddenLayer1(25, 25);
    LinearFCL_F outputLayer(25, 1);

    NeuralNetwork_F<SigmoidFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F, LinearFCL_F> layerTest(inputLayer, hiddenLayer1, hiddenLayer1, outputLayer);

    layerTest.feedForward(input).display("Testing Layer Manager feedForward");

    // Let's create an optimizer!
    ai::NeuralNetworkOptimizer<Matrix_F, ai::mse_prime<Matrix_F>> optimizer;

    std::cout << "Testing Layer Manager training" << std::endl;
    expectedOutput.display("Expected");
    // Train for 1 iteration (default).
    optimizer.train(layerTest, input, expectedOutput, 0.01);
    // Train for 1000 iterations.
    optimizer.train<1000>(layerTest, input, expectedOutput, 0.01);
    layerTest.feedForward(input).display("Actual");

    // Let's do weight saving using a saver!
    ai::NeuralNetworkSaver saver;
    saver.save(layerTest, "./test/networkWeights.bin");

    NeuralNetwork_F<SigmoidFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F, LinearFCL_F> loadingTest(inputLayer, hiddenLayer1, hiddenLayer1, outputLayer);
    // Let's do weights loading! We can use both the save and load methods statically too!
    ai::NeuralNetworkSaver::load(loadingTest, "./test/networkWeights.bin");
    loadingTest.feedForward(input).display("Loaded Network");

}
