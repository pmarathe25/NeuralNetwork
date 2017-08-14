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

    ai::DataSet<Matrix_F> trainingInputs, trainingExpectedOutputs;
    trainingInputs.push_back(input);
    trainingExpectedOutputs.push_back(expectedOutput);


    SigmoidFCL_F inputLayer(1, 50);
    LeakyReLUFCL_F hiddenLayer1(50, 50);
    LinearFCL_F outputLayer(50, 1);

    NeuralNetwork_F<SigmoidFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F, LinearFCL_F> layerTest(inputLayer, hiddenLayer1, hiddenLayer1, outputLayer);

    ai::NeuralNetworkOptimizer<Matrix_F, ai::mse<Matrix_F>, ai::mse_prime<Matrix_F>> optimizer;
    layerTest.feedForward(input).display("Testing Layer Manager feedForward");
    optimizer.getAverageCost(layerTest, input, expectedOutput).display("Initial average cost");

    // Let's create an optimizer!
    expectedOutput.display("Testing Layer Manager training\nExpected Output");
    // Train for 1 iteration.
    optimizer.trainMinibatch(layerTest, input, expectedOutput, 0.01);
    // Train for 1000 epochs.
    optimizer.train<20>(layerTest, trainingInputs, trainingExpectedOutputs, trainingInputs, trainingExpectedOutputs, 0.01);
    // optimizer.trainMinibatch<1000>(layerTest, input, expectedOutput, 0.01);
    layerTest.feedForward(input).display("Actual");
    optimizer.getAverageCost(layerTest, input, expectedOutput).display("Final average cost");

    // Let's do weight saving using a saver!
    ai::NeuralNetworkSaver saver;
    saver.save(layerTest, "./test/networkWeights.bin");

    NeuralNetwork_F<SigmoidFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F, LinearFCL_F> loadingTest(inputLayer, hiddenLayer1, hiddenLayer1, outputLayer);
    // Let's do weights loading! We can use both the save and load methods statically too!
    ai::NeuralNetworkSaver::load(loadingTest, "./test/networkWeights.bin");
    loadingTest.feedForward(input).display("Loaded Network");
}
