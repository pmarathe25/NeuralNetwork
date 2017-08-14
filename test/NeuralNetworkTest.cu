#include "StealthMatrix.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkOptimizer.hpp"
#include "NeuralNetworkSaver.hpp"
// Define layers using a custom matrix class.
typedef StealthAI::LinearFCL<StealthMatrix_F> LinearFCL_F;
typedef StealthAI::SigmoidFCL<StealthMatrix_F> SigmoidFCL_F;
typedef StealthAI::ReLUFCL<StealthMatrix_F> ReLUFCL_F;
typedef StealthAI::LeakyReLUFCL<StealthMatrix_F> LeakyReLUFCL_F;
// Define a network using a custom matrix class.
template <typename... Layers>
using NeuralNetwork_F = StealthAI::NeuralNetwork<StealthMatrix_F, Layers...>;

int main() {
    StealthMatrix_F input({100, 10, 7.5, 5, 2.5, 0, -2.5, -7.5, -10, -100}, 10);
    StealthMatrix_F expectedOutput = input.applyFunction<StealthAI::sigmoid>();

    StealthAI::DataSet<StealthMatrix_F> trainingInputs, trainingExpectedOutputs;
    trainingInputs.push_back(input);
    trainingExpectedOutputs.push_back(expectedOutput);


    SigmoidFCL_F inputLayer(1, 50);
    LeakyReLUFCL_F hiddenLayer1(50, 50);
    LinearFCL_F outputLayer(50, 1);

    NeuralNetwork_F<SigmoidFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F, LinearFCL_F> layerTest(inputLayer, hiddenLayer1, hiddenLayer1, outputLayer);

    StealthAI::NeuralNetworkOptimizer<StealthMatrix_F, StealthAI::mse<StealthMatrix_F>, StealthAI::mse_prime<StealthMatrix_F>> optimizer;
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
    StealthAI::NeuralNetworkSaver saver;
    saver.save(layerTest, "./test/networkWeights.bin");

    NeuralNetwork_F<SigmoidFCL_F, LeakyReLUFCL_F, LeakyReLUFCL_F, LinearFCL_F> loadingTest(inputLayer, hiddenLayer1, hiddenLayer1, outputLayer);
    // Let's do weights loading! We can use both the save and load methods statically too!
    StealthAI::NeuralNetworkSaver::load(loadingTest, "./test/networkWeights.bin");
    loadingTest.feedForward(input).display("Loaded Network");
}
