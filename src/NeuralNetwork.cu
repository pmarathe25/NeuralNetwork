#include "NeuralNetwork/NeuralNetwork.hpp"
#include <math.h>
#include <fstream>
#include <iostream>
// Include Neural Network functions.
#include "NeuralNetworkCUDAFunctions.cu"

namespace ai {
    template <typename T>
    NeuralNetwork<T>::NeuralNetwork(aFunc func, cFunc func2) {
        activationFunc = func;
        costFunc = func2;
    }

    template <typename T>
    NeuralNetwork<T>::NeuralNetwork(std::vector<int> layers, aFunc func, cFunc func2) : NeuralNetwork(func, func2) {
        // Initialize weights and biases.
        numLayers = layers.size();
        inputSize = layers.front();
        for (int i = 0; i < numLayers - 1; ++i) {
            weights.push_back(math::Matrix<T>(layers[i], layers[i + 1]));
            biases.push_back(math::Matrix<T>(1, layers[i + 1]));
            deltas.push_back(math::Matrix<T>(1, layers[i + 1]));
            // Initialize output cache.
            outputs.push_back(math::Matrix<T>(1, layers[i]));
            activationOutputs.push_back(math::Matrix<T>(1, layers[i]));
        }
        // Outputs have one extra layer at the very end.
        outputs.push_back(math::Matrix<T>(1, layers.back()));
        activationOutputs.push_back(math::Matrix<T>(1, layers.back()));
        // Keep track of input size.
        // Randomize weights and biases.
        initializeWeights();
    }

    template <typename T>
    void NeuralNetwork<T>::train(const math::Matrix<T>& input, const math::Matrix<T>& desiredOutput, T learningRate) {
        feedForward(input);
        T scaleFactor = 1 / (double) input.numRows();
        deltas.back() = costDerivative(activationOutputs.back(), desiredOutput).hadamard(computeActivationFunctionDerivative(activationOutputs.size() - 1));
        // Debug
        std::cout << "\n========Output before activation========" << std::endl;
        (outputs.back()).display();
        std::cout << "\n========Activation========" << std::endl;
        (activationOutputs.back()).display();
        std::cout << "========Cost Derivative========" << std::endl;
        (costDerivative(activationOutputs.back(), desiredOutput)).display();
        std::cout << "\n========Activation Derivative========" << std::endl;
        (computeActivationFunctionDerivative(activationOutputs.size() - 1)).display();
        std::cout << std::endl;
        std::cout << "\n========Deltas========" << std::endl;
        deltas.back().display();
        // End Debug
        for (int i = deltas.size() - 2; i >= 0; --i) {
            deltas[i] = (deltas[i + 1] * weights[i + 1].transpose()).hadamard(computeActivationFunctionDerivative(i + 1));
            // Debug
            std::cout << "Layer " << i + 1 << std::endl;
            std::cout << "========Weights Transpose========" << std::endl;
            weights[i + 1].transpose().display();
            std::cout << "\n========Deltas========" << std::endl;
            deltas[i + 1].display();
            std::cout << "\n========Weights Transpose * Deltas========" << std::endl;
            (deltas[i + 1] * weights[i + 1].transpose()).display();
            std::cout << "\n========Output After Activation Derivative========" << std::endl;
            (computeActivationFunctionDerivative(i + 1)).display();
            std::cout << std::endl;
            // End Debug
        }
        // Gradients for each layers.
        std::vector<math::Matrix<T> > weightDeltas;
        std::vector<math::Matrix<T> > biasDeltas;
        // Compute gradients.
        for (int i = 0; i < deltas.size(); ++i) {
            // Debug
            // std::cout << "Layer " << i + 1 << std::endl;
            // std::cout << "========Outputs transpose========" << std::endl;
            // (activationOutputs[i + 1].transpose()).display();
            // std::cout << "\n========Deltas========" << std::endl;
            // (deltas[i]).display();
            // End Debug
            // Get average weight deltas instead of the sum so learning rate is not affected.
            weightDeltas.push_back((activationOutputs[i + 1].transpose() * deltas[i]) * scaleFactor);
            biasDeltas.push_back(deltas[i].rowMean());
            // Debug
            // std::cout << "\n========Kronecker========" << std::endl;
            // (activationOutputs[i + 1].transpose() * deltas[i] * scaleFactor).display();
            // std::cout << "========Weights========" << std::endl;
            // (weights[i]).display();
            // std::cout << "========Weight Deltas========" << std::endl;
            // (weightDeltas[i]).display();
            // std::cout << std::endl;
            // End Debug
        }
        for (int i = 0; i < weightDeltas.size(); ++i) {
            // Debug
            std::cout << "========Weights " << i << " ========" << std::endl;
            (weights[i]).display();
            std::cout << "========Weight Deltas " << i << " ========" << std::endl;
            weightDeltas[i].display();
            std::cout << "========Biases " << i << " ========" << std::endl;
            (biases[i]).display();
            std::cout << "========Bias Deltas " << i << " ========" << std::endl;
            biasDeltas[i].display();
            // End Debug
            weights[i] = weights[i] - (learningRate * weightDeltas[i]);
            biases[i] = biases[i] - (learningRate * biasDeltas[i]);
        }
    }

    template <typename T>
    const math::Matrix<T>& NeuralNetwork<T>::feedForward(const math::Matrix<T>& input) {
        return getLayerOutput(input, numLayers);
    }

    template <typename T>
    const math::Matrix<T>& NeuralNetwork<T>::getLayerOutput(const math::Matrix<T>& input, int layerNum) {
        outputs[0] = input;
        activationOutputs[0] = outputs[0];
        for (int i = 1; i < layerNum; ++i) {
            // outputs[i] = (activationOutputs[i - 1] * weights[i - 1]).addVector(biases[i - 1]);
            // if (outputs[i].numRows() == 1) {
            //     outputs[i] = math::Matrix<T>(input.numRows(), outputs[i].numColumns());
            //     activationOutputs[i] = math::Matrix<T>(input.numRows(), outputs[i].numColumns());
            // }

            outputs[i] = math::Matrix<T>(input.numRows(), weights[i - 1].numColumns());
            dim3 blocks(std::ceil(outputs[i].numRows() / (float) BLOCK_DIM), std::ceil(outputs[i].numColumns() / (float) BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            activationOutputs[i] = math::Matrix<T>(outputs[i].numRows(), outputs[i].numColumns());

            computeFeedForward<<<blocks, threads>>>(activationOutputs[i - 1].data(), weights[i - 1].data(), biases[i - 1].data(),
                activationOutputs[i - 1].numRows(), activationOutputs[i - 1].numColumns(), weights[i - 1].numColumns(),
                activationOutputs[i - 1].size(), weights[i - 1].size(), outputs[i].data(), activationOutputs[i].data(), activationFunc);
            cudaDeviceSynchronize();
            // Save this output as input to the next layer.
            // computeActivationFunction(i);
            // Debug
            // std::cout << "Layer " << i - 1 << std::endl;
            // std::cout << "========Input========" << std::endl;
            // activationOutputs[i - 1].display();
            // std::cout << "========Weights========" << std::endl;
            // weights[i - 1].display();
            // std::cout << "\n========Biases========" << std::endl;
            // biases[i - 1].display();
            // std::cout << "\n========Input * Weight========" << std::endl;
            // (activationOutputs[i - 1] * weights[i - 1]).display();
            // std::cout << "\n========Input * Weight + Bias========" << std::endl;
            // (activationOutputs[i - 1] * weights[i - 1]).addVector(biases[i - 1]).display();
            // std::cout << "\n========Output========" << std::endl;
            // outputs[i].display();
            // std::cout << "\n========Activated Output========" << std::endl;
            // activationOutputs[i].display();
            // std::cout << std::endl;
            // End Debug
            // if (i + 1 < layerNum) {
            //     outputs[i + 1] = activationOutputs[i];
            // }
        }
        return activationOutputs.back();
    }

    template <typename T>
    void NeuralNetwork<T>::saveWeights(const std::string& filePath) {
        std::ofstream saveFile(filePath);
        if (saveFile.is_open()) {
            saveFile << numLayers << std::endl;
            for (int i = 0; i < numLayers - 1; ++i) {
                weights[i].write(saveFile);
                biases[i].write(saveFile);
            }
            saveFile.close();
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template <typename T>
    void NeuralNetwork<T>::loadWeights(const std::string& filePath) {
        std::ifstream saveFile(filePath);
        if (saveFile.is_open()) {
            std::string temp;
            saveFile >> temp;
            numLayers = std::stoi(temp);
            weights = std::vector<math::Matrix<T> > (numLayers - 1);
            biases = std::vector<math::Matrix<T> > (numLayers - 1);
            deltas = std::vector<math::Matrix<T> > (numLayers - 1);
            outputs = std::vector<math::Matrix<T> > (numLayers);
            activationOutputs = std::vector<math::Matrix<T> > (numLayers);
            for (int i = 0; i < numLayers - 1 && !saveFile.eof(); ++i) {
                weights[i].read(saveFile);
                biases[i].read(saveFile);
            }
            inputSize = weights[0].numRows();
            saveFile.close();
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template <typename T>
    int NeuralNetwork<T>::getNumLayers() const {
        return numLayers;
    }

    template <typename T>
    NeuralNetwork<T>::aFunc& NeuralNetwork<T>::activationFunction() {
        return activationFunc;
    }

    template <typename T>
    NeuralNetwork<T>::cFunc& NeuralNetwork<T>::costFunction() {
        return costFunc;
    }

    template <typename T>
    const NeuralNetwork<T>::aFunc& NeuralNetwork<T>::activationFunction() const {
        return activationFunc;
    }

    template <typename T>
    const NeuralNetwork<T>::cFunc& NeuralNetwork<T>::costFunction() const {
        return costFunc;
    }

    template <typename T>
    void NeuralNetwork<T>::initializeWeights() {
        for (int i = 0; i < numLayers - 1; ++i) {
            double weightRange = 2 / sqrt(weights[i].numRows());
            if (activationFunc == RELU) {
                weights[i] = math::Matrix<T>::randomUniformLike(weights[i], 0, weightRange);
                biases[i] = math::Matrix<T>::randomNormalLike(biases[i], 0, weightRange).template applyFunction<abs>();
            } else {
                weights[i] = math::Matrix<T>::randomUniformLike(weights[i], -weightRange, weightRange);
                biases[i] = math::Matrix<T>::randomNormalLike(biases[i], 0, weightRange);
            }
        }
    }

    template <typename T>
    math::Matrix<T> NeuralNetwork<T>::computeActivationFunctionDerivative(int layerNum) const {
        switch (activationFunction()) {
            case SIGMOID:
                return activationOutputs[layerNum].hadamard(1 - activationOutputs[layerNum]);
            case ANALYTIC: {
                math::Matrix<T> output = activationOutputs[layerNum];
                dim3 blocks(std::ceil(output.size() / (float) THREADS_PER_BLOCK));
                dim3 threads(THREADS_PER_BLOCK);
                activationFunctionSigmoid<<<blocks, threads>>>(outputs[layerNum].data(), output.size(), output.data());
                cudaDeviceSynchronize();
                return output;
            }
            case RELU: {
                math::Matrix<T> output = activationOutputs[layerNum];
                dim3 blocks(std::ceil(output.size() / (float) THREADS_PER_BLOCK));
                dim3 threads(THREADS_PER_BLOCK);
                activationFunctionRELUDerivative<<<blocks, threads>>>(outputs[layerNum].data(), output.size(), output.data());
                cudaDeviceSynchronize();
                return output;
            }
        }
        return T();
    }

    template <typename T>
    math::Matrix<T> NeuralNetwork<T>::cost(const math::Matrix<T>& output, const math::Matrix<T>& expectedOutput) {
        math::Matrix<T> error;
        switch (costFunction()) {
            case MSE:
                error = expectedOutput - output;
                error = (error.dot(error) * 0.5);
                break;
        }
        return error;
    }

    template <typename T>
    math::Matrix<T> NeuralNetwork<T>::costDerivative(const math::Matrix<T>& output, const math::Matrix<T>& expectedOutput) {
        switch (costFunction()) {
            case MSE:
                return output - expectedOutput;
        }
        return T();
    }

    template class NeuralNetwork<float>;
    template class NeuralNetwork<double>;

}
