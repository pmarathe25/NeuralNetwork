#include "NeuralNetwork/NeuralNetwork.hpp"
#include <math.h>
#include <fstream>
#include <iostream>
// Include Neural Network functions.
#include "NeuralNetworkCUDAFunctions.cu"
#include "NeuralNetworkCPUFunctions.cpp"

namespace ai {
    template <typename T>
    NeuralNetwork<T>::NeuralNetwork(aFunc func, cFunc func2) {
        activationFunction = func;
        costFunction = func2;
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
        deltas.back() = costDerivative(activationOutputs.back(), desiredOutput).hadamard(applyActivationFunctionDerivative(outputs.back())).rowMean();
        for (int i = numLayers - 3; i >= 0; --i) {
            // Debug
            std::cout << "Layer " << i + 1 << std::endl;
            std::cout << "========Weights========" << std::endl;
            math::display(weights[i + 1]);
            std::cout << "\n========Deltas========" << std::endl;
            math::display(deltas[i + 1].transpose());
            std::cout << "\n========Weights * (Deltas Transpose) Transpose========" << std::endl;
            math::display((weights[i + 1] * deltas[i + 1].transpose()).transpose());
            std::cout << "\n========Output After Activation Derivative========" << std::endl;
            math::display(applyActivationFunctionDerivative(outputs[i + 1]));
            std::cout << std::endl;
            deltas[i] = ((weights[i + 1] * deltas[i + 1].transpose()).transpose()).hadamard(applyActivationFunctionDerivative(outputs[i + 1].rowMean()));
        }
        // Gradients for each layers.
        std::vector<math::Matrix<T> > weightDeltas;
        std::vector<math::Matrix<T> > biasDeltas;
        // Compute gradients.
        for (int i = 0; i < deltas.size(); ++i) {
            // Debug
            // std::cout << "Layer " << i + 1 << std::endl;
            // std::cout << "========Outputs transpose========" << std::endl;
            // math::display(activationOutputs[i].transpose());
            // std::cout << "\n========Deltas========" << std::endl;
            // math::display(deltas[i]);
            // std::cout << "\n========Kronecker========" << std::endl;
            // math::display((activationOutputs[i].transpose()).kronecker(deltas[i]));
            // std::cout << "========Weights========" << std::endl;
            // math::display(weights[i]);
            // std::cout << std::endl;
            // TODO: Compute average activationOutputs[i] and deltas[i] by performig a row average.
            weightDeltas.push_back((activationOutputs[i].rowMean().transpose()).kronecker(deltas[i]));
        }
        biasDeltas = deltas;
        for (int i = 0; i < biases.size(); ++i) {
            weights[i] = weights[i] + (learningRate * weightDeltas[i]);
            biases[i] = biases[i] - (learningRate * biasDeltas[i]);
        }
    }

    template <typename T>
    const math::Matrix<T>& NeuralNetwork<T>::feedForward(const math::Matrix<T>& input) {
        return getLayerOutput(input, numLayers);
    }

    template <typename T>
    const math::Matrix<T>& NeuralNetwork<T>::getLayerOutput(const math::Matrix<T>& input, int layerNum) {
        if (layerNum > numLayers) {
            throw std::invalid_argument("Layer does not exist.");
        }
        outputs[0] = input;
        activationOutputs[0] = applyActivationFunction(outputs[0]);
        for (int i = 1; i < layerNum; ++i) {
            outputs[i] = activationOutputs[i - 1] * weights[i - 1] + biases[i - 1];
            // Save this output as input to the next layer.
            activationOutputs[i] = applyActivationFunction(outputs[i]);
            if (i + 1 < layerNum) {
                outputs[i + 1] = activationOutputs[i];
            }
            // Debug
            // std::cout << "Layer " << i - 1 << std::endl;
            // std::cout << "========Weights========" << std::endl;
            // math::display(weights.at(i - 1));
            // std::cout << "\n========Biases========" << std::endl;
            // math::display(biases.at(i - 1));
            // std::cout << "\n========Output After Activation========" << std::endl;
            // math::display(activationOutputs[i]);
            // std::cout << std::endl;
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
    void NeuralNetwork<T>::activationFunction() {
        return actFunc;
    }

    template <typename T>
    void NeuralNetwork<T>::costFunction() {
        return costFunc;
    }

    template <typename T>
    void NeuralNetwork<T>::initializeWeights() {
        double weightRange = 2 / sqrt(inputSize);
        for (int i = 0; i < numLayers - 1; ++i) {
            weights[i].randomizeUniform(-weightRange, weightRange);
            biases[i].randomizeNormal();
        }
    }

    template <typename T>
    math::Matrix<T> NeuralNetwork<T>::applyActivationFunction(const math::Matrix<T>& layer) const {
        math::Matrix<T> out = math::Matrix<T>(layer.numRows(), layer.numColumns());
        if (layer.size() < CPU_SATURATION_LIMIT) {
            switch (activationFunction()) {
                case SIGMOID:
                    out = applySigmoidCPU(layer);
                    return out;
            }
        } else {
            int rawSize = layer.size();
            // Initialize device copies.
            T* dev_mat;
            // // Allocate memory for device copies.
            cudaMalloc((void**)&dev_mat, rawSize * sizeof(T));
            // Copy inputs to device.
            cudaMemcpy(dev_mat, layer.data(), rawSize * sizeof(T), cudaMemcpyHostToDevice);
            // // Launch kernel where numThreads = size of matrix.
            dim3 blocks(std::ceil(rawSize / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            switch (activationFunction) {
                case SIGMOID:
                activationFunctionSigmoid<<<blocks, threads>>>(dev_mat, layer.size());
                break;
            }
            // // Get result.
            cudaMemcpy(out.data(), dev_mat, rawSize * sizeof(T), cudaMemcpyDeviceToHost);
            // // Free memory.
            cudaFree(dev_mat);
        }
        return out;
    }

    template <typename T>
    math::Matrix<T> NeuralNetwork<T>::applyActivationFunctionDerivative(const math::Matrix<T>& layer) const {
        math::Matrix<T> out = math::Matrix<T>(layer.numRows(), layer.numColumns());
        switch (activationFunction()) {
            case SIGMOID:
                out = applyActivationFunction(layer);
                out = out.hadamard(1 - out);

                break;
        }
        return out;
    }

    template <typename T>
    math::Matrix<T> NeuralNetwork<T>::cost(const math::Matrix<T>& output, const math::Matrix<T>& expectedOutput) {
        math::Matrix<T> error;
        switch (costFunction()) {
            case MSE:
                error = expectedOutput - output;
                error = error.dot(error) * (1 / (2.0 * inputSize));
                break;
        }
        return error;
    }

    template <typename T>
    math::Matrix<T> NeuralNetwork<T>::costDerivative(const math::Matrix<T>& output, const math::Matrix<T>& expectedOutput) {
        math::Matrix<T> error;
        switch (costFunction()) {
            case MSE:
                error = output - expectedOutput;
                break;
        }
        return error;
    }

    template class NeuralNetwork<float>;
    template class NeuralNetwork<double>;

}
