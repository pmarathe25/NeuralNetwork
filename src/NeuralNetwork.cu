#include "NeuralNetwork/NeuralNetwork.hpp"
#include <math.h>
#include <fstream>
#include <iostream>
// Include Neural Network functions.
#include "NeuralNetworkCUDAFunctions.cu"

namespace ai {
    template <typename T>
    NeuralNetwork<T>::NeuralNetwork(aFunc func, cFunc func2) {
        activationFunction = func;
        costFunction = func2;
    }

    template <typename T>
    NeuralNetwork<T>::NeuralNetwork(std::vector<int> layers, aFunc func, cFunc func2) : NeuralNetwork(func, func2) {
        // Initialize weights and biases.
        for (int i = 0; i < layers.size() - 1; ++i) {
            weights.push_back(math::Matrix<T>(layers[i], layers[i + 1]));
            biases.push_back(math::Matrix<T>(1, layers[i + 1]));
        }
        // Initialize output layer cache.
        output = math::Matrix<T>(1, layers.back());
        // Keep track of input size.
        inputSize = layers.front();
        // Randomize weights and biases.
        initializeWeights();
    }

    template <typename T>
    const math::Matrix<T>& NeuralNetwork<T>::feedForward(const math::Matrix<T>& input) {
        return getLayerOutput(input, weights.size());
    }

    template <typename T>
    const math::Matrix<T>& NeuralNetwork<T>::getLayerOutput(const math::Matrix<T>& input, int layerNum) {
        output = input;
        if (layerNum > weights.size()) {
            throw std::invalid_argument("Layer does not exist.");
        }
        for (int i = 0; i < layerNum; ++i) {
            output = output * weights[i] + biases[i];
            applyActivationFunction(output);
            // Debug.
            // std::cout << "Layer " << i << std::endl;
            // std::cout << "========Weights========" << std::endl;
            // math::display(weights.at(i));
            // std::cout << "\n========Biases========" << std::endl;
            // math::display(biases.at(i));
            // std::cout << "\n========Output After Activation========" << std::endl;
            // math::display(output);
            // std::cout << std::endl;
        }
        return output;
    }

    template <typename T>
    void NeuralNetwork<T>::saveWeights(const std::string& filePath) {
        std::ofstream saveFile(filePath);
        if (saveFile.is_open()) {
            saveFile << weights.size() << std::endl;
            for (int i = 0; i < weights.size(); ++i) {
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
            int numWeights = std::stoi(temp);
            weights = std::vector<math::Matrix<T> > (numWeights);
            biases = std::vector<math::Matrix<T> > (numWeights);
            for (int i = 0; i < weights.size() && !saveFile.eof(); ++i) {
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
    void NeuralNetwork<T>::setActivationFunction(aFunc func) {
        activationFunction = func;
    }

    template <typename T>
    void NeuralNetwork<T>::setCostFunction(cFunc func) {
        costFunction = func;
    }

    template <typename T>
    void NeuralNetwork<T>::initializeWeights() {
        double weightRange = 1 / sqrt(inputSize);
        for (int i = 0; i < weights.size(); ++i) {
            weights[i].randomizeUniform(-weightRange, weightRange);
            biases[i].randomizeNormal();
        }
    }

    template <typename T>
    void NeuralNetwork<T>::applyActivationFunction(math::Matrix<T>& layer) {
        int rawSize = layer.sizeRaw();
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
                if (layer.isVector()) {
                    activationFunctionVectorSigmoid<<<blocks, threads>>>(dev_mat, layer.size());;
                } else {
                    activationFunctionSigmoid<<<blocks, threads>>>(dev_mat);;
                }
                break;
        }
        // // Get result.
        cudaMemcpy(layer.data(), dev_mat, rawSize * sizeof(T), cudaMemcpyDeviceToHost);
        // // Free memory.
        cudaFree(dev_mat);
    }

    template <typename T>
    math::Matrix<T> NeuralNetwork<T>::cost(const math::Matrix<T>& output, const math::Matrix<T>& expectedOutput) {
        math::Matrix<T> error;
        switch (costFunction) {
            case MSE:
                error = expectedOutput - output;
                error = error.dot(error) * (1 / (2.0 * inputSize));
                break;
        }
        return error;
    }

    template class NeuralNetwork<float>;
    template class NeuralNetwork<double>;

}
