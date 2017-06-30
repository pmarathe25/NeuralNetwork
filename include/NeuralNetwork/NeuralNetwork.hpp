#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "Math/Matrix.hpp"

namespace ai {
    template <typename T>
    class NeuralNetwork {
        public:
            enum aFunc {
                SIGMOID = 0,
                ANALYTIC,
            };
            enum cFunc {
                MSE = 0,
            };
            // Constructors.
            NeuralNetwork(aFunc func = SIGMOID, cFunc func2 = MSE);
            NeuralNetwork(std::vector<int> layers, aFunc func = ANALYTIC, cFunc func2 = MSE);
            // Usage methods.
            void train(const math::Matrix<T>& input, const math::Matrix<T>& desiredOutput, T learningRate);
            const math::Matrix<T>& feedForward(const math::Matrix<T>& input);
            const math::Matrix<T>& getLayerOutput(const math::Matrix<T>& input, int layerNum);
            // File I/O.
            void saveWeights(const std::string& filePath);
            void loadWeights(const std::string& filePath);
            // Getter functions.
            int getNumLayers() const;
            // Setter functions.
            aFunc& activationFunction();
            cFunc& costFunction();
            const aFunc& activationFunction() const;
            const cFunc& costFunction() const;
        private:
            std::vector<math::Matrix<T> > weights;
            std::vector<math::Matrix<T> > biases;
            // Cache previous outputs.
            std::vector<math::Matrix<T> > outputs;
            std::vector<math::Matrix<T> > activationOutputs;
            // Error for each layer.
            std::vector<math::Matrix<T> > deltas;
            // Other data members.
            aFunc activationFunc;
            cFunc costFunc;
            int inputSize, numLayers;
            // Initialization.
            void initializeWeights();
            // Helper functions.
            math::Matrix<T> computeActivationFunctionDerivative(int layerNum) const;
            math::Matrix<T> cost(const math::Matrix<T>& output, const math::Matrix<T>& expectedOutput);
            math::Matrix<T> costDerivative(const math::Matrix<T>& output, const math::Matrix<T>& expectedOutput);
    };
}

#endif
