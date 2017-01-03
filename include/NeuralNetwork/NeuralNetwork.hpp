#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "Math/Matrix.hpp"

namespace ai {
    template <typename T>
    class NeuralNetwork {
        public:
            enum aFunc {
                SIGMOID = 0,
            };
            enum cFunc {
                MSE = 0,
            };
            // Constructors.
            NeuralNetwork(aFunc func = SIGMOID, cFunc func2 = MSE);
            NeuralNetwork(std::vector<int> layers, aFunc func = SIGMOID, cFunc func2 = MSE);
            // Usage methods.
            const math::Matrix<T>& feedForward(const std::vector<T>& input);
            const math::Matrix<T>& getLayerOutput(const std::vector<T>& input, int layerNum);
            // File I/O.
            void saveWeights(const std::string& filePath);
            void loadWeights(const std::string& filePath);
            // Setter functions.
            void setActivationFunction(aFunc func);
            void setCostFunction(cFunc func);
        private:
            std::vector<math::Matrix<T> > weights;
            std::vector<math::Matrix<T> > biases;
            // Cache previous output.
            math::Matrix<T> output;
            // Other data members.
            aFunc activationFunction;
            cFunc costFunction;
            int inputSize;
            // Initialization.
            void initializeWeights();
            // Helper functions.
            void applyActivationFunction(math::Matrix<T>& layer);
            math::Matrix<T> cost(const math::Matrix<T>& output, const math::Matrix<T>& expectedOutput);
    };
}

#endif
