#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H
#include "NeuralNetwork/Layer/Layer.hpp"

#include <iostream>

namespace ai {
    template <typename T, activationFunction aFunc>
    class FullyConnectedLayer : Layer<T> {
    public:
        FullyConnectedLayer(int inputSize, int outputSize) {
            this -> weights = math::Matrix<T>(inputSize, outputSize);
            this -> biases = math::Matrix<T>(1, outputSize);
            initializeWeights();
        }

        FullyConnectedLayer(math::Matrix<T> weights, math::Matrix<T> biases) {
            this -> weights = weights;
            this -> biases = biases;
        }

        math::Matrix<T> feedForward(const math::Matrix<T>& input) {
            switch (aFunc) {
                case SIGMOID:
                    return (input * weights).addVector(biases).template applyFunction<sigmoid>();
                case ANALYTIC:
                    return (input * weights).addVector(biases).template applyFunction<analytic>();
                case RELU:
                    return (input * weights).addVector(biases).template applyFunction<relu>();
                default:
                    return (input * weights).addVector(biases).template applyFunction<sigmoid>();
            }
        }

        void initializeWeights() {
            double weightRange = 2 / sqrt(weights.numRows());
            if (aFunc == RELU) {
                weights.randomizeUniform(0, weightRange);
                biases.randomizeNormal(0, weightRange).template applyFunction<abs>();
            } else {
                weights.randomizeUniform(-weightRange, weightRange);
                biases.randomizeNormal(0, weightRange);
            }
        }
    private:
        math::Matrix<T> weights, biases;
    };
}

#endif
