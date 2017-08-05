#ifndef LAYER_H
#define LAYER_H
#include <fstream>

namespace ai {
    template <typename T>
    __device__ T sigmoid(T in) {
        return 1 / (1 + exp(-in));
    }

    template <typename T>
    __device__ T sigmoid_prime(T in) {
        T sig = sigmoid(in);
        return sig * (1 - sig);
    }

    template <typename T>
    __device__ T analytic(T in) {
        return log(1 + exp(in));
    }

    template <typename T>
    __device__ T relu(T in) {
        return (in < 0) ? 0 : in;
    }

    template <typename T>
    __device__ T relu_prime(T in) {
        return (in < 0) ? 0 : 1;
    }

    template <typename T>
    __device__ T leakyRelu(T in) {
        return (in < 0) ? 0.01 * in : in;
    }

    template <typename T>
    __device__ T leakyRelu_prime(T in) {
        return (in < 0) ? 0.01 : 1;
    }

    enum activationFunction {
        SIGMOID = 0,
        ANALYTIC,
        RELU,
    };

    template <typename Matrix>
    class Layer {
    public:
        virtual void save(std::ofstream& saveFile) const = 0;
        virtual void load(std::ifstream& saveFile) = 0;
        virtual Matrix feedForward(const Matrix& input) const = 0;
        virtual Matrix getWeightedOutput(const Matrix& input) const = 0;
        virtual Matrix activate(const Matrix& weightedOutput) const = 0;
        virtual Matrix computeDeltas(const Matrix& intermediateDeltas, const Matrix& weightedOutput) const = 0;
        virtual Matrix backpropagate(const Matrix& deltas) = 0;
    };
} /* namespace ai */

#endif
