#ifndef LAYER_H
#define LAYER_H
#include "Math/Matrix.hpp"


namespace ai {
    namespace NeuralNetwork {
        template <typename T>
        class Layer {
        public:
            Layer() {}
            virtual math::Matrix<T> feedForward(const math::Matrix<T>& input) = 0;
            virtual void initializeWeights() = 0;
        };
    }
}

#endif
