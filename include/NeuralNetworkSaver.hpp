#ifndef NEURAL_NETWORK_SAVER_H
#define NEURAL_NETWORK_SAVER_H
#include "NeuralNetwork.hpp"
#include <string>
#include <fstream>

namespace ai {
    template <typename Matrix, typename... Layers>
    class NeuralNetworkSaver {
        public:
            NeuralNetworkSaver(NeuralNetwork<Matrix, Layers...>& target) : layers(target.getLayers()) { }

            void save(const std::string& filePath) {
                std::ofstream saveFile(filePath);
                if (saveFile.is_open()) {
                    int numLayers = sizeof...(Layers);
                    saveFile.write(reinterpret_cast<char*>(&numLayers), sizeof numLayers);
                    saveUnpacker(saveFile, typename sequenceGenerator<sizeof...(Layers)>::type());
                } else {
                    throw std::invalid_argument("Could not open file.");
                }
            }

            void load(const std::string& filePath) {
                std::ifstream saveFile(filePath);
                if (saveFile.is_open()) {
                    // Check if it has the correct number of layers.
                    int numLayers = sizeof...(Layers);
                    saveFile.read(reinterpret_cast<char*>(&numLayers), sizeof numLayers);
                    if (numLayers != sizeof...(Layers)) {
                        throw std::invalid_argument("Network depth mismatch during load from file.");
                    }
                    loadUnpacker(saveFile, typename sequenceGenerator<sizeof...(Layers)>::type());
                } else {
                    throw std::invalid_argument("Could not open file.");
                }
            }

        private:
            // Weight saving unpacker.
            template <int... S>
            inline void saveUnpacker(std::ofstream& saveFile, sequence<S...>) {
                saveRecursive(saveFile, std::get<S>(layers)...);
            }

            // Weight saving base case.
            template <typename BackLayer>
            inline void saveRecursive(std::ofstream& saveFile, BackLayer& backLayer) {
                backLayer.save(saveFile);
            }

            // Weight saving recursion.
            template <typename FrontLayer, typename... BackLayers>
            inline void saveRecursive(std::ofstream& saveFile, FrontLayer& frontLayer, BackLayers&... otherLayers) {
                // Save this layer then do the others.
                frontLayer.save(saveFile);
                saveRecursive(saveFile, otherLayers...);
            }

            // Weight loading unpacker.
            template <int... S>
            inline void loadUnpacker(std::ifstream& saveFile, sequence<S...>) {
                loadRecursive(saveFile, std::get<S>(layers)...);
            }

            // Weight loading base case.
            template <typename BackLayer>
            inline void loadRecursive(std::ifstream& saveFile, BackLayer& backLayer) {
                backLayer.load(saveFile);
            }

            // Weight loading recursion.
            template <typename FrontLayer, typename... BackLayers>
            inline void loadRecursive(std::ifstream& saveFile, FrontLayer& frontLayer, BackLayers&... otherLayers) {
                // Save this layer then do the others.
                frontLayer.load(saveFile);
                loadRecursive(saveFile, otherLayers...);
            }

            std::tuple<Layers&...> layers;
    };
} /* namespace ai */

#endif
