#ifndef NEURAL_NETWORK_SAVER_H
#define NEURAL_NETWORK_SAVER_H
#include "NeuralNetwork.hpp"
#include <unordered_set>
#include <tuple>
#include <string>
#include <fstream>

namespace StealthAI {
    class NeuralNetworkSaver {
        public:
            NeuralNetworkSaver() { }

            template <typename Matrix, typename... Layers>
            static void save(NeuralNetwork<Matrix, Layers...>& network, const std::string& filePath) {
                std::ofstream saveFile(filePath);
                if (saveFile.is_open()) {
                    saveUnpacker(saveFile, typename sequenceGenerator<sizeof...(Layers)>::type(), network.getLayers());
                } else {
                    throw std::invalid_argument("Could not open file.");
                }
            }

            template <typename Matrix, typename... Layers>
            static void load(NeuralNetwork<Matrix, Layers...>& network, const std::string& filePath) {
                std::ifstream loadFile(filePath);
                if (loadFile.is_open()) {
                    try {
                        loadUnpacker(loadFile, typename sequenceGenerator<sizeof...(Layers)>::type(), network.getLayers());
                    } catch (const std::exception& e) {
                        throw std::invalid_argument("Network mismatch during load from file.");
                    }
                } else {
                    throw std::invalid_argument("Could not open file.");
                }
            }

        private:
            // Weight saving unpacker.
            template <int... S, typename... Layers>
            static inline void saveUnpacker(std::ofstream& saveFile, sequence<S...>, std::tuple<Layers&...> layers) {
                std::unordered_set<void*> savedLayers;
                saveRecursive(saveFile, savedLayers, std::get<S>(layers)...);
            }

            // Weight saving base case.
            template <typename BackLayer>
            static inline void saveRecursive(std::ofstream& saveFile, std::unordered_set<void*>& savedLayers, BackLayer& backLayer) {
                if (!savedLayers.count(&backLayer)) {
                    backLayer.save(saveFile);
                    savedLayers.insert(&backLayer);
                }
            }

            // Weight saving recursion.
            template <typename FrontLayer, typename... BackLayers>
            static inline void saveRecursive(std::ofstream& saveFile, std::unordered_set<void*>& savedLayers, FrontLayer& frontLayer, BackLayers&... otherLayers) {
                // Save this layer if it has not been already then do the others.
                if (!savedLayers.count(&frontLayer)) {
                    frontLayer.save(saveFile);
                    savedLayers.insert(&frontLayer);
                }
                saveRecursive(saveFile, savedLayers, otherLayers...);
            }

            // Weight loading unpacker.
            template <int... S, typename... Layers>
            static inline void loadUnpacker(std::ifstream& loadFile, sequence<S...>, std::tuple<Layers&...> layers) {
                std::unordered_set<void*> loadedLayers;
                loadRecursive(loadFile, loadedLayers, std::get<S>(layers)...);
            }

            // Weight loading base case.
            template <typename BackLayer>
            static inline void loadRecursive(std::ifstream& loadFile, std::unordered_set<void*>& loadedLayers, BackLayer& backLayer) {
                if (!loadedLayers.count(&backLayer)) {
                    backLayer.load(loadFile);
                    loadedLayers.insert(&backLayer);
                }
            }

            // Weight loading recursion.
            template <typename FrontLayer, typename... BackLayers>
            static inline void loadRecursive(std::ifstream& loadFile, std::unordered_set<void*>& loadedLayers, FrontLayer& frontLayer, BackLayers&... otherLayers) {
                // Load this layer if it hasn't been already then do the others.
                if (!loadedLayers.count(&frontLayer)) {
                    frontLayer.load(loadFile);
                    loadedLayers.insert(&frontLayer);
                }
                loadRecursive(loadFile, loadedLayers, otherLayers...);
            }

    };
} /* namespace StealthAI */

#endif
