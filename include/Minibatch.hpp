#ifndef MINIBATCH_H
#define MINIBATCH_H
#include <string>
#include <fstream>

namespace ai {
    const std::string& MINIBATCH_EXTENSION = ".minibatch";

    template <typename Matrix>
    class Minibatch {
        public:
            Minibatch() { }

            Minibatch(Matrix data, Matrix labels) {
                this -> data = data;
                this -> labels = labels;
            }

            Minibatch(const std::string& filePath) {
                load(filePath);
            }

            // File I/O
            void save(const std::string& filePath) {
                std::ofstream saveFile(filePath);
                if (saveFile.is_open()) {
                    save(saveFile);
                } else {
                    throw std::invalid_argument("Could not open file.");
                }
            }

            void save(std::ofstream& saveFile) {
                data.save(saveFile);
                labels.save(saveFile);
            }

            void load(const std::string& filePath) {
                std::ifstream loadFile(filePath);
                if (loadFile.is_open()) {
                    load(loadFile);
                } else {
                    throw std::invalid_argument("Could not open file.");
                }
            }

            void load(std::ifstream& loadFile) {
                data.load(loadFile);
                labels.load(loadFile);
            }

            // Accessors
            inline const Matrix& getData() const {
                return data;
            }

            inline const Matrix& getLabels() const {
                return labels;
            }

            static bool isMinibatchFile(const std::string& fileName) {
                return fileName.rfind(MINIBATCH_EXTENSION) + MINIBATCH_EXTENSION.size() == fileName.size();
            }

        private:
            Matrix data, labels;
    };
} /* namespace ai */

#endif
