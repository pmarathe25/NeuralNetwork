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

            Minibatch(Matrix data, Matrix expectedOutput) {
                this -> data = data;
                this -> expectedOutput = expectedOutput;
            }

            Minibatch(const std::string& filePath) {
                load(filePath);
            }

            // File I/O
            void save(const std::string& filePath) const {
                std::ofstream saveFile(filePath);
                if (saveFile.is_open()) {
                    save(saveFile);
                } else {
                    throw std::invalid_argument("Could not open file.");
                }
            }

            void save(std::ofstream& saveFile) const {
                data.save(saveFile);
                expectedOutput.save(saveFile);
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
                expectedOutput.load(loadFile);
            }

            // Accessors
            inline const Matrix& getData() const {
                return data;
            }

            inline const Matrix& getExpectedOutput() const {
                return expectedOutput;
            }

            static bool isMinibatchFile(const std::string& fileName) {
                return fileName.rfind(MINIBATCH_EXTENSION) + MINIBATCH_EXTENSION.size() == fileName.size();
            }

        private:
            Matrix data, expectedOutput;
    };
} /* namespace ai */

#endif
