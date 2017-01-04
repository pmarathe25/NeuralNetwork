BUILDDIR = build/
BINDIR = bin/
INCLUDEDIR = include/
OBJS = $(BUILDDIR)/NeuralNetwork.o
TESTOBJS = $(BUILDDIR)/NeuralNetworkTest.o
EXECOBJS =
LIBDIR = $(CURDIR)/lib/
TESTDIR = test/
SRCDIR = src/
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 -I$(INCLUDEDIR)
LFLAGS = -shared -Wno-deprecated-gpu-targets
TESTLFLAGS = -Wno-deprecated-gpu-targets
EXECLFLAGS = -Wno-deprecated-gpu-targets

$(LIBDIR)/NeuralNetwork/libneuralnetwork.so: $(OBJS)
	$(CXX) $(LFLAGS) $(OBJS) -o $(LIBDIR)/NeuralNetwork/libneuralnetwork.so

$(BUILDDIR)/NeuralNetwork.o: $(INCLUDEDIR)/NeuralNetwork/NeuralNetwork.hpp $(INCLUDEDIR)/Math/Matrix.hpp $(SRCDIR)/NeuralNetwork.cu \
	$(SRCDIR)/NeuralNetworkCUDAFunctions.cu $(SRCDIR)/NeuralNetworkCPUFunctions.cpp
	$(CXX) $(CFLAGS) $(SRCDIR)/NeuralNetwork.cu -o $(BUILDDIR)/NeuralNetwork.o

$(TESTDIR)/NeuralNetworkTest: $(TESTOBJS) $(LIBDIR)/Math/libmath.so $(LIBDIR)/NeuralNetwork/libneuralnetwork.so
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) $(LIBDIR)/Math/libmath.so $(LIBDIR)/NeuralNetwork/libneuralnetwork.so -o $(TESTDIR)/NeuralNetworkTest

$(BUILDDIR)/NeuralNetworkTest.o: $(TESTDIR)/NeuralNetworkTest.cpp $(INCLUDEDIR)/NeuralNetwork/NeuralNetwork.hpp \
	$(LIBDIR)/Math/libmath.so $(LIBDIR)/NeuralNetwork/libneuralnetwork.so
	$(CXX) $(CFLAGS) $(TESTDIR)/NeuralNetworkTest.cpp -o $(BUILDDIR)/NeuralNetworkTest.o

clean:
	rm $(OBJS) $(TESTOBJS) $(LIBDIR)/NeuralNetwork/libneuralnetwork.so

test: $(TESTDIR)/NeuralNetworkTest
	$(TESTDIR)/NeuralNetworkTest

exec:
