BUILDDIR = build/
BINDIR = bin/
LIBDIR = $(CURDIR)/lib/
LIBMATRIX = ~/C++/Math/lib/libmatrix.so
LIBMATH = ~/C++/Math/lib/libmath.so
LIBMATHINCLUDEPATH = /home/pranav/C++/Math/include/
LIBNEURALNETWORK = lib/libneuralnetwork.so
LIBS = $(LIBMATRIX) $(LIBMATH) $(LIBNEURALNETWORK)
INCLUDEDIR = -Iinclude/ -I$(LIBMATHINCLUDEPATH)
OBJS = $(addprefix $(BUILDDIR)/, NeuralNetwork.o)
TESTOBJS = $(BUILDDIR)/NeuralNetworkTest.o
EXECOBJS =
TESTDIR = test/
SRCDIR = src/
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 $(INCLUDEDIR)
LFLAGS = -shared -Wno-deprecated-gpu-targets
TESTLFLAGS = -Wno-deprecated-gpu-targets

$(LIBDIR)/libneuralnetwork.so: $(OBJS)
	$(CXX) $(LFLAGS) $(OBJS) -o $(LIBDIR)/libneuralnetwork.so

$(TESTDIR)/NeuralNetworkTest: $(TESTOBJS) $(LIBMATRIX)
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) $(LIBS) -o $(TESTDIR)/NeuralNetworkTest

$(BUILDDIR)/NeuralNetworkTest.o: $(TESTDIR)/NeuralNetworkTest.cu include/NeuralNetwork/NeuralNetwork.hpp include/NeuralNetwork/Layer/Layer.hpp \
	include/NeuralNetwork/Layer/FullyConnectedLayer.hpp $(LIBDIR)/libneuralnetwork.so
	$(CXX) $(CFLAGS) $(TESTDIR)/NeuralNetworkTest.cu -o $(BUILDDIR)/NeuralNetworkTest.o

$(BUILDDIR)/NeuralNetwork.o: include/NeuralNetwork/NeuralNetwork.hpp $(SRCDIR)/NeuralNetwork.cu \
	$(SRCDIR)/NeuralNetworkCUDAFunctions.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/NeuralNetwork.cu -o $(BUILDDIR)/NeuralNetwork.o

clean:
	rm $(OBJS) $(TESTOBJS) $(LIBDIR)/libneuralnetwork.so

test: $(TESTDIR)/NeuralNetworkTest
	$(TESTDIR)/NeuralNetworkTest
