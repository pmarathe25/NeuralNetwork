BUILDDIR = build/
BINDIR = bin/
LIBDIR = $(CURDIR)/lib/
MATLIB = ~/C++/Math/lib/Math/libmatrix.so
LIBS = lib/NeuralNetwork/libneuralnetwork.so
LIBINCLUDEPATH = /home/pranav/C++/Math/include/
INCLUDEDIR = -Iinclude/ -I$(LIBINCLUDEPATH)
OBJS = $(addprefix $(BUILDDIR)/, NeuralNetwork.o)
TESTOBJS = $(BUILDDIR)/NeuralNetworkTest.o
EXECOBJS =
TESTDIR = test/
SRCDIR = src/
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 $(INCLUDEDIR)
LFLAGS = -shared -Wno-deprecated-gpu-targets
TESTLFLAGS = -Wno-deprecated-gpu-targets

$(LIBDIR)/NeuralNetwork/libneuralnetwork.so: $(OBJS)
	$(CXX) $(LFLAGS) $(OBJS) -o $(LIBDIR)/NeuralNetwork/libneuralnetwork.so

$(BUILDDIR)/NeuralNetwork.o: include/NeuralNetwork/NeuralNetwork.hpp $(SRCDIR)/NeuralNetwork.cu \
	$(SRCDIR)/NeuralNetworkCUDAFunctions.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/NeuralNetwork.cu -o $(BUILDDIR)/NeuralNetwork.o

$(TESTDIR)/NeuralNetworkTest: $(TESTOBJS) $(MATLIB)
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) $(LIBS) $(MATLIB) -o $(TESTDIR)/NeuralNetworkTest

$(BUILDDIR)/NeuralNetworkTest.o: $(TESTDIR)/NeuralNetworkTest.cu include/NeuralNetwork/NeuralNetwork.hpp include/NeuralNetwork/Layer/Layer.hpp \
	include/NeuralNetwork/Layer/FullyConnectedLayer.hpp $(LIBDIR)/NeuralNetwork/libneuralnetwork.so
	$(CXX) $(CFLAGS) $(TESTDIR)/NeuralNetworkTest.cu -o $(BUILDDIR)/NeuralNetworkTest.o

clean:
	rm $(OBJS) $(TESTOBJS) $(LIBDIR)/NeuralNetwork/libneuralnetwork.so

test: $(TESTDIR)/NeuralNetworkTest
	$(TESTDIR)/NeuralNetworkTest

exec:
