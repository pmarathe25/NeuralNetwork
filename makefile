BUILDDIR = build/
BINDIR = bin/
LIBDIR = $(CURDIR)/lib/
LIBMAT = ~/C++/Math/lib/libmatrix.so
LIBMATINCLUDEPATH = /home/pranav/C++/Math/include/
LIBS = lib/libneuralnetwork.so
INCLUDEDIR = -Iinclude/ -I$(LIBMATINCLUDEPATH)
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

$(BUILDDIR)/NeuralNetwork.o: include/NeuralNetwork/NeuralNetwork.hpp $(SRCDIR)/NeuralNetwork.cu \
	$(SRCDIR)/NeuralNetworkCUDAFunctions.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/NeuralNetwork.cu -o $(BUILDDIR)/NeuralNetwork.o

$(TESTDIR)/NeuralNetworkTest: $(TESTOBJS) $(LIBMAT)
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) $(LIBS) $(LIBMAT) -o $(TESTDIR)/NeuralNetworkTest

$(BUILDDIR)/NeuralNetworkTest.o: $(TESTDIR)/NeuralNetworkTest.cu include/NeuralNetwork/NeuralNetwork.hpp include/NeuralNetwork/Layer/Layer.hpp \
	include/NeuralNetwork/Layer/FullyConnectedLayer.hpp $(LIBDIR)/libneuralnetwork.so
	$(CXX) $(CFLAGS) $(TESTDIR)/NeuralNetworkTest.cu -o $(BUILDDIR)/NeuralNetworkTest.o

clean:
	rm $(OBJS) $(TESTOBJS) $(LIBDIR)/libneuralnetwork.so

test: $(TESTDIR)/NeuralNetworkTest
	$(TESTDIR)/NeuralNetworkTest

exec:
