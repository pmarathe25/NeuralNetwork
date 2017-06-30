BUILDDIR = build/
BINDIR = bin/
LIBDIR = $(CURDIR)/lib/
LIBS = ~/C++/Math/lib/Math/libmatrix.so lib/NeuralNetwork/libneuralnetwork.so
LIBINCLUDEPATH = /home/pranav/C++/Math/include/
INCLUDEDIR = -Iinclude/ -I$(LIBINCLUDEPATH)
OBJS = $(BUILDDIR)/NeuralNetwork.o
TESTOBJS = $(BUILDDIR)/NeuralNetworkTest.o
EXECOBJS =
TESTDIR = test/
SRCDIR = src/
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 $(INCLUDEDIR)
LFLAGS = -shared -Wno-deprecated-gpu-targets
TESTLFLAGS = -Wno-deprecated-gpu-targets
EXECLFLAGS = -Wno-deprecated-gpu-targets

$(LIBDIR)/NeuralNetwork/libneuralnetwork.so: $(OBJS) ~/C++/Math/lib/Math/libmatrix.so
	$(CXX) $(LFLAGS) $(OBJS) ~/C++/Math/lib/Math/libmatrix.so -o $(LIBDIR)/NeuralNetwork/libneuralnetwork.so

$(BUILDDIR)/NeuralNetwork.o: include/NeuralNetwork/NeuralNetwork.hpp $(LIBINCLUDEPATH)/Math/Matrix.hpp $(SRCDIR)/NeuralNetwork.cu \
	$(SRCDIR)/NeuralNetworkCUDAFunctions.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/NeuralNetwork.cu -o $(BUILDDIR)/NeuralNetwork.o

$(TESTDIR)/NeuralNetworkTest: $(TESTOBJS)
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) $(LIBS) -o $(TESTDIR)/NeuralNetworkTest

$(BUILDDIR)/NeuralNetworkTest.o: $(TESTDIR)/NeuralNetworkTest.cpp include/NeuralNetwork/NeuralNetwork.hpp \
	$(LIBDIR)/NeuralNetwork/libneuralnetwork.so
	$(CXX) $(CFLAGS) $(TESTDIR)/NeuralNetworkTest.cpp -o $(BUILDDIR)/NeuralNetworkTest.o

clean:
	rm $(OBJS) $(TESTOBJS) $(LIBDIR)/NeuralNetwork/libneuralnetwork.so

test: $(TESTDIR)/NeuralNetworkTest
	$(TESTDIR)/NeuralNetworkTest

exec:
