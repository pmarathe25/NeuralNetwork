BUILDDIR = build/
BINDIR = bin/
LIBMATRIX = ~/C++/Math/lib/libmatrix.so
LIBMATH = ~/C++/Math/lib/libmath.so
LIBMATHINCLUDEPATH = /home/pranav/C++/Math/include/
LIBS = $(LIBMATRIX) $(LIBMATH)
INCLUDEDIR = -Iinclude/ -I$(LIBMATHINCLUDEPATH)
TESTOBJS = $(BUILDDIR)/NeuralNetworkTest.o
TESTDIR = test/
SRCDIR = src/
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 $(INCLUDEDIR)
LFLAGS = -shared -Wno-deprecated-gpu-targets
TESTLFLAGS = -Wno-deprecated-gpu-targets

$(TESTDIR)/NeuralNetworkTest: $(TESTOBJS) $(LIBMATRIX)
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) $(LIBS) -o $(TESTDIR)/NeuralNetworkTest

$(BUILDDIR)/NeuralNetworkTest.o: $(TESTDIR)/NeuralNetworkTest.cu include/NeuralNetwork.hpp include/Layer/Layer.hpp \
	include/Layer/FullyConnectedLayer.hpp
	$(CXX) $(CFLAGS) $(TESTDIR)/NeuralNetworkTest.cu -o $(BUILDDIR)/NeuralNetworkTest.o

clean:
	rm $(OBJS) $(TESTOBJS) $(LIBDIR)/libneuralnetwork.so

test: $(TESTDIR)/NeuralNetworkTest
	$(TESTDIR)/NeuralNetworkTest
