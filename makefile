BUILDDIR = build/
BINDIR = bin/
LIBMATRIX = ~/C++/Math/lib/libmatrix.so
LIBMATH = ~/C++/Math/lib/libmath.so
# LIBNEURALNETWORK = lib/libneuralnetwork.so
LIBMATHINCLUDEPATH = /home/pranav/C++/Math/include/
LIBS = $(LIBMATRIX) $(LIBMATH) $(LIBNEURALNETWORK)
INCLUDEDIR = -Iinclude/ -I$(LIBMATHINCLUDEPATH)
# OBJS = $(addprefix $(BUILDDIR)/, )
TESTOBJS = $(addprefix $(BUILDDIR)/, NeuralNetworkTest.o)
TESTDIR = test/
SRCDIR = src/
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 $(INCLUDEDIR)
LFLAGS = -shared -Wno-deprecated-gpu-targets
TESTLFLAGS = -Wno-deprecated-gpu-targets

# $(LIBNEURALNETWORK): $(OBJS)
# 	$(CXX) $(LFLAGS) $(OBJS) -o $(LIBNEURALNETWORK)

$(TESTDIR)/NeuralNetworkTest: $(TESTOBJS) $(LIBS)
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) $(LIBS) -o $(TESTDIR)/NeuralNetworkTest

$(BUILDDIR)/NeuralNetworkTest.o: $(TESTDIR)/NeuralNetworkTest.cu include/NeuralNetwork.hpp \
	include/NeuralNetworkOptimizer.hpp include/Layer/Layer.hpp include/Layer/FullyConnectedLayer.hpp
	$(CXX) $(CFLAGS) $(TESTDIR)/NeuralNetworkTest.cu -o $(BUILDDIR)/NeuralNetworkTest.o

clean:
	rm $(TESTOBJS) # $(OBJS) $(LIBNEURALNETWORK)

test: $(TESTDIR)/NeuralNetworkTest
	$(TESTDIR)/NeuralNetworkTest
