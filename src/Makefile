CXX=g++
CFLAGS=-O3 -DEIGEN_NO_DEBUG -DNDEBUG
ARCH:=$(shell uname -m)
OS:=$(shell uname -s)

# Set this to the root directory of Boost
# (should have a subdirectory named boost):
BOOST=/home/sagae/krusty1/boost/boost_1_57_0
# Where to find Boost header files
BOOST_INC=/home/sagae/krusty1/boost/boost_1_57_0
BOOST_LIB=$(BOOST)/stage/lib

# Set this to the root directory of Eigen 
# (should have a subdirectory named Eigen):
EIGEN=../../../3rdparty

# To disable multithreading, comment out the line below:
OMP=1

#TCLAP=../../../3rdparty/tclap/include

BOOST_CFLAGS=-I$(BOOST_INC)
BOOST_LDFLAGS=-L$(BOOST_LIB)
BOOST_LDLIBS=-lboost_iostreams -lboost_system -lboost_filesystem

ALL_CFLAGS=$(BOOST_CFLAGS) -I$(EIGEN) $(CFLAGS) 
#ALL_CFLAGS=$(BOOST_CFLAGS) -I$(TCLAP) -I$(EIGEN) $(CFLAGS)
ALL_LDFLAGS=$(BOOST_LDFLAGS) $(LDFLAGS) -Wl,-rpath=$(BOOST_LIB)
ALL_LDLIBS=$(BOOST_LDLIBS)

BINS=nndep
OBJS=util.o model.o

all: $(BINS)

clean:
	rm -f *.o $(BINS)

%.o: %.cpp
	$(CXX) -c $(ALL_CFLAGS) $< -o $@

nndep: nndep.o $(OBJS)
	$(CXX) $(ALL_LDFLAGS) $^ -o $@ $(ALL_LDLIBS)

