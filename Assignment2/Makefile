
CXX?=g++
CXXFLAGS?=-g2
NVCC?=nvcc
NVFLAGS?=-g --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets
MPICXX?=mpic++

MAKEDEPEND = $(CXX) -MM -MT $@ $(CXXFLAGS) -o $*.Td $<
MAKEDEPENDCUDA = $(NVCC) -MM -MT $@ $(NVFLAGS) -o $*.Td $<

# Makedepend with post-processing to add dummy rules for each dependency
MAKEDEPEND_INFO = $(MAKEDEPEND); \
        if [ -f $*.Td ] ; then cp -f $*.Td $*.d; \
          sed -e 's/\#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
              -e '/^$$/ d' -e 's/$$/ :/' < $*.Td >> $*.d; \
          rm -f $*.Td ; else \
          echo "warning: unable to obtain dependency information for $<"; fi

MAKEDEPENDCUDA_INFO = $(MAKEDEPENDCUDA); \
        if [ -f $*.Td ] ; then cp -f $*.Td $*.d; \
          sed -e 's/\#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
              -e '/^$$/ d' -e 's/$$/ :/' < $*.Td >> $*.d; \
          rm -f $*.Td ; else \
          echo "warning: unable to obtain dependency information for $<"; fi

targets = Assignment2_serial Assignment2_openmp Assignment2_mpi Assignment2_cuda

$(targets) : randutil.o eigensolver.o

LIBS_Assignment2_serial = -larpack
LIBS_Assignment2_openmp = -larpack
LIBS_Assignment2_mpi    = -larpack
LIBS_Assignment2_cuda   = -larpack

CXXFLAGS_Assignment2_openmp = -fopenmp

# Assignment2_mpi is special because we need to call the MPI compiler but there is no distinguishing extension
# we can use for a general rule.
Assignment2_mpi : Assignment2_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(LDFLAGS) $(sort $(patsubst ./%,%, $(filter %.o %.cpp, $^))) $(LIBS_$@) $(LIB) -o $@

% : %.cu
	@$(MAKEDEPENDCUDA_INFO)
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$@) $(LDFLAGS) $(sort $(patsubst ./%,%, $(filter %.o %.cu, $^))) $(LIBS_$@) $(LIB) -o $@

% : %.cpp
	@$(MAKEDEPEND_INFO)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(LDFLAGS) $(sort $(patsubst ./%,%, $(filter %.o %.cpp, $^))) $(LIBS_$@) $(LIB) -o $@

%.o : %.cpp %.d
	@$(MAKEDEPEND_INFO)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_$@) -c $< -o $@

%.d :
	\

clean :
	rm -f *.o *.d $(targets)
