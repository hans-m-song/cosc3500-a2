CXX?=g++
CXXFLAGS?=-g2

NVCC?=nvcc
NVFLAGS?=-g --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets

MAKEDEPEND = $(CXX) -MM -MT $@ $(CXXFLAGS) -o $*.Td $<

MAKEDEPENDCUDA = $(NVCC) -MM -MT $@ $(NVFLAGS) -o $*.Td $<

# Makedepend with post-processing to add dummy rules for each dependency
MAKEDEPEND_INFO = $(MAKEDEPEND); \
        if [ -f $*.Td ] ; then cp -f $*.Td $*.d; \
          sed -e 's/\#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
              -e '/^$$/ d' -e 's/$$/ :/' < $*.Td >> $*.d; \
          rm -f $*.Td ; else \
          echo "warning: unable to obtain dependency information for $<"; fi

# Makedepend with post-processing to add dummy rules for each dependency
MAKEDEPENDCUDA_INFO = $(MAKEDEPENDCUDA); \
        if [ -f $*.Td ] ; then cp -f $*.Td $*.d; \
          sed -e 's/\#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
              -e '/^$$/ d' -e 's/$$/ :/' < $*.Td >> $*.d; \
          rm -f $*.Td ; else \
          echo "warning: unable to obtain dependency information for $<"; fi


% : %.cu
	@$(MAKEDEPENDCUDA_INFO)
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$@) $(LDFLAGS) $(sort $(patsubst ./%,%, $(filter %.o %.cu, $^))) $(LIBS_$@) $(LIB) -o $@

%.o : %.cu %.d
	@$(MAKEDEPENDCUDA_INFO)
	(NVCC) $(NVFLAGS) $(NVFLAGS_$@) -c $< -o $@

% : %.cpp
	@$(MAKEDEPEND_INFO)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(LDFLAGS) $(sort $(patsubst ./%,%, $(filter %.o %.cpp, $^))) $(LIBS_$@) $(LIB) -o $@

%.o : %.cpp %.d
	@$(MAKEDEPEND_INFO)
	(CXX) $(CXXFLAGS) $(CXXFLAGS_$@) -c $< -o $@

%.d :
	\

clean :
	rm -f *.o
	rm -f *.d

