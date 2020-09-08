
MPICXX?=mpic++
CXXFLAGS?=-g2

MAKEDEPEND = $(CXX) -MM -MT $@ $(CXXFLAGS) -o $*.Td $<

# Makedepend with post-processing to add dummy rules for each dependency
MAKEDEPEND_INFO = $(MAKEDEPEND); \
        if [ -f $*.Td ] ; then cp -f $*.Td $*.d; \
          sed -e 's/\#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
              -e '/^$$/ d' -e 's/$$/ :/' < $*.Td >> $*.d; \
          rm -f $*.Td ; else \
          echo "warning: unable to obtain dependency information for $<"; fi


% : %.cpp
	@$(MAKEDEPEND_INFO)
	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(LDFLAGS) $(sort $(patsubst ./%,%, $(filter %.o %.cpp, $^))) $(LIBS_$@) $(LIB) -o $@

%.o : %.cpp %.d
	@$(MAKEDEPEND_INFO)
	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) -c $< -o $@

%.d :
	\

clean :
	rm -f *.o
	rm -f *.d

