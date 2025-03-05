# Use the Open MPI-provided wrapper compilers.

MPICC = mpicc
MPIFC = mpifort
MPIJAVAC = mpijavac
SHMEMCC = shmemcc
SHMEMCXX = shmemc++
SHMEMFC = shmemfort

# Using -g is not necessary, but it is helpful for example programs,
# especially if users want to examine them with debuggers.  Note that
# gmake requires the CXXFLAGS macro, while other versions of make
# (such as Sun's make) require the CCFLAGS macro.

CFLAGS += -g
CXXFLAGS += -g
CCFLAGS += -g
FCFLAGS += -g

# Example programs to build

EXAMPLES = \
	template




template: template.c
	$(MPICC) $(CFLAGS) $(LDFLAGS) $? $(LDLIBS) -o $@


all:
	$(MAKE)

# The usual "clean" target
clean:
	rm -f $(EXAMPLES) *~ *.o


