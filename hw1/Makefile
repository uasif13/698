#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2018 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2006-2007 Sun Microsystems, Inc.  All rights reserved.
# Copyright (c) 2011-2020 Cisco Systems, Inc.  All rights reserved
# Copyright (c) 2012      Los Alamos National Security, Inc.  All rights reserved.
# Copyright (c) 2013      Mellanox Technologies, Inc.  All rights reserved.
# Copyright (c) 2017-2018 Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# Use the Open MPI-provided wrapper compilers.

MPICC = mpicc

# Using -g is not necessary, but it is helpful for example programs,
# especially if users want to examine them with debuggers.  Note that
# gmake requires the CXXFLAGS macro, while other versions of make
# (such as Sun's make) require the CCFLAGS macro.

CFLAGS += -g
CXXFLAGS += -g
CCFLAGS += -g
FCFLAGS += -g

# Example programs to build


# Default target.  Always build the C MPI examples.  Only build the
# others if we have the appropriate Open MPI / OpenSHMEM language
# bindings.

# MPI examples

clean:
	rm -f $(EXAMPLES) *~ *.o

# Don't rely on default rules for the Fortran and Java examples

hw1: hw1.c
	mpicc -g hw1.c -o hw1
