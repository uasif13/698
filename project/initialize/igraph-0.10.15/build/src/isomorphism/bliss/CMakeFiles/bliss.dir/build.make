# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/asifuddin/698/project/initialize/igraph-0.10.15

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/asifuddin/698/project/initialize/igraph-0.10.15/build

# Include any dependencies generated for this target.
include src/isomorphism/bliss/CMakeFiles/bliss.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/isomorphism/bliss/CMakeFiles/bliss.dir/compiler_depend.make

# Include the progress variables for this target.
include src/isomorphism/bliss/CMakeFiles/bliss.dir/progress.make

# Include the compile flags for this target's objects.
include src/isomorphism/bliss/CMakeFiles/bliss.dir/flags.make

src/isomorphism/bliss/CMakeFiles/bliss.dir/defs.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/flags.make
src/isomorphism/bliss/CMakeFiles/bliss.dir/defs.cc.o: /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/defs.cc
src/isomorphism/bliss/CMakeFiles/bliss.dir/defs.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/asifuddin/698/project/initialize/igraph-0.10.15/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/isomorphism/bliss/CMakeFiles/bliss.dir/defs.cc.o"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/isomorphism/bliss/CMakeFiles/bliss.dir/defs.cc.o -MF CMakeFiles/bliss.dir/defs.cc.o.d -o CMakeFiles/bliss.dir/defs.cc.o -c /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/defs.cc

src/isomorphism/bliss/CMakeFiles/bliss.dir/defs.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bliss.dir/defs.cc.i"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/defs.cc > CMakeFiles/bliss.dir/defs.cc.i

src/isomorphism/bliss/CMakeFiles/bliss.dir/defs.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bliss.dir/defs.cc.s"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/defs.cc -o CMakeFiles/bliss.dir/defs.cc.s

src/isomorphism/bliss/CMakeFiles/bliss.dir/graph.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/flags.make
src/isomorphism/bliss/CMakeFiles/bliss.dir/graph.cc.o: /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/graph.cc
src/isomorphism/bliss/CMakeFiles/bliss.dir/graph.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/asifuddin/698/project/initialize/igraph-0.10.15/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/isomorphism/bliss/CMakeFiles/bliss.dir/graph.cc.o"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/isomorphism/bliss/CMakeFiles/bliss.dir/graph.cc.o -MF CMakeFiles/bliss.dir/graph.cc.o.d -o CMakeFiles/bliss.dir/graph.cc.o -c /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/graph.cc

src/isomorphism/bliss/CMakeFiles/bliss.dir/graph.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bliss.dir/graph.cc.i"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/graph.cc > CMakeFiles/bliss.dir/graph.cc.i

src/isomorphism/bliss/CMakeFiles/bliss.dir/graph.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bliss.dir/graph.cc.s"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/graph.cc -o CMakeFiles/bliss.dir/graph.cc.s

src/isomorphism/bliss/CMakeFiles/bliss.dir/heap.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/flags.make
src/isomorphism/bliss/CMakeFiles/bliss.dir/heap.cc.o: /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/heap.cc
src/isomorphism/bliss/CMakeFiles/bliss.dir/heap.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/asifuddin/698/project/initialize/igraph-0.10.15/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/isomorphism/bliss/CMakeFiles/bliss.dir/heap.cc.o"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/isomorphism/bliss/CMakeFiles/bliss.dir/heap.cc.o -MF CMakeFiles/bliss.dir/heap.cc.o.d -o CMakeFiles/bliss.dir/heap.cc.o -c /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/heap.cc

src/isomorphism/bliss/CMakeFiles/bliss.dir/heap.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bliss.dir/heap.cc.i"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/heap.cc > CMakeFiles/bliss.dir/heap.cc.i

src/isomorphism/bliss/CMakeFiles/bliss.dir/heap.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bliss.dir/heap.cc.s"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/heap.cc -o CMakeFiles/bliss.dir/heap.cc.s

src/isomorphism/bliss/CMakeFiles/bliss.dir/orbit.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/flags.make
src/isomorphism/bliss/CMakeFiles/bliss.dir/orbit.cc.o: /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/orbit.cc
src/isomorphism/bliss/CMakeFiles/bliss.dir/orbit.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/asifuddin/698/project/initialize/igraph-0.10.15/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/isomorphism/bliss/CMakeFiles/bliss.dir/orbit.cc.o"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/isomorphism/bliss/CMakeFiles/bliss.dir/orbit.cc.o -MF CMakeFiles/bliss.dir/orbit.cc.o.d -o CMakeFiles/bliss.dir/orbit.cc.o -c /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/orbit.cc

src/isomorphism/bliss/CMakeFiles/bliss.dir/orbit.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bliss.dir/orbit.cc.i"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/orbit.cc > CMakeFiles/bliss.dir/orbit.cc.i

src/isomorphism/bliss/CMakeFiles/bliss.dir/orbit.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bliss.dir/orbit.cc.s"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/orbit.cc -o CMakeFiles/bliss.dir/orbit.cc.s

src/isomorphism/bliss/CMakeFiles/bliss.dir/partition.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/flags.make
src/isomorphism/bliss/CMakeFiles/bliss.dir/partition.cc.o: /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/partition.cc
src/isomorphism/bliss/CMakeFiles/bliss.dir/partition.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/asifuddin/698/project/initialize/igraph-0.10.15/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/isomorphism/bliss/CMakeFiles/bliss.dir/partition.cc.o"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/isomorphism/bliss/CMakeFiles/bliss.dir/partition.cc.o -MF CMakeFiles/bliss.dir/partition.cc.o.d -o CMakeFiles/bliss.dir/partition.cc.o -c /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/partition.cc

src/isomorphism/bliss/CMakeFiles/bliss.dir/partition.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bliss.dir/partition.cc.i"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/partition.cc > CMakeFiles/bliss.dir/partition.cc.i

src/isomorphism/bliss/CMakeFiles/bliss.dir/partition.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bliss.dir/partition.cc.s"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/partition.cc -o CMakeFiles/bliss.dir/partition.cc.s

src/isomorphism/bliss/CMakeFiles/bliss.dir/uintseqhash.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/flags.make
src/isomorphism/bliss/CMakeFiles/bliss.dir/uintseqhash.cc.o: /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/uintseqhash.cc
src/isomorphism/bliss/CMakeFiles/bliss.dir/uintseqhash.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/asifuddin/698/project/initialize/igraph-0.10.15/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/isomorphism/bliss/CMakeFiles/bliss.dir/uintseqhash.cc.o"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/isomorphism/bliss/CMakeFiles/bliss.dir/uintseqhash.cc.o -MF CMakeFiles/bliss.dir/uintseqhash.cc.o.d -o CMakeFiles/bliss.dir/uintseqhash.cc.o -c /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/uintseqhash.cc

src/isomorphism/bliss/CMakeFiles/bliss.dir/uintseqhash.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bliss.dir/uintseqhash.cc.i"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/uintseqhash.cc > CMakeFiles/bliss.dir/uintseqhash.cc.i

src/isomorphism/bliss/CMakeFiles/bliss.dir/uintseqhash.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bliss.dir/uintseqhash.cc.s"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/uintseqhash.cc -o CMakeFiles/bliss.dir/uintseqhash.cc.s

src/isomorphism/bliss/CMakeFiles/bliss.dir/utils.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/flags.make
src/isomorphism/bliss/CMakeFiles/bliss.dir/utils.cc.o: /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/utils.cc
src/isomorphism/bliss/CMakeFiles/bliss.dir/utils.cc.o: src/isomorphism/bliss/CMakeFiles/bliss.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/asifuddin/698/project/initialize/igraph-0.10.15/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/isomorphism/bliss/CMakeFiles/bliss.dir/utils.cc.o"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/isomorphism/bliss/CMakeFiles/bliss.dir/utils.cc.o -MF CMakeFiles/bliss.dir/utils.cc.o.d -o CMakeFiles/bliss.dir/utils.cc.o -c /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/utils.cc

src/isomorphism/bliss/CMakeFiles/bliss.dir/utils.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bliss.dir/utils.cc.i"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/utils.cc > CMakeFiles/bliss.dir/utils.cc.i

src/isomorphism/bliss/CMakeFiles/bliss.dir/utils.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bliss.dir/utils.cc.s"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss/utils.cc -o CMakeFiles/bliss.dir/utils.cc.s

bliss: src/isomorphism/bliss/CMakeFiles/bliss.dir/defs.cc.o
bliss: src/isomorphism/bliss/CMakeFiles/bliss.dir/graph.cc.o
bliss: src/isomorphism/bliss/CMakeFiles/bliss.dir/heap.cc.o
bliss: src/isomorphism/bliss/CMakeFiles/bliss.dir/orbit.cc.o
bliss: src/isomorphism/bliss/CMakeFiles/bliss.dir/partition.cc.o
bliss: src/isomorphism/bliss/CMakeFiles/bliss.dir/uintseqhash.cc.o
bliss: src/isomorphism/bliss/CMakeFiles/bliss.dir/utils.cc.o
bliss: src/isomorphism/bliss/CMakeFiles/bliss.dir/build.make
.PHONY : bliss

# Rule to build all files generated by this target.
src/isomorphism/bliss/CMakeFiles/bliss.dir/build: bliss
.PHONY : src/isomorphism/bliss/CMakeFiles/bliss.dir/build

src/isomorphism/bliss/CMakeFiles/bliss.dir/clean:
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss && $(CMAKE_COMMAND) -P CMakeFiles/bliss.dir/cmake_clean.cmake
.PHONY : src/isomorphism/bliss/CMakeFiles/bliss.dir/clean

src/isomorphism/bliss/CMakeFiles/bliss.dir/depend:
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/asifuddin/698/project/initialize/igraph-0.10.15 /home/asifuddin/698/project/initialize/igraph-0.10.15/src/isomorphism/bliss /home/asifuddin/698/project/initialize/igraph-0.10.15/build /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss /home/asifuddin/698/project/initialize/igraph-0.10.15/build/src/isomorphism/bliss/CMakeFiles/bliss.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/isomorphism/bliss/CMakeFiles/bliss.dir/depend

