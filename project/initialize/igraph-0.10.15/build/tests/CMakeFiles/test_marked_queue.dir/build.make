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
include tests/CMakeFiles/test_marked_queue.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test_marked_queue.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test_marked_queue.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test_marked_queue.dir/flags.make

tests/CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.o: tests/CMakeFiles/test_marked_queue.dir/flags.make
tests/CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.o: /home/asifuddin/698/project/initialize/igraph-0.10.15/tests/unit/marked_queue.c
tests/CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.o: tests/CMakeFiles/test_marked_queue.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/asifuddin/698/project/initialize/igraph-0.10.15/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object tests/CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.o"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT tests/CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.o -MF CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.o.d -o CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.o -c /home/asifuddin/698/project/initialize/igraph-0.10.15/tests/unit/marked_queue.c

tests/CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.i"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/asifuddin/698/project/initialize/igraph-0.10.15/tests/unit/marked_queue.c > CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.i

tests/CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.s"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/asifuddin/698/project/initialize/igraph-0.10.15/tests/unit/marked_queue.c -o CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.s

# Object files for target test_marked_queue
test_marked_queue_OBJECTS = \
"CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.o"

# External object files for target test_marked_queue
test_marked_queue_EXTERNAL_OBJECTS = \
"/home/asifuddin/698/project/initialize/igraph-0.10.15/build/tests/CMakeFiles/test_utilities.dir/unit/test_utilities.c.o"

tests/test_marked_queue: tests/CMakeFiles/test_marked_queue.dir/unit/marked_queue.c.o
tests/test_marked_queue: tests/CMakeFiles/test_utilities.dir/unit/test_utilities.c.o
tests/test_marked_queue: tests/CMakeFiles/test_marked_queue.dir/build.make
tests/test_marked_queue: src/libigraph.a
tests/test_marked_queue: /usr/lib64/libm.so
tests/test_marked_queue: /usr/lib/gcc/x86_64-redhat-linux/12/libgomp.so
tests/test_marked_queue: /usr/lib64/libpthread.a
tests/test_marked_queue: tests/CMakeFiles/test_marked_queue.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/asifuddin/698/project/initialize/igraph-0.10.15/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_marked_queue"
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_marked_queue.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test_marked_queue.dir/build: tests/test_marked_queue
.PHONY : tests/CMakeFiles/test_marked_queue.dir/build

tests/CMakeFiles/test_marked_queue.dir/clean:
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_marked_queue.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test_marked_queue.dir/clean

tests/CMakeFiles/test_marked_queue.dir/depend:
	cd /home/asifuddin/698/project/initialize/igraph-0.10.15/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/asifuddin/698/project/initialize/igraph-0.10.15 /home/asifuddin/698/project/initialize/igraph-0.10.15/tests /home/asifuddin/698/project/initialize/igraph-0.10.15/build /home/asifuddin/698/project/initialize/igraph-0.10.15/build/tests /home/asifuddin/698/project/initialize/igraph-0.10.15/build/tests/CMakeFiles/test_marked_queue.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/test_marked_queue.dir/depend

