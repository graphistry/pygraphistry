# Compile all OpenCL kernels and report errors/warnings if any.
# Download and install clcc at http://sourceforge.net/projects/clcc/

CLOPTIONS=--cloptions "-I ."
CLCCOPT=--add_headers
SRC=$(wildcard *.cl)

all: $(SRC:.cl=.o)

%.o : %.cl
	clcc $(CLCCOPT) $(CLOPTIONS) $< -o /dev/null
