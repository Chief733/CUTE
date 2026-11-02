CXX ?= g++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic

.PHONY: all clean test

all: fast_parse

fast_parse: src/fast_parse.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

test: fast_parse
	python3 -m unittest discover -s tests -v

clean:
	rm -f fast_parse

