#CXX = g++
CXX = clang++-12
TARGET = main
CXXFLAGS = -std=c++2a -Ofast -Wall -Wextra -march=native -mtune=native -fopenmp 
LDLFLAGS = -lstdc++
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

run: all
	./$(TARGET)

clean:
	rm $(TARGET)
