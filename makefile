CXX = g++
TARGET = main
CXXFLAGS = -std=c++2a -O2 -Wall -Wextra -march=native -mtune=native -fopenmp 
LDLFLAGS = -lstdc++
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

run: all
	./$(TARGET)

clean:
	rm $(TARGET)
