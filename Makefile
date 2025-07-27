# Define the compiler and flags
NVCC = /usr/local/cuda/bin/nvcc
CXXFLAGS = -std=c++17 -I/usr/local/cuda/include
# IMPORTANT: Added -lcufft for the FFT library
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcufft

# Define directories
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data
# IMPORTANT: Set this to the path containing AudioFile.h
LIB_INCLUDE = -I./lib

# Define source files and target executable
SRC = $(SRC_DIR)/fft.cu
TARGET = $(BIN_DIR)/fft

# Define the default rule
all: $(TARGET)

# Rule for building the target executable
$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(LIB_INCLUDE) $(SRC) -o $(TARGET) $(LDFLAGS)

# Rule for running the application
run: $(TARGET)
	./$(TARGET) $(DATA_DIR)/input $(DATA_DIR)/output

# Clean up
clean:
	rm -rf $(BIN_DIR)/*

# Installation rule (not much to install, but here for completeness)
install:
	@echo "No installation required."

# Help command
help:
	@echo "Available make commands:"
	@echo "  make      - Build the project."
	@echo "  make run  - Run the project."
	@echo "  make clean- Clean up the build files."
	@echo "  make help - Display this help message."