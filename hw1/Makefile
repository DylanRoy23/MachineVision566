CXX := g++
CXXFLAGS := -std=c++17 -Wall -O2
OPENCV := $(shell pkg-config --exists opencv4 && echo opencv4 || echo opencv)
CXXFLAGS += $(shell pkg-config --cflags $(OPENCV))
LDFLAGS := $(shell pkg-config --libs $(OPENCV))

SRC_DIR := src
INCLUDE_DIR := include
OUTPUT_DIR := output
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
HEADERS := $(wildcard $(INCLUDE_DIR)/*.hpp)
TARGETS := $(patsubst $(SRC_DIR)/%.cpp,$(OUTPUT_DIR)/%,$(SRCS))
PROGRAMS := $(notdir $(TARGETS))

CXXFLAGS += -I$(INCLUDE_DIR)

.PHONY: all run clean

all: $(TARGETS)

run: all
	@cd $(OUTPUT_DIR) && for program in $(PROGRAMS); do \
		echo "Running $$program"; \
		./$$program; \
	done

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OUTPUT_DIR)/%: $(SRC_DIR)/%.cpp $(HEADERS) | $(OUTPUT_DIR)
	@$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(OUTPUT_DIR)
