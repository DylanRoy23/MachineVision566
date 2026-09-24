CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O2
OPENCV := $(shell pkg-config --exists opencv4 && echo opencv4 || echo opencv)
CXXFLAGS += $(shell pkg-config --cflags $(OPENCV))
LDFLAGS := $(shell pkg-config --libs $(OPENCV))

SRC_DIR := src
APP_DIR := apps
INCLUDE_DIR := include
BUILD_DIR := build
OUTPUT_DIR := output

LIB_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
LIB_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(LIB_SRCS))
APP_SRCS := $(wildcard $(APP_DIR)/*.cpp)
TARGETS := $(patsubst $(APP_DIR)/%.cpp,$(OUTPUT_DIR)/%,$(APP_SRCS))
LIBRARY := $(BUILD_DIR)/libhw2.a
HEADERS := $(wildcard $(INCLUDE_DIR)/hw2/*.hpp)

CXXFLAGS += -I$(INCLUDE_DIR)

.PHONY: all run clean

all: $(TARGETS)

run: all
	@cd $(OUTPUT_DIR) && ./detector_comparison
	@cd $(OUTPUT_DIR) && ./feature_selection
	@cd $(OUTPUT_DIR) && ./descriptor_matching
	@cd $(OUTPUT_DIR) && ./roc_curve

$(BUILD_DIR) $(OUTPUT_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIBRARY): $(LIB_OBJECTS)
	ar rcs $@ $^

$(OUTPUT_DIR)/%: $(APP_DIR)/%.cpp $(HEADERS) $(LIBRARY) | $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) $< $(LIBRARY) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)
