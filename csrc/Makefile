# ================= Compiler ==============================================

MPICC := /opt/openmpi/gcc-14.2.0/bin/mpicc
CC    := $(MPICC)

# ================= Flags =================================================

CSTD   := -std=c17
WARN   := -Wall -Wextra -pedantic
OPT    := -O3
ALIAS  := -fstrict-aliasing

# Apple Silicon tuning (override on command line if needed)
#   make ARCH_FLAGS=
ARCH_FLAGS ?= -mcpu=apple-m1

CFLAGS   := $(CSTD) $(WARN) $(OPT) $(ALIAS) $(ARCH_FLAGS)
CPPFLAGS := -Iinclude

LDFLAGS :=
LDLIBS  := -lm

# ================= Layout ===============================================

SRC_DIR   := src
TEST_DIR  := tests
BUILD_DIR := build
BIN_DIR   := bin

# ================= Library sources =======================================

LIB_SRCS := $(SRC_DIR)/base.c \
            $(SRC_DIR)/temporal_profile.c \
            $(SRC_DIR)/transverse_profile.c \
            $(SRC_DIR)/polarization.c \
            $(SRC_DIR)/ionization_model.c \
            $(SRC_DIR)/core.c \
            $(SRC_DIR)/mdf.c

LIB_OBJS := $(LIB_SRCS:%.c=$(BUILD_DIR)/%.o)

# ================= Test executables ======================================

TEST_BASE_SRC     := $(TEST_DIR)/test_base.c
TEST_TEMPORAL_SRC := $(TEST_DIR)/test_temporal.c

TEST_BASE_OBJ     := $(TEST_BASE_SRC:%.c=$(BUILD_DIR)/%.o)
TEST_TEMPORAL_OBJ := $(TEST_TEMPORAL_SRC:%.c=$(BUILD_DIR)/%.o)

TEST_BASE_EXE     := $(BIN_DIR)/test_base
TEST_TEMPORAL_EXE := $(BIN_DIR)/test_temporal

TEST_EXES := $(TEST_BASE_EXE) $(TEST_TEMPORAL_EXE)

# ================= Default target ========================================

all: $(TEST_EXES)

# ================= Linking ===============================================

$(TEST_BASE_EXE): $(LIB_OBJS) $(TEST_BASE_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(TEST_TEMPORAL_EXE): $(LIB_OBJS) $(TEST_TEMPORAL_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# ================= Compilation ===========================================

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

# ================= Convenience targets ===================================

test: all
	$(TEST_BASE_EXE)
	$(TEST_TEMPORAL_EXE) 60 -200 200 0.5 /tmp/out_temporal_c.csv
	@echo "Wrote /tmp/out_temporal_c.csv"

run-temporal: $(TEST_TEMPORAL_EXE)
	$(TEST_TEMPORAL_EXE) 60 -200 200 0.5 /tmp/out_temporal_c.csv
	@echo "Wrote /tmp/out_temporal_c.csv"

debug: OPT := -O0
debug: ARCH_FLAGS :=
debug: CFLAGS := $(CSTD) $(WARN) -g
debug: clean all

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean debug test run-temporal

