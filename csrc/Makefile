# ================= Compiler ==============================================

MPICC := /opt/openmpi/gcc-14.2.0/bin/mpicc
CC    := $(MPICC)


# ================= HDF5 ==================================================

H5_ROOT  := /opt/hdf5/gcc-14.2.0
H5_FLAGS := -I$(H5_ROOT)/include
H5_LINK  := -L$(H5_ROOT)/lib -lhdf5 -lz -lm -ldl

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

LIB_SRCS := \
	$(SRC_DIR)/base.c \
	$(SRC_DIR)/temporal_profile.c \
	$(SRC_DIR)/transverse_profile.c \
	$(SRC_DIR)/polarization.c \
	$(SRC_DIR)/ionization_model.c \
	$(SRC_DIR)/core.c \
      $(SRC_DIR)/diagnostics_field.c \
      $(SRC_DIR)/diag_h5.c \
	$(SRC_DIR)/mdf.c

LIB_OBJS := $(LIB_SRCS:%.c=$(BUILD_DIR)/%.o)

# ================= Tests ==================================================
# Add new tests by listing the names here (without .c).
# Each test source must be: tests/test_<name>.c
# Each executable will be:  bin/test_<name>

TESTS := temporal transverse adk pulse pulse_A multipulse
TEST_EXES := $(TESTS:%=$(BIN_DIR)/test_%)

# ================= Default target ========================================

all: $(TEST_EXES)

# ================= Generic build rules ===================================

# Compile any .c file into build/<path>.o
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(H5_FLAGS) -c $< -o $@

# Link each test executable:
#   bin/test_<name> = (all library objs) + build/tests/test_<name>.o
$(BIN_DIR)/test_%: $(LIB_OBJS) $(BUILD_DIR)/$(TEST_DIR)/test_%.o
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(H5_FLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS) $(H5_LINK)

# ================= Convenience targets ===================================

debug: OPT := -O0
debug: ARCH_FLAGS :=
debug: CFLAGS := $(CSTD) $(WARN) -g
debug: clean all

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean debug

