# ================= Layout ===============================================

SRC_DIR   := src
BUILD_DIR := build
BIN_DIR   := bin

# ================= Compiler ==============================================

##MPICC := /opt/openmpi/gcc-14.2.0/bin/mpicc
MPICC ?= mpicc
#MPICC := /opt/openmpi/gcc-14.2.0/bin/mpicc
CC    := $(MPICC)


# ================= HDF5 ==================================================

H5CC     ?= h5cc
H5_ROOT  ?= $(shell $(H5CC) -showconfig 2>/dev/null \
               | sed -n 's/^[[:space:]]*Installation point:[[:space:]]*//p')
H5_ROOT := /opt/hdf5/gcc-14.2.0/

H5_FLAGS := -I$(H5_ROOT)/include
H5_LINK  := -L$(H5_ROOT)/lib -lhdf5 -lz -ldl

# ================= TOML (tomlc99) ========================================

TOML_DIR   := vendor/tomlc99
TOML_FLAGS := -I$(TOML_DIR)
TOML_SRC   := $(TOML_DIR)/toml.c
TOML_OBJ   := $(BUILD_DIR)/$(TOML_DIR)/toml.o

# ================= Flags =================================================

CSTD   := -std=c17
WARN   := -Wall -Wextra -pedantic
OPT    := -O3
ALIAS  := -fstrict-aliasing
OMP    := -fopenmp

# === OS / architecture detection =========================================

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(OS),Windows_NT)
    ARCH_FLAGS = -march=native

else ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        ARCH_FLAGS = -mcpu=apple-m1
    else
        ARCH_FLAGS = -march=native
    endif

else
    ARCH_FLAGS = -march=native
endif

CFLAGS   := $(CSTD) $(WARN) $(OPT) $(ALIAS) $(ARCH_FLAGS) $(OMP)
CPPFLAGS := -Iinclude $(TOML_FLAGS)

LDFLAGS :=
LDLIBS  := -lm

# ================= Library sources =======================================

LIB_SRCS := \
	$(SRC_DIR)/base.c \
	$(SRC_DIR)/temporal_profile.c \
	$(SRC_DIR)/transverse_profile.c \
	$(SRC_DIR)/polarization.c \
	$(SRC_DIR)/ionization_model.c \
	$(SRC_DIR)/ionization.c \
	$(SRC_DIR)/laser.c \
	$(SRC_DIR)/diag_h5.c \
	$(SRC_DIR)/diag_grid.c \
	$(SRC_DIR)/field_diag.c \
	$(SRC_DIR)/ionization_diag.c \
	$(SRC_DIR)/mdf.c \
	$(SRC_DIR)/mdf_diag.c \
	$(SRC_DIR)/mdf_particles.c \
	$(SRC_DIR)/inputdeck.c \
	$(SRC_DIR)/field_cache.c \
	$(SRC_DIR)/run.c \
	$(SRC_DIR)/laser_build.c \
	$(SRC_DIR)/main.c

LIB_OBJS := $(LIB_SRCS:%.c=$(BUILD_DIR)/%.o)

# ================= Main executable =======================================

TARGET := $(BIN_DIR)/laserion

# ================= Default target ========================================

all: $(TARGET)

# ================= Generic build rules ===================================

# Compile any .c file into build/<path>.o
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(H5_FLAGS) -c $< -o $@

# Build tomlc99 object
$(TOML_OBJ): $(TOML_SRC)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

# Link main executable:
#   bin/laserion = (all library objs, including main.o) + toml.o
$(TARGET): $(LIB_OBJS) $(TOML_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(H5_FLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS) $(H5_LINK)

# ================= Convenience targets ===================================

profile: clean
	$(MAKE) CC="scorep --mpp=mpi $(CC)" all

debug: OPT := -O0
debug: ARCH_FLAGS :=
debug: CFLAGS := $(CSTD) $(WARN) -g
debug: clean all

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean debug
