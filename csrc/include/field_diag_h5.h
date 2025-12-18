// field_diag_h5.h
#pragma once
#include "field_result.h"

int field_diag_h5_write(const char *path, const FieldResult *res);

int field_diag_h5_write_split(const char *prefix, const FieldResult *res);

