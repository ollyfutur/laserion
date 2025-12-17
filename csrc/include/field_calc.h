/* ============================================================================
 * File: field_calc.h
 *
 * Purpose:
 *   High-level calculator entry point:
 *     FieldRequest + LaserPulse + MPI_Comm  -> FieldResult
 *
 * The implementation will dispatch based on request kind:
 *   - FIELD_REQ_TIMESERIES_POINTS -> field_calc_timeseries
 *   - FIELD_REQ_1D_LINEOUT        -> field_calc_1d
 *   - FIELD_REQ_2D_SLICE          -> field_calc_2d
 * ============================================================================
 */
#pragma once

#include "field_request.h"
#include "field_result.h"
#include "core.h"  /* LaserPulse */
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compute and assemble according to req->opt.assemble.
 * If req->opt.assemble == GATHER_TO_ROOT:
 *   - On root: res is fully populated.
 *   - On non-root: res is initialized but may have zero datasets (implementation choice).
 *
 * Return 0 on success.
 */
int field_calc_compute(const struct LaserPulse *pulse,
                       const FieldRequest *req,
                       FieldResult *out,
                       MPI_Comm comm);

#ifdef __cplusplus
}
#endif

