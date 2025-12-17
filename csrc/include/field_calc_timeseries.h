/* ============================================================================
 * File: field_calc_timeseries.h
 * ============================================================================
 */
#pragma once

#include "field_request.h"
#include "field_result.h"
#include "core.h"

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Computes FIELD_REQ_TIMESERIES_POINTS requests.
 *
 * If req->opt.assemble == FIELD_ASSEMBLE_GATHER_TO_ROOT:
 *   - On root_rank: out is fully populated.
 *   - On other ranks: out is initialized; out->ndatasets may be 0.
 *
 * Returns 0 on success.
 */
int field_calc_timeseries_points(const struct LaserPulse *pulse,
                                 const FieldRequest *req,
                                 FieldResult *out,
                                 MPI_Comm comm);

#ifdef __cplusplus
}
#endif

