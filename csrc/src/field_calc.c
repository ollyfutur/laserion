/* ============================================================================
 * File: field_calc.c
 *
 * Dispatcher: FieldRequest -> specific calculator implementation.
 * ============================================================================
 */
#include "field_calc.h"
#include "field_calc_timeseries.h"

int field_calc_compute(const struct LaserPulse *pulse,
                       const FieldRequest *req,
                       FieldResult *out,
                       MPI_Comm comm)
{
    if (!pulse || !req || !out) return 1;

    switch (req->kind)
    {
        case FIELD_REQ_TIMESERIES_POINTS:
            return field_calc_timeseries_points(pulse, req, out, comm);

        /* Add later:
         * case FIELD_REQ_1D_LINEOUT: return field_calc_1d(...);
         * case FIELD_REQ_2D_SLICE:   return field_calc_2d(...);
         */

        default:
            return 2;
    }
}

