/* ============================================================================
 * File: field_request.c
 * ============================================================================
 */
#include "field_request.h"

static int validate_axis_spec(const FieldAxisSpec *a, FieldAxisId expected_id)
{
    if (!a)
        return 1;
    if (a->id != expected_id)
        return 2;

    if (a->kind == FIELD_AXIS_VALUES)
    {
        if (!a->values)
            return 3;
        if (a->n < 2)
            return 4;
        return 0;
    }
    else if (a->kind == FIELD_AXIS_LINSPACE)
    {
        if (a->lin.n < 2)
            return 5;
        /* min/max can be equal only if n==1, but we disallow n<2 anyway */
        return 0;
    }
    return 6;
}

FieldRequestOptions field_request_default_options(void)
{
    FieldRequestOptions opt;
    opt.quantity_mask = (uint32_t)FIELD_Q_E;
    opt.component_mask = (uint32_t)FIELD_C_ALL;
    opt.assemble = FIELD_ASSEMBLE_GATHER_TO_ROOT;
    opt.root_rank = 0;
    opt.label = NULL;
    return opt;
}

int field_request_validate(const FieldRequest *req)
{
    if (!req)
        return 1;

    /* quantity/component masks must request something */
    if (req->opt.quantity_mask == 0)
        return 2;
    if (req->opt.component_mask == 0)
        return 3;

    /* Only allow known bits */
    if (req->opt.quantity_mask & ~((uint32_t)FIELD_Q_E | (uint32_t)FIELD_Q_A))
        return 4;
    if (req->opt.component_mask & ~((uint32_t)FIELD_C_ALL))
        return 5;

    /* Assemble policy must be known */
    if (req->opt.assemble != FIELD_ASSEMBLE_GATHER_TO_ROOT &&
        req->opt.assemble != FIELD_ASSEMBLE_KEEP_DISTRIBUTED)
        return 6;

    /* Root rank can be any int; calculators will validate against comm size */

    switch (req->kind)
    {

    case FIELD_REQ_TIMESERIES_POINTS:
    {
        const FieldRequestTimeseriesPoints *r = &req->u.ts_points;
        if (r->np > 0 && !r->points_um)
            return 10;
        if (validate_axis_spec(&r->t, FIELD_AXIS_T) != 0)
            return 11;
        return 0;
    }

    case FIELD_REQ_1D_LINEOUT:
    {
        const FieldRequest1DLineout *r = &req->u.line1d;
        if (r->axis != FIELD_AXIS_X && r->axis != FIELD_AXIS_Y && r->axis != FIELD_AXIS_Z)
            return 20;
        if (validate_axis_spec(&r->a1, r->axis) != 0)
            return 21;
        return 0;
    }

    case FIELD_REQ_2D_SLICE:
    {
        const FieldRequest2DSlice *r = &req->u.slice2d;

        FieldAxisId a1_expected, a2_expected;
        if (r->plane == FIELD_PLANE_XY)
        {
            a1_expected = FIELD_AXIS_X;
            a2_expected = FIELD_AXIS_Y;
        }
        else if (r->plane == FIELD_PLANE_XZ)
        {
            a1_expected = FIELD_AXIS_X;
            a2_expected = FIELD_AXIS_Z;
        }
        else if (r->plane == FIELD_PLANE_YZ)
        {
            a1_expected = FIELD_AXIS_Y;
            a2_expected = FIELD_AXIS_Z;
        }
        else
            return 30;

        if (validate_axis_spec(&r->a1, a1_expected) != 0)
            return 31;
        if (validate_axis_spec(&r->a2, a2_expected) != 0)
            return 32;

        return 0;
    }

    default:
        return 100;
    }
}
