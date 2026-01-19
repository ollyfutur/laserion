#ifndef LASER_BUILD_H
#define LASER_BUILD_H

#include <stddef.h>
#include "inputdeck.h"
#include "laser.h"
#include "base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    size_t count;

    // One per input laser
    SinglePulse *single;                // [count]
    const TemporalProfile **temporal;   // [count] (owned by this bundle)
    const TransverseProfile **trans;    // [count] (owned by this bundle)

    // Convenience pointers to pass into MultiPulse_init
    const LaserPulse **pulse_ptrs;      // [count]

    // Optional sum pulse (active pulse)
    int has_sum;
    MultiPulse sum;
} BuiltLasers;

/* Build runtime laser objects from parsed input. Returns 0 on success. */
int BuiltLasers_build(const InputSimSpec *sim, BuiltLasers *out);

/* Free all allocations inside BuiltLasers. Safe to call multiple times. */
void BuiltLasers_free(BuiltLasers *b);

/* Returns a LaserPulse you can evaluate (SinglePulse if count=1, otherwise MultiPulse). */
static inline const LaserPulse *BuiltLasers_active(const BuiltLasers *b)
{
    if (!b || b->count == 0) return NULL;
    return (b->count == 1) ? (const LaserPulse *)&b->single[0]
                           : (const LaserPulse *)&b->sum;
}

#ifdef __cplusplus
}
#endif

#endif

