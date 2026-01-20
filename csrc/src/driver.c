#include "driver.h"

#include <stdio.h>

#include "inputdeck.h"
#include "laser_build.h"
#include "field_cache.h"

int run_from_inputdeck(const char *toml_path,
                       const char *field_cache_prefix,
                       MPI_Comm comm)
{
    InputSimSpec sim;
    int rc = inputdeck_read(toml_path, &sim);
    if (rc != 0) return rc;

    BuiltLasers bl;
    int rc2 = BuiltLasers_build(&sim, &bl);
    if (rc2 != 0) {
        inputdeck_free(&sim);
        return rc2;
    }

    const LaserPulse *pulse = BuiltLasers_active(&bl);

    FieldCacheOptions opt = field_cache_default_options();
    // opt.compute_A = 1;
    // opt.merge_on_root = 1;

    int rc3 = field_cache_run(&sim, pulse, field_cache_prefix, &opt, comm);

    BuiltLasers_free(&bl);
    inputdeck_free(&sim);
    return rc3;
}

