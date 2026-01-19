// tests/test_inputdeck.c
#include <stdio.h>
#include <stdlib.h>

#include "inputdeck.h"
#include "laser_build.h"

static void print_vec3(const char *label, const double v[3])
{
    printf("%s[%g,%g,%g]\n", label, v[0], v[1], v[2]);
}

static void eval_and_print_E(const char *label, const LaserPulse *p, double t, const double r[3])
{
    double E[3];
    LaserPulse_E(p, t, r, E);

    char buf[256];
    snprintf(buf, sizeof(buf), "%s (t=%g, r=[%g,%g,%g]) = ",
             label, t, r[0], r[1], r[2]);
    print_vec3(buf, E);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s <input.toml>\n", argv[0]);
        return 2;
    }

    const char *path = argv[1];

    InputSimSpec sim;
    int rc = inputdeck_read(path, &sim);
    if (rc != 0)
    {
        fprintf(stderr, "inputdeck_read failed (rc=%d) for \"%s\"\n", rc, path);
        return rc;
    }

    // Show what we parsed
    inputdeck_dump(&sim);

    // Build runtime lasers
    BuiltLasers L;
    int rc2 = BuiltLasers_build(&sim, &L);
    if (rc2 != 0)
    {
        fprintf(stderr, "BuiltLasers_build failed (rc=%d)\n", rc2);
        inputdeck_free(&sim);
        return rc2;
    }

    printf("\n=== Laser evaluation sanity checks ===\n");
    printf("BuiltLasers.count = %zu\n", L.count);

    // Evaluate at a couple of times to avoid “accidental cancellation”
    const double tvals[] = {0.0, 10.0, 50.0};
    const size_t nt = sizeof(tvals) / sizeof(tvals[0]);

    // Evaluate at origin (you can change this later)
    const double r[3] = {0.0, 1.0, 0.0};

    for (size_t kt = 0; kt < nt; ++kt)
    {
        double t = tvals[kt];
        printf("\n--- t = %g fs ---\n", t);

        // Sum of individual pulses (explicit, computed in test)
        double Esum[3] = {0.0, 0.0, 0.0};

        for (size_t i = 0; i < L.count; ++i)
        {
            const LaserPulse *pi = (const LaserPulse *)&L.single[i];

            double Ei[3];
            LaserPulse_E(pi, t, r, Ei);

            char label[128];
            snprintf(label, sizeof(label), "E_single[%zu]", i);
            eval_and_print_E(label, pi, t, r);

            Esum[0] += Ei[0];
            Esum[1] += Ei[1];
            Esum[2] += Ei[2];
        }

        // Active pulse (should be sum if L.count > 1)
        const LaserPulse *p = BuiltLasers_active(&L);
        eval_and_print_E("E_active", p, t, r);

        // Print explicit sum for comparison
        print_vec3("E_sum(single pulses) = ", Esum);
    }

    BuiltLasers_free(&L);
    inputdeck_free(&sim);

    return 0;
}
