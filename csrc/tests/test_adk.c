/* csrc/tests/test_adk.c
 *
 * Exhaustive ADK sweep:
 *   species,Z,E_abs,ion_ene_eV,w
 *
 * E_abs = 10^k, k = Emin_exp ... Emax_exp
 *
 * Usage:
 *   test_adk --Emin-exp <int> --Emax-exp <int>
 *
 * Example:
 *   ./bin/test_adk --Emin-exp 0 --Emax-exp 9
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ionization_model.h"

static void die_usage(const char *prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s --Emin-exp <int> --Emax-exp <int>\n\n"
        "Outputs CSV to stdout:\n"
        "  species,Z,E_abs,ion_ene_eV,w\n",
        prog);
    exit(2);
}

static int streq(const char *a, const char *b)
{
    return (a && b && strcmp(a, b) == 0);
}

int main(int argc, char **argv)
{
    int Emin_exp = 0, Emax_exp = 0;
    int got_min = 0, got_max = 0;

    for (int i = 1; i < argc; ++i) {
        if (streq(argv[i], "--Emin-exp") && i+1 < argc) {
            Emin_exp = atoi(argv[++i]);
            got_min = 1;
        } else if (streq(argv[i], "--Emax-exp") && i+1 < argc) {
            Emax_exp = atoi(argv[++i]);
            got_max = 1;
        } else {
            die_usage(argv[0]);
        }
    }

    if (!got_min || !got_max || Emin_exp > Emax_exp)
        die_usage(argv[0]);

    /* Keep this list consistent with your compiled-in tables. */
    static const char *species_list[] = {
        "H", "He", "Ne", "Ar", "Kr", "Xe"
    };
    const size_t nspecies = sizeof(species_list) / sizeof(species_list[0]);

    printf("species,Z,E_abs,ion_ene_eV,w\n");

    for (size_t is = 0; is < nspecies; ++is) {
        const char *species = species_list[is];

        /* Discover how many charge states exist by probing the table. */
        for (int Z = 1; ; ++Z) {
            double ion_ene = NAN;
            if (ADK_ionization_energy(&ion_ene, species, Z) != 0) {
                /* No more levels for this species. */
                break;
            }

            for (int k = Emin_exp; k <= Emax_exp; ++k) {
                double E = pow(10.0, (double)k);

                double w = ADK_scalar_rate(E, ion_ene, Z);

                /* Guard against NaN/inf/negatives (should not happen, but keeps CSV clean). */
                if (!isfinite(w) || w < 0.0)
                    w = 0.0;

                printf("%s,%d,%.17g,%.17g,%.17g\n",
                       species, Z, E, ion_ene, w);
            }
        }
    }

    return 0;
}

