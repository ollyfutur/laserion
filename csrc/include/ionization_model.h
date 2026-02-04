#ifndef IONIZATION_MODEL_H
#define IONIZATION_MODEL_H

#include "base.h"

/* ADK ionization model using built-in ionization energy tables (eV). */

typedef struct
{
    IonizationModel base; /* must be first */
    /* currently no runtime parameters; data tables are compiled in */
} ADKModel;

/* Init ADK model (no heap allocations). */
void ADKModel_init(ADKModel *m);

/* Optional helper: query ionization energy [eV]. Returns 0 on success. */
int ADK_ionization_energy(double *E_eV_out, const char *species, int Z);

/* Optional helper: compute scalar ADK rate w(|E|) [1/fs] given ionization energy. */
double ADK_scalar_rate(double E_abs, double ion_ene_eV, int Z);

// Returns 1 if species exists in the internal tables, 0 otherwise.
int ionization_species_supported(const char *species);

// Returns 0 on success and writes Zmax; nonzero on failure.
int ionization_species_Zmax(const char *species, int *Zmax_out);

#endif /* IONIZATION_MODEL_H */
