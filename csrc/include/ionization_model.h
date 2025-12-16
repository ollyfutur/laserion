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

#endif /* IONIZATION_MODEL_H */
