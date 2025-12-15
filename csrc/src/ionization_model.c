#include <math.h>
#include <string.h>
#include "ionization_model.h"

/* -------------------- Ionization data tables (from Python) ------------- */
/* Z is assumed to be consecutive 1..N for each species in your dataset.   */

typedef struct {
    const char *name;
    int Zmax;
    const double *E; /* ionization energies [eV], length Zmax */
} IonSpeciesTable;

/* Ar: Z=1..18 */
static const double E_Ar[] = {
    15.7596119, 27.62967, 40.735, 59.58, 74.84, 91.290, 124.41, 143.4567,
    422.60, 479.76, 540.4, 619.0, 685.5, 755.13, 855.5, 918.375,
    4120.6657, 4426.22407
};

/* He: Z=1..2 */
static const double E_He[] = {
    24.587389011, 54.4177655282
};

/* H: Z=1 */
static const double E_H[] = {
    13.598434599702
};

/* Li: Z=1..3 */
static const double E_Li[] = {
    5.391714996, 75.6400970, 122.45435913
};

/* Ne: Z=1..10 */
static const double E_Ne[] = {
    21.564541, 40.96297, 63.4233, 97.1900, 126.247, 157.934,
    207.271, 239.0970, 1195.80784, 1362.199256
};

/* Xe: Z=1..54 */
static const double E_Xe[] = {
    12.1298437, 20.975, 31.05, 42.20, 54.1, 66.703, 91.6, 105.9778,
    179.84, 202.0, 229.02, 255.0, 281, 314, 343, 374, 404, 434,
    549, 582, 616, 650, 700, 736, 818, 857.0, 1493, 1571, 1653,
    1742, 1826, 1919, 2023, 2113, 2209, 2300, 2556, 2637, 2726,
    2811, 2975, 3068, 3243, 3333.8, 7660, 7889, 8144, 8382,
    8971, 9243, 9581, 9810.37, 40271.724, 41299.892
};

/* O: Z=1..8 */
static const double E_O[] = {
    13.618055, 35.12112, 54.93554, 77.41350, 113.8990, 138.1189,
    739.32683, 871.4099138
};

/* N: Z=1..7 */
static const double E_N[] = {
    14.53413, 29.60125, 47.4453, 77.4735, 97.8901, 552.06733, 667.0461377
};

static const IonSpeciesTable ION_TABLES[] = {
    {"Ar", 18, E_Ar},
    {"He",  2, E_He},
    {"H",   1, E_H},
    {"Li",  3, E_Li},
    {"Ne", 10, E_Ne},
    {"Xe", 54, E_Xe},
    {"O",   8, E_O},
    {"N",   7, E_N},
};

static const int ION_TABLES_COUNT = (int)(sizeof(ION_TABLES) / sizeof(ION_TABLES[0]));

/* -------------------- Helpers ------------------------------------------ */

static const IonSpeciesTable *find_species(const char *species)
{
    if (!species) return NULL;
    for (int i = 0; i < ION_TABLES_COUNT; ++i) {
        if (strcmp(species, ION_TABLES[i].name) == 0) return &ION_TABLES[i];
    }
    return NULL;
}

int ADK_ionization_energy(double *E_eV_out, const char *species, int Z)
{
    if (!E_eV_out) return 1;
    const IonSpeciesTable *tab = find_species(species);
    if (!tab) return 2;
    if (Z < 1 || Z > tab->Zmax) return 3;

    *E_eV_out = tab->E[Z - 1];
    return 0;
}

/* Direct translation of your Python ADKModel._scalar_rate using tgamma(). */
double ADK_scalar_rate(double E_abs, double ion_ene_eV, int Z)
{
    double E = fabs(E_abs);
    if (E <= 0.0) return 0.0;

    /* n = 3.69 * Z / sqrt(ion_ene) */
    double n = 3.69 * (double)Z / sqrt(ion_ene_eV);

    double C = 2.0 * n - 1.0;
    /* A = 1.52 * 4^n * ion_ene * (20.5 * ion_ene^1.5)^C / (n * gamma(2n)) */
    double A = 1.52
             * pow(4.0, n)
             * ion_ene_eV
             * pow(20.5 * pow(ion_ene_eV, 1.5), C)
             / (n * tgamma(2.0 * n));

    /* B = 6.83 * ion_ene^1.5 */
    double B = 6.83 * pow(ion_ene_eV, 1.5);

    return A / pow(E, C) * exp(-B / E);
}

/* -------------------- IonizationModel vtable --------------------------- */

static double adk_rate_impl(const IonizationModel *base,
                            double E_abs, const char *species, int Z)
{
    (void)base;

    double ion_ene_eV;
    int rc = ADK_ionization_energy(&ion_ene_eV, species, Z);
    if (rc != 0) {
        /* For invalid inputs, return 0.0 (non-fatal) */
        return 0.0;
    }
    return ADK_scalar_rate(E_abs, ion_ene_eV, Z);
}

static void adk_destroy_impl(IonizationModel *base)
{
    (void)base; /* no heap allocations */
}

static const IonizationModelVTable ADK_VT = {
    .rate    = adk_rate_impl,
    .destroy = adk_destroy_impl
};

void ADKModel_init(ADKModel *m)
{
    m->base.vt = &ADK_VT;
}

