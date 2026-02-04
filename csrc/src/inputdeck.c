#include "inputdeck.h"
#include "ionization_model.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// tomlc99
#include "toml.h"

/* -------------------------- TOML getters (forward decls) -------------------------- */
static int get_bool(toml_table_t *tab, const char *key, bool *out);
static char *get_string_dup(toml_table_t *tab, const char *key);
static int get_double(toml_table_t *tab, const char *key, double *out);

/* ----------------------------- utilities ----------------------------- */

static int derive_n_from_dx(double amin, double amax, double dx, int *n_out)
{
    if (!(dx > 0.0))
        return 1;
    if (!(amax > amin))
        return 2;

    // number of points including endpoints:
    // n = floor((amax-amin)/dx + 0.5) + 1  (rounded)
    double span = amax - amin;
    double cells = span / dx;
    long n = (long)llround(cells) + 1;

    if (n < 2)
        return 3;
    if (n > 2147483647L)
        return 4;

    *n_out = (int)n;
    return 0;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p)
    {
        fprintf(stderr, "fatal: out of memory\n");
        exit(1);
    }
    return p;
}

static void *xrealloc(void *p, size_t n)
{
    void *q = realloc(p, n);
    if (!q)
    {
        fprintf(stderr, "fatal: out of memory\n");
        exit(1);
    }
    return q;
}

static char *read_entire_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

    if (fseek(f, 0, SEEK_END) != 0)
    {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0)
    {
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0)
    {
        fclose(f);
        return NULL;
    }

    char *buf = (char *)xmalloc((size_t)n + 1);
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);

    if (got != (size_t)n)
    {
        free(buf);
        return NULL;
    }
    buf[n] = '\0';
    return buf;
}

static const char *axis_to_str(Axis a)
{
    switch (a)
    {
    case AXIS_X:
        return "x";
    case AXIS_Y:
        return "y";
    case AXIS_Z:
        return "z";
    default:
        return "?";
    }
}

static bool parse_axis_char(char c, Axis *out)
{
    c = (char)tolower((unsigned char)c);
    if (c == 'x')
    {
        *out = AXIS_X;
        return true;
    }
    if (c == 'y')
    {
        *out = AXIS_Y;
        return true;
    }
    if (c == 'z')
    {
        *out = AXIS_Z;
        return true;
    }
    return false;
}

static int parse_diag_axes(const char *s, char axes_out[3])
{
    if (!s)
        return 1;
    size_t n = strlen(s);
    if (n != 1 && n != 2)
        return 2;

    char a0 = (char)tolower((unsigned char)s[0]);
    if (!(a0 == 't' || a0 == 'x' || a0 == 'y' || a0 == 'z'))
        return 3;

    if (n == 1)
    {
        axes_out[0] = a0;
        axes_out[1] = '\0';
        axes_out[2] = '\0';
        return 0;
    }

    char a1 = (char)tolower((unsigned char)s[1]);
    if (!(a1 == 't' || a1 == 'x' || a1 == 'y' || a1 == 'z'))
        return 4;
    if (a1 == a0)
        return 5;

    axes_out[0] = a0;
    axes_out[1] = a1;
    axes_out[2] = '\0';
    return 0;
}

static int parse_component_token(const char *s, char out[4])
{
    /* Expect exactly: 'E' or 'A' followed by 'x','y','z' (case-insensitive) */
    if (!s)
        return 1;
    if (strlen(s) != 2)
        return 2;

    char c0 = (char)toupper((unsigned char)s[0]);
    char c1 = (char)tolower((unsigned char)s[1]);

    if (!(c0 == 'E' || c0 == 'A'))
        return 3;
    if (!(c1 == 'x' || c1 == 'y' || c1 == 'z'))
        return 4;

    out[0] = c0;
    out[1] = c1;
    out[2] = '\0';
    out[3] = '\0';
    return 0;
}

static bool parse_spatial_axes(const char *s, Axis *ax1, Axis *ax2, bool *has_ax2)
{
    if (!s)
        return false;
    size_t n = strlen(s);
    if (n != 1 && n != 2)
        return false;

    if (!parse_axis_char(s[0], ax1))
        return false;
    if (n == 1)
    {
        *has_ax2 = false;
        *ax2 = AXIS_X;
        return true;
    }

    if (!parse_axis_char(s[1], ax2))
        return false;
    if (*ax1 == *ax2)
        return false;
    *has_ax2 = true;
    return true;
}

static int parse_components_array(toml_table_t *t, const char *key,
                                  FieldDiagSpec *d, int diag_index)
{
    toml_array_t *arr = toml_array_in(t, key);
    if (!arr)
    {
        fprintf(stderr, "inputdeck: field_diag[%d] missing key %s (expected array of strings)\n",
                diag_index, key);
        return 1;
    }

    int n = toml_array_nelem(arr);
    if (n <= 0)
    {
        fprintf(stderr, "inputdeck: field_diag[%d] %s is empty\n", diag_index, key);
        return 2;
    }
    if (n > 8)
    {
        fprintf(stderr, "inputdeck: field_diag[%d] too many components (%d), max=8\n",
                diag_index, n);
        return 3;
    }

    d->ncomp = 0;
    for (int i = 0; i < n; ++i)
    {
        toml_datum_t ds = toml_string_at(arr, i);
        if (!ds.ok || !ds.u.s)
        {
            fprintf(stderr, "inputdeck: field_diag[%d] components[%d] must be a string\n",
                    diag_index, i);
            return 4;
        }

        char token[4] = {0, 0, 0, 0};
        int rc = parse_component_token(ds.u.s, token);
        free(ds.u.s);

        if (rc != 0)
        {
            fprintf(stderr, "inputdeck: field_diag[%d] invalid component \"%s\" (expected Ex,Ey,Ez,Ax,Ay,Az)\n",
                    diag_index, token);
            return 5;
        }

        snprintf(d->comp[d->ncomp], sizeof(d->comp[d->ncomp]), "%s", token);
        d->ncomp++;
    }

    return 0;
}

static bool streqi(const char *a, const char *b)
{
    if (!a || !b)
        return false;
    while (*a && *b)
    {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return false;
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

static int parse_sense(const char *s, PolarizationSense *out)
{
    if (streqi(s, "right") || streqi(s, "r"))
    {
        *out = SENSE_RIGHT;
        return 0;
    }
    if (streqi(s, "left") || streqi(s, "l"))
    {
        *out = SENSE_LEFT;
        return 0;
    }
    return 1;
}

/* -------------------------- field_cache defaults/parse -------------------------- */

static void run_defaults(RunSpec *r)
{
    memset(r, 0, sizeof(*r));
    snprintf(r->working_dir, sizeof(r->working_dir), ".");
    snprintf(r->gas, sizeof(r->gas), "H");
    snprintf(r->ionization_model, sizeof(r->ionization_model), "adk");
}

static void field_cache_defaults(FieldCacheSpec *fc)
{
    memset(fc, 0, sizeof(*fc));
    fc->mode = FC_MODE_AUTO;
    /* default: current directory */
    snprintf(fc->out_dir, sizeof(fc->out_dir), ".");
    fc->write = true;
    fc->keep_in_memory = false;
    fc->block_t = 64;
}

static int get_int(toml_table_t *tab, const char *key, int *out)
{
    toml_datum_t d = toml_int_in(tab, key);
    if (!d.ok)
        return 0;
    if (d.u.i < -2147483648LL || d.u.i > 2147483647LL)
        return 0;
    *out = (int)d.u.i;
    return 1;
}

static int parse_field_cache_mode(const char *s, FieldCacheMode *out)
{
    if (streqi(s, "auto"))
    {
        *out = FC_MODE_AUTO;
        return 0;
    }
    if (streqi(s, "compute") || streqi(s, "calc") || streqi(s, "calculate"))
    {
        *out = FC_MODE_COMPUTE;
        return 0;
    }
    if (streqi(s, "load") || streqi(s, "read"))
    {
        *out = FC_MODE_LOAD;
        return 0;
    }
    if (streqi(s, "off") || streqi(s, "none") || streqi(s, "disable"))
    {
        *out = FC_MODE_OFF;
        return 0;
    }
    return 1;
}

static int parse_run(toml_table_t *root, RunSpec *r)
{
    toml_table_t *tr = toml_table_in(root, "run");
    if (!tr)
        return 0; /* optional */

    char *wd = get_string_dup(tr, "working_dir");
    if (wd)
    {
        if (wd[0] == '\0')
        {
            fprintf(stderr, "inputdeck: [run] working_dir must not be empty\n");
            free(wd);
            return 1;
        }
        snprintf(r->working_dir, sizeof(r->working_dir), "%s", wd);
        free(wd);
    }
    {
        toml_datum_t d = toml_string_in(tr, "gas");
        if (d.ok)
        {
            snprintf(r->gas, sizeof(r->gas), "%s", d.u.s);
            free(d.u.s);
        }
    }

    {
        toml_datum_t d = toml_string_in(tr, "ionization_model");
        if (d.ok)
        {
            snprintf(r->ionization_model, sizeof(r->ionization_model), "%s", d.u.s);
            free(d.u.s);
        }
    }

    if (strcmp(r->ionization_model, "adk") != 0)
    {
        fprintf(stderr,
                "inputdeck: [run].ionization_model=\"%s\" is not supported (only \"adk\" is available).\n",
                r->ionization_model);
        return 1;
    }

    // Validate gas against ionization_model tables
    if (!ionization_species_supported(r->gas))
    {
        fprintf(stderr,
                "inputdeck: [run].gas=\"%s\" is not supported. "
                "Supported gases are those listed in ionization_model.c tables.\n",
                r->gas);
        return 1;
    }

    return 0;
}

static int parse_field_cache(toml_table_t *root, FieldCacheSpec *fc)
{
    toml_table_t *tfc = toml_table_in(root, "field_cache");
    if (!tfc)
        return 0; /* optional */

    /* mode */
    char *mode = get_string_dup(tfc, "mode");
    if (mode)
    {
        FieldCacheMode m;
        if (parse_field_cache_mode(mode, &m) != 0)
        {
            fprintf(stderr, "inputdeck: [field_cache] invalid mode=\"%s\" (expected auto|compute|load|off)\n", mode);
            free(mode);
            return 1;
        }
        fc->mode = m;
        free(mode);
    }

    /* out_dir (preferred) or legacy prefix/dir */
    char *od = get_string_dup(tfc, "out_dir");
    if (!od)
        od = get_string_dup(tfc, "dir");
    if (!od)
        od = get_string_dup(tfc, "prefix");
    if (od)
    {
        snprintf(fc->out_dir, sizeof(fc->out_dir), "%s", od);
        free(od);
    }

    (void)get_bool(tfc, "write", &fc->write);
    (void)get_bool(tfc, "keep_in_memory", &fc->keep_in_memory);

    int bt;
    if (get_int(tfc, "block_t", &bt))
    {
        if (bt <= 0)
        {
            fprintf(stderr, "inputdeck: [field_cache] block_t must be > 0\n");
            return 2;
        }
        fc->block_t = bt;
    }

    return 0;
}

static int parse_field_diag(toml_table_t *root, FieldDiagList *L)
{
    toml_array_t *arr = toml_array_in(root, "field_diag");
    if (!arr)
        return 0; /* optional */

    int n = toml_array_nelem(arr);
    if (n <= 0)
        return 0;

    L->v = (FieldDiagSpec *)xmalloc((size_t)n * sizeof(FieldDiagSpec));
    L->n = 0;

    for (int i = 0; i < n; ++i)
    {
        toml_table_t *td = toml_table_at(arr, i);
        if (!td)
        {
            fprintf(stderr, "inputdeck: field_diag[%d] is not a table\n", i);
            return 1;
        }

        FieldDiagSpec d;
        memset(&d, 0, sizeof(d));
        d.pos_x = 0.0;
        d.pos_y = 0.0;
        d.pos_z = 0.0;
        d.pos_t = 0.0;
        snprintf(d.axes, sizeof(d.axes), "t"); /* default */

        /* axes */
        char *axes = get_string_dup(td, "axes");
        if (!axes)
        {
            fprintf(stderr, "inputdeck: field_diag[%d] missing key axes\n", i);
            return 2;
        }
        int ra = parse_diag_axes(axes, d.axes);
        if (ra != 0)
        {
            fprintf(stderr, "inputdeck: field_diag[%d] invalid axes=\"%s\" (use 1 or 2 of t,x,y,z; distinct)\n",
                    i, axes);
            free(axes);
            return 3;
        }
        free(axes);

        /* components array */
        int rc = parse_components_array(td, "components", &d, i);
        if (rc != 0)
            return 4;

        /* positions */
        (void)get_double(td, "pos_x", &d.pos_x);
        (void)get_double(td, "pos_y", &d.pos_y);
        (void)get_double(td, "pos_z", &d.pos_z);
        (void)get_double(td, "pos_t", &d.pos_t);

        L->v[L->n++] = d;
    }

    return 0;
}

static int parse_ionization_frac(toml_table_t *root, IonFracSpec *s)
{
    toml_table_t *t = toml_table_in(root, "ionization_frac");
    if (!t)
    {
        s->enabled = false;
        return 0;
    }
    s->enabled = true;
    return 0;
}


/* -------------------------- deck memory -------------------------- */

static void lasers_init(InputLaserDeck *d)
{
    d->items = NULL;
    d->count = 0;
    d->capacity = 0;
}

static void lasers_push(InputLaserDeck *d, const InputLaserSpec *L)
{
    if (d->count == d->capacity)
    {
        size_t newcap = (d->capacity == 0) ? 4 : (2 * d->capacity);
        d->items = (InputLaserSpec *)xrealloc(d->items, newcap * sizeof(InputLaserSpec));
        d->capacity = newcap;
    }
    d->items[d->count++] = *L;
}

static void field_diag_init(FieldDiagList *L)
{
    L->n = 0;
    L->v = NULL;
}

static void field_diag_free(FieldDiagList *L)
{
    if (!L)
        return;
    free(L->v);
    L->v = NULL;
    L->n = 0;
}


/* -------------------------- defaults -------------------------- */

static void grid_defaults(InputGridSpec *g)
{
    memset(g, 0, sizeof(*g));
    g->t_min = -200.0;
    g->t_max = 200.0;
    g->dt = 0.02;

    g->ax1 = AXIS_Z;
    g->ax2 = AXIS_X;
    g->has_ax2 = false;

    g->fixed_x = 0.0;
    g->fixed_y = 0.0;
    g->fixed_z = 0.0;

    g->ax1_min = -5.0;
    g->ax1_max = 5.0;
    g->dx1 = 0.1;

    g->ax2_min = -5.0;
    g->ax2_max = 5.0;
    g->dx2 = 0.1;
}

static void laser_defaults(InputLaserSpec *L)
{
    memset(L, 0, sizeof(*L));
    L->type = LASER_STANDARD;

    L->E0 = 100.0;
    L->wavelength = 0.8;
    L->phase0 = 0.0;

    L->k_vec[0] = 0.0;
    L->k_vec[1] = 0.0;
    L->k_vec[2] = 1.0;
    L->r_start[0] = 0.0;
    L->r_start[1] = 0.0;
    L->r_start[2] = 0.0;
    L->use_retarded_time = true;

    L->temporal_type = TEMP_GAUSSIAN;
    L->tau = 30.0;

    L->transverse_type = TRANS_GAUSSIAN;
    L->w0 = 4.0;
    L->zf = 0.0;

    L->has_hermite = false;
    L->herm_l = 0;
    L->herm_m = 0;

    L->polarization = POL_LINEAR;
    L->angle = 0.0;

    L->sense = SENSE_RIGHT;

    L->has_jones = false;
    L->p1 = 1.0;
    L->p2 = 0.0;
    L->delta = 0.0;
}

/* -------------------------- TOML getters -------------------------- */

static int get_double(toml_table_t *tab, const char *key, double *out)
{
    toml_datum_t d = toml_double_in(tab, key);
    if (!d.ok)
        return 0;
    *out = d.u.d;
    return 1;
}

static int get_bool(toml_table_t *tab, const char *key, bool *out)
{
    toml_datum_t d = toml_bool_in(tab, key);
    if (!d.ok)
        return 0;
    *out = d.u.b ? true : false;
    return 1;
}

static char *get_string_dup(toml_table_t *tab, const char *key)
{
    toml_datum_t d = toml_string_in(tab, key);
    if (!d.ok)
        return NULL;
    // tomlc99 allocates; we return it, caller must free(d.u.s)
    return d.u.s;
}

static int get_vec3(toml_table_t *tab, const char *key, double v[3])
{
    toml_array_t *arr = toml_array_in(tab, key);
    if (!arr)
        return 0;
    if (toml_array_nelem(arr) != 3)
        return -1;

    for (int i = 0; i < 3; ++i)
    {
        toml_datum_t d = toml_double_at(arr, i);
        if (!d.ok)
            return -2;
        v[i] = d.u.d;
    }
    return 1;
}

static int get_int2(toml_table_t *tab, const char *key, int v[2])
{
    toml_array_t *arr = toml_array_in(tab, key);
    if (!arr)
        return 0;
    if (toml_array_nelem(arr) != 2)
        return -1;

    for (int i = 0; i < 2; ++i)
    {
        toml_datum_t d = toml_int_at(arr, i);
        if (!d.ok)
            return -2;
        if (d.u.i < -2147483648LL || d.u.i > 2147483647LL)
            return -3;
        v[i] = (int)d.u.i;
    }
    return 1;
}

/* -------------------------- parsing: grid -------------------------- */

static int parse_grid(toml_table_t *root, InputGridSpec *g)
{
    toml_table_t *tg = toml_table_in(root, "grid");
    if (!tg)
    {
        fprintf(stderr, "inputdeck: missing required [grid] table\n");
        return 1;
    }

    (void)get_double(tg, "t_min", &g->t_min);
    (void)get_double(tg, "t_max", &g->t_max);
    (void)get_double(tg, "dt", &g->dt);

    char *axes = get_string_dup(tg, "spatial_axes");
    if (!axes)
    {
        fprintf(stderr, "inputdeck: [grid] missing key spatial_axes\n");
        return 2;
    }
    if (!parse_spatial_axes(axes, &g->ax1, &g->ax2, &g->has_ax2))
    {
        fprintf(stderr, "inputdeck: invalid spatial_axes=\"%s\" (expected 1 or 2 distinct axes from x,y,z)\n", axes);
        free(axes);
        return 3;
    }
    free(axes);

    // fixed = { x=..., y=..., z=... }
    toml_table_t *tfixed = toml_table_in(tg, "fixed");
    if (tfixed)
    {
        (void)get_double(tfixed, "x", &g->fixed_x);
        (void)get_double(tfixed, "y", &g->fixed_y);
        (void)get_double(tfixed, "z", &g->fixed_z);
    }

    (void)get_double(tg, "ax1_min", &g->ax1_min);
    (void)get_double(tg, "ax1_max", &g->ax1_max);
    (void)get_double(tg, "dx1", &g->dx1);

    if (g->has_ax2)
    {
        (void)get_double(tg, "ax2_min", &g->ax2_min);
        (void)get_double(tg, "ax2_max", &g->ax2_max);
        (void)get_double(tg, "dx2", &g->dx2);
    }

    int rc;
    rc = derive_n_from_dx(g->ax1_min, g->ax1_max, g->dx1, &g->ax1_n);

    if (rc != 0)
    {
        fprintf(stderr, "inputdeck: [grid] invalid (ax1_min,ax1_max,dx1) combination\n");
        return 20;
    }

    if (g->has_ax2)
    {
        rc = derive_n_from_dx(g->ax2_min, g->ax2_max, g->dx2, &g->ax2_n);
        if (rc != 0)
        {
            fprintf(stderr, "inputdeck: [grid] invalid (ax2_min,ax2_max,dx2) combination\n");
            return 21;
        }
    }

    int rt;
    rt = derive_n_from_dx(g->t_min, g->t_max, g->dt, &g->t_n);

    if (rt != 0)
    {
        fprintf(stderr, "inputdeck: [grid] invalid (t_min,t_max,dt) combination\n");
        return 200;
    }

    // Basic validation
    if (!(g->dt > 0.0))
    {
        fprintf(stderr, "inputdeck: [grid] dt must be > 0\n");
        return 6;
    }
    if (!(g->t_max > g->t_min))
    {
        fprintf(stderr, "inputdeck: [grid] t_max must be > t_min\n");
        return 7;
    }

    if (!(g->ax1_max > g->ax1_min))
    {
        fprintf(stderr, "inputdeck: [grid] ax1_max must be > ax1_min\n");
        return 9;
    }
    if (g->has_ax2)
    {
        if (!(g->ax2_max > g->ax2_min))
        {
            fprintf(stderr, "inputdeck: [grid] ax2_max must be > ax2_min\n");
            return 11;
        }
    }

    return 0;
}

/* -------------------------- parsing: laser enums -------------------------- */

static int parse_laser_type(const char *s, LaserType *out)
{
    if (streqi(s, "standard"))
    {
        *out = LASER_STANDARD;
        return 0;
    }
    return 1;
}

static int parse_temporal_type(const char *s, TemporalType *out)
{
    if (streqi(s, "gaussian"))
    {
        *out = TEMP_GAUSSIAN;
        return 0;
    }
    return 1;
}

static int parse_transverse_type(const char *s, TransverseType *out)
{
    if (streqi(s, "gaussian"))
    {
        *out = TRANS_GAUSSIAN;
        return 0;
    }
    if (streqi(s, "hermite"))
    {
        *out = TRANS_HERMITE;
        return 0;
    }
    return 1;
}

static int parse_polarization(const char *s, PolarizationType *out)
{
    if (streqi(s, "linear"))
    {
        *out = POL_LINEAR;
        return 0;
    }
    if (streqi(s, "circular"))
    {
        *out = POL_CIRCULAR;
        return 0;
    }
    if (streqi(s, "jones"))
    {
        *out = POL_JONES;
        return 0;
    }
    return 1;
}

/* -------------------------- parsing: lasers -------------------------- */

static int parse_lasers(toml_table_t *root, InputLaserDeck *deck)
{
    toml_array_t *arr = toml_array_in(root, "laser");
    if (!arr)
    {
        fprintf(stderr, "inputdeck: missing required [[laser]] array (at least one laser is required)\n");
        return 1;
    }

    int n = toml_array_nelem(arr);
    if (n <= 0)
    {
        fprintf(stderr, "inputdeck: [[laser]] present but empty\n");
        return 2;
    }

    for (int i = 0; i < n; ++i)
    {
        toml_table_t *tl = toml_table_at(arr, i);
        if (!tl)
        {
            fprintf(stderr, "inputdeck: laser[%d] is not a table\n", i);
            return 3;
        }

        InputLaserSpec L;
        laser_defaults(&L);

        // type
        char *stype = get_string_dup(tl, "type");
        if (!stype)
        {
            fprintf(stderr, "inputdeck: laser[%d] missing key type\n", i);
            return 4;
        }
        if (parse_laser_type(stype, &L.type) != 0)
        {
            fprintf(stderr, "inputdeck: laser[%d] invalid type=\"%s\"\n", i, stype);
            free(stype);
            return 5;
        }
        free(stype);

        (void)get_double(tl, "E0", &L.E0);
        (void)get_double(tl, "wavelength", &L.wavelength);
        (void)get_double(tl, "phase0", &L.phase0);

        int r = get_vec3(tl, "k_vec", L.k_vec);
        if (r < 0)
        {
            fprintf(stderr, "inputdeck: laser[%d] invalid k_vec (expected 3 doubles)\n", i);
            return 6;
        }
        r = get_vec3(tl, "r_start", L.r_start);
        if (r < 0)
        {
            fprintf(stderr, "inputdeck: laser[%d] invalid r_start (expected 3 doubles)\n", i);
            return 7;
        }

        (void)get_bool(tl, "use_retarded_time", &L.use_retarded_time);

        // temporal_type
        char *tt = get_string_dup(tl, "temporal_type");
        if (tt)
        {
            if (parse_temporal_type(tt, &L.temporal_type) != 0)
            {
                fprintf(stderr, "inputdeck: laser[%d] invalid temporal_type=\"%s\"\n", i, tt);
                free(tt);
                return 8;
            }
            free(tt);
        }
        (void)get_double(tl, "tau", &L.tau);

        // transverse_type
        char *tr = get_string_dup(tl, "transverse_type");
        if (tr)
        {
            if (parse_transverse_type(tr, &L.transverse_type) != 0)
            {
                fprintf(stderr, "inputdeck: laser[%d] invalid transverse_type=\"%s\"\n", i, tr);
                free(tr);
                return 9;
            }
            free(tr);
        }
        (void)get_double(tl, "w0", &L.w0);
        (void)get_double(tl, "zf", &L.zf);

        // hermite indices
        int hm[2];
        r = get_int2(tl, "herm_lm", hm);
        if (r == 1)
        {
            L.has_hermite = true;
            L.herm_l = hm[0];
            L.herm_m = hm[1];
        }
        else if (r < 0)
        {
            fprintf(stderr, "inputdeck: laser[%d] invalid herm_lm (expected [int,int])\n", i);
            return 10;
        }

        // polarization
        char *pol = get_string_dup(tl, "polarization");
        if (pol)
        {
            if (parse_polarization(pol, &L.polarization) != 0)
            {
                fprintf(stderr, "inputdeck: laser[%d] invalid polarization=\"%s\"\n", i, pol);
                free(pol);
                return 11;
            }
            free(pol);
        }
        (void)get_double(tl, "angle", &L.angle);

        char *sen = get_string_dup(tl, "sense");
        if (sen)
        {
            if (parse_sense(sen, &L.sense) != 0)
            {
                fprintf(stderr, "inputdeck: laser[%d] invalid sense=\"%s\" (use \"right\" or \"left\")\n", i, sen);
                free(sen);
                return 18;
            }
            free(sen);
        }

        // Jones params (optional)
        // If any Jones field present, mark has_jones; also if polarization explicitly Jones.
        double tmp;
        bool any_jones = false;
        if (get_double(tl, "p1", &tmp))
        {
            L.p1 = tmp;
            any_jones = true;
        }
        if (get_double(tl, "p2", &tmp))
        {
            L.p2 = tmp;
            any_jones = true;
        }
        if (get_double(tl, "delta", &tmp))
        {
            L.delta = tmp;
            any_jones = true;
        }

        if (L.polarization == POL_JONES || any_jones)
        {
            L.has_jones = true;
        }

        // Validation
        if (!(L.E0 >= 0.0))
        {
            fprintf(stderr, "inputdeck: laser[%d] E0 must be >= 0\n", i);
            return 12;
        }
        if (!(L.wavelength > 0.0))
        {
            fprintf(stderr, "inputdeck: laser[%d] wavelength must be > 0\n", i);
            return 13;
        }
        if (!(L.tau > 0.0))
        {
            fprintf(stderr, "inputdeck: laser[%d] tau must be > 0\n", i);
            return 14;
        }
        if (!(L.w0 > 0.0))
        {
            fprintf(stderr, "inputdeck: laser[%d] w0 must be > 0\n", i);
            return 15;
        }
        if (L.transverse_type == TRANS_HERMITE && !L.has_hermite)
        {
            fprintf(stderr, "inputdeck: laser[%d] transverse_type=\"hermite\" requires herm_lm=[l,m]\n", i);
            return 16;
        }
        if (L.polarization == POL_JONES && !L.has_jones)
        {
            fprintf(stderr, "inputdeck: laser[%d] polarization=\"Jones\" requires p1,p2,delta\n", i);
            return 17;
        }
        if (L.polarization == POL_CIRCULAR && !sen)
        {
            fprintf(stderr, "inputdeck: laser[%d] polarization=\"Circular\" requires sense=\"right\" or \"left\"\n", i);
            return 19;
        }

        lasers_push(deck, &L);
    }

    return 0;
}

/* -------------------------- public API -------------------------- */

int inputdeck_read(const char *path, InputSimSpec *sim)
{
    if (!sim || !path)
        return 1;

    // init defaults
    run_defaults(&sim->run);
    grid_defaults(&sim->grid);
    lasers_init(&sim->lasers);
    field_cache_defaults(&sim->field_cache);
    field_diag_init(&sim->field_diag);
    sim->ionization_frac.enabled = false;


    char *text = read_entire_file(path);
    if (!text)
    {
        fprintf(stderr, "inputdeck: failed to read file \"%s\": %s\n", path, strerror(errno));
        return 2;
    }

    char errbuf[256];
    toml_table_t *root = toml_parse(text, errbuf, sizeof(errbuf));
    if (!root)
    {
        fprintf(stderr, "inputdeck: TOML parse error: %s\n", errbuf);
        free(text);
        return 3;
    }

    int rc = 0;

    rc = parse_run(root, &sim->run);
    if (rc != 0)
        goto done;

    rc = parse_grid(root, &sim->grid);
    if (rc != 0)
        goto done;

    rc = parse_lasers(root, &sim->lasers);
    if (rc != 0)
        goto done;

    rc = parse_field_cache(root, &sim->field_cache);
    if (rc != 0)
        goto done;

    rc = parse_field_diag(root, &sim->field_diag);
    if (rc != 0)
        goto done;

    rc = parse_ionization_frac(root, &sim->ionization_frac);
    if (rc != 0)
        goto done;

done:
    toml_free(root);
    free(text);

    if (rc != 0)
    {
        inputdeck_free(sim);
        return 10 + rc;
    }
    return 0;
}

void inputdeck_free(InputSimSpec *sim)
{
    if (!sim)
        return;
    free(sim->lasers.items);
    sim->lasers.items = NULL;
    sim->lasers.count = 0;
    sim->lasers.capacity = 0;
    field_diag_free(&sim->field_diag);
}

void inputdeck_dump(const InputSimSpec *sim)
{
    if (!sim)
        return;

    const InputGridSpec *g = &sim->grid;
    printf("[run]\n");
    printf("  working_dir=%s\n\n", sim->run.working_dir);
    /* field_cache */
    {
        const FieldCacheSpec *fc = &sim->field_cache;
        const char *m = "?";
        switch (fc->mode)
        {
        case FC_MODE_AUTO:
            m = "auto";
            break;
        case FC_MODE_COMPUTE:
            m = "compute";
            break;
        case FC_MODE_LOAD:
            m = "load";
            break;
        case FC_MODE_OFF:
            m = "off";
            break;
        }
        printf("\n[field_cache]\n");
        printf("  mode=%s\n", m);
        printf("  out_dir=%s\n", fc->out_dir);
        printf("  write=%s keep_in_memory=%s block_t=%d\n",
               fc->write ? "true" : "false",
               fc->keep_in_memory ? "true" : "false",
               fc->block_t);
    }
    printf("[grid]\n");
    printf("  t_min=%g t_max=%g dt=%g n=%d\n", g->t_min, g->t_max, g->dt, g->t_n);
    printf("  spatial_axes=%s%s\n", axis_to_str(g->ax1), g->has_ax2 ? axis_to_str(g->ax2) : "");
    printf("  fixed={x=%g,y=%g,z=%g}\n", g->fixed_x, g->fixed_y, g->fixed_z);
    printf("  ax1=[%g,%g], n=%d, dx1=%g\n", g->ax1_min, g->ax1_max, g->ax1_n, g->dx1);
    if (g->has_ax2)
        printf("  ax2=[%g,%g], n=%d, dx2=%g\n", g->ax2_min, g->ax2_max, g->ax2_n, g->dx2);

    printf("\n[lasers] count=%zu\n", sim->lasers.count);
    for (size_t i = 0; i < sim->lasers.count; ++i)
    {
        const InputLaserSpec *L = &sim->lasers.items[i];
        printf("  laser[%zu]: type=%d E0=%g lambda=%g phase0=%g\n",
               i, (int)L->type, L->E0, L->wavelength, L->phase0);
        printf("    k_vec=[%g,%g,%g] r_start=[%g,%g,%g] retarded=%s\n",
               L->k_vec[0], L->k_vec[1], L->k_vec[2],
               L->r_start[0], L->r_start[1], L->r_start[2],
               L->use_retarded_time ? "true" : "false");
        printf("    temporal=%d tau=%g transverse=%d w0=%g zf=%g\n",
               (int)L->temporal_type, L->tau, (int)L->transverse_type, L->w0, L->zf);
        if (L->has_hermite)
            printf("    herm_lm=[%d,%d]\n", L->herm_l, L->herm_m);
        printf("    pol=%d angle=%g\n", (int)L->polarization, L->angle);
        if (L->has_jones)
            printf("    Jones: p1=%g p2=%g delta=%g\n", L->p1, L->p2, L->delta);
    }
    printf("\n[ionization_frac]\n");
    printf("  enabled=%s\n", sim->ionization_frac.enabled ? "true" : "false");
}

/* -------------------------- optional test main -------------------------- */
#ifdef INPUTDECK_TEST_MAIN
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s input.toml\n", argv[0]);
        return 2;
    }
    InputSimSpec sim;
    int rc = inputdeck_read(argv[1], &sim);
    if (rc != 0)
        return rc;
    inputdeck_dump(&sim);
    inputdeck_free(&sim);
    return 0;
}
#endif
