#ifndef INPUTDECK_H
#define INPUTDECK_H
#define FIELD_DIAG_MAX_COMP 8
#define FIELD_DIAG_COMP_STR 4

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum
    {
        AXIS_X = 0,
        AXIS_Y = 1,
        AXIS_Z = 2
    } Axis;

    typedef enum
    {
        LASER_STANDARD = 0
    } LaserType;

    typedef enum
    {
        TEMP_GAUSSIAN = 0
    } TemporalType;

    typedef enum
    {
        TRANS_GAUSSIAN = 0,
        TRANS_HERMITE = 1
    } TransverseType;

    typedef enum
    {
        POL_LINEAR = 0,
        POL_CIRCULAR = 1,
        POL_JONES = 2
    } PolarizationType;

    typedef enum
    {
        SENSE_RIGHT = 0,
        SENSE_LEFT = 1
    } PolarizationSense;

    typedef struct
    {
        // time axis
        double t_min, t_max, dt;
        int t_n;

        // spatial axes selection:
        // 1D: "x"|"y"|"z"
        // 2D: "xy","xz","yz" and permutations (e.g. "zx")
        Axis ax1, ax2;
        bool has_ax2;

        // fixed coordinates for the non-swept axes (and for convenience)
        double fixed_x, fixed_y, fixed_z;

        // axis 1 sampling
        double ax1_min, ax1_max;
        double dx1;
        int ax1_n;

        // axis 2 sampling (only if has_ax2)
        double ax2_min, ax2_max;
        double dx2;
        int ax2_n;
    } InputGridSpec;

    typedef struct
    {
        LaserType type;

        double E0;
        double wavelength;
        double phase0;

        double k_vec[3];
        double r_start[3];
        bool use_retarded_time;

        TemporalType temporal_type;
        double tau;

        TransverseType transverse_type;
        double w0;
        double zf;

        // hermite params (only meaningful for TRANS_HERMITE)
        int herm_l;
        int herm_m;
        bool has_hermite;

        PolarizationType polarization;
        double angle;

        PolarizationSense sense; // used only for POL_CIRCULAR

        // Jones params (only meaningful for POL_JONES)
        double p1, p2, delta;
        bool has_jones;
    } InputLaserSpec;

    typedef struct
    {
        InputLaserSpec *items;
        size_t count;
        size_t capacity;
    } InputLaserDeck;

    /* -------------------------- General run parameters -------------------------- */

    typedef struct
    {
        /* Base working directory for the run. "." means current directory. */
        char working_dir[256];
    } RunSpec;

    /* -------------------------- Field cache control -------------------------- */

    typedef enum
    {
        FC_MODE_AUTO = 0,    /* load if compatible, else compute */
        FC_MODE_COMPUTE = 1, /* always compute (ignore existing) */
        FC_MODE_LOAD = 2,    /* require existing compatible cache */
        FC_MODE_OFF = 3      /* do not use field cache */
    } FieldCacheMode;

    typedef struct
    {
        FieldCacheMode mode;

        /* Output directory that will contain Ex.h5, Ey.h5, ... (no prefix) */
        char out_dir[256];

        /* If computing: write HDF5 outputs */
        bool write;

        /* Future: keep computed blocks in memory for other diagnostics */
        bool keep_in_memory;

        /* Future: timesteps per block for streaming/diagnostics */
        int block_t;
    } FieldCacheSpec;

    typedef struct
    {
        /* components: array of strings like "Ex", "Ay" */
        int ncomp;
        char comp[FIELD_DIAG_MAX_COMP][FIELD_DIAG_COMP_STR];

        /* axes string: 1 or 2 chars from {t,x,y,z}; order matters */
        char axes[3]; /* "t", "xz", etc */

        /* positions for fixed coordinates */
        double pos_x, pos_y, pos_z, pos_t;
    } FieldDiagSpec;

    typedef struct
    {
        int n;
        FieldDiagSpec *v; /* dynamically allocated list */
    } FieldDiagList;

    typedef struct
    {
        RunSpec run;
        InputGridSpec grid;
        InputLaserDeck lasers;
        FieldCacheSpec field_cache;
        FieldDiagList field_diag;
        // Future: diagnostics, outputs, species, etc.
    } InputSimSpec;

    /**
     * Read TOML inputdeck from `path` and populate `sim`.
     * Returns 0 on success, nonzero on failure.
     */
    int inputdeck_read(const char *path, InputSimSpec *sim);

    /** Free internal allocations inside InputSimSpec. Safe to call multiple times. */
    void inputdeck_free(InputSimSpec *sim);

    /** Optional: print parsed parameters (for debugging). */
    void inputdeck_dump(const InputSimSpec *sim);

#ifdef __cplusplus
}
#endif

#endif // INPUTDECK_H
