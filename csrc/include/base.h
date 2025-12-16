#ifndef LASERION_BASE_H
#define LASERION_BASE_H

#ifdef __cplusplus
extern "C"
{
#endif

    /* =========================
       LaserPulse interface
       ========================= */

    typedef struct LaserPulse LaserPulse;

    typedef struct LaserPulseVTable
    {
        /* E(t, r) -> E[3] */
        void (*E)(const LaserPulse *self, double t_fs, const double r_um[3], double E_out[3]);

        /* A(t, r) -> A[3] */
        void (*A)(const LaserPulse *self, double t_fs, const double r_um[3], double A_out[3]);

        /* Optional destructor for concrete implementations (can be NULL). */
        void (*destroy)(LaserPulse *self);
    } LaserPulseVTable;

    struct LaserPulse
    {
        const LaserPulseVTable *vt;
    };

    /* Convenience wrappers */
    static inline void LaserPulse_E(const LaserPulse *p, double t_fs,
                                    const double r_um[3], double E_out[3])
    {
        p->vt->E(p, t_fs, r_um, E_out);
    }

    static inline void LaserPulse_A(const LaserPulse *p, double t_fs,
                                    const double r_um[3], double A_out[3])
    {
        p->vt->A(p, t_fs, r_um, A_out);
    }

    static inline void LaserPulse_destroy(LaserPulse *p)
    {
        if (p && p->vt && p->vt->destroy)
            p->vt->destroy(p);
    }

    /* =========================
       TransverseProfile interface
       ========================= */

    typedef struct TransverseProfile TransverseProfile;

    typedef struct TransverseProfileVTable
    {
        /* envelope(r, wavelength) -> scalar */
        double (*eval)(const TransverseProfile *self,
                       const double r_beam_um[3],
                       double wavelength_um);

        /* phase(r, wavelength) -> scalar (default 0 if not provided) */
        double (*phase)(const TransverseProfile *self,
                        const double r_beam_um[3],
                        double wavelength_um);

        void (*destroy)(TransverseProfile *self);
    } TransverseProfileVTable;

    struct TransverseProfile
    {
        const TransverseProfileVTable *vt;
    };

    static inline double TransverseProfile_eval(const TransverseProfile *p,
                                                const double r_beam_um[3],
                                                double wavelength_um)
    {
        return p->vt->eval(p, r_beam_um, wavelength_um);
    }

    static inline double TransverseProfile_phase(const TransverseProfile *p,
                                                 const double r_beam_um[3],
                                                 double wavelength_um)
    {
        return p->vt->phase(p, r_beam_um, wavelength_um);
    }

    static inline void TransverseProfile_destroy(TransverseProfile *p)
    {
        if (p && p->vt && p->vt->destroy)
            p->vt->destroy(p);
    }

    /* =========================
       TemporalProfile interface
       ========================= */

    typedef struct TemporalProfile TemporalProfile;

    typedef struct TemporalProfileVTable
    {
        /* envelope(t) -> scalar */
        double (*eval)(const TemporalProfile *self, double t_fs);
        void (*destroy)(TemporalProfile *self);
    } TemporalProfileVTable;

    struct TemporalProfile
    {
        const TemporalProfileVTable *vt;
    };

    static inline double TemporalProfile_eval(const TemporalProfile *p, double t_fs)
    {
        return p->vt->eval(p, t_fs);
    }

    static inline void TemporalProfile_destroy(TemporalProfile *p)
    {
        if (p && p->vt && p->vt->destroy)
            p->vt->destroy(p);
    }

    /* =========================
       IonizationModel interface
       ========================= */

    typedef struct IonizationModel IonizationModel;

    typedef struct IonizationModelVTable
    {
        /* rate(|E|, species, Z) -> 1/fs */
        double (*rate)(const IonizationModel *self,
                       double E_abs, const char *species, int Z);

        void (*destroy)(IonizationModel *self);
    } IonizationModelVTable;

    struct IonizationModel
    {
        const IonizationModelVTable *vt;
    };

    static inline double IonizationModel_rate(const IonizationModel *m,
                                              double E_abs, const char *species, int Z)
    {
        return m->vt->rate(m, E_abs, species, Z);
    }

    static inline void IonizationModel_destroy(IonizationModel *m)
    {
        if (m && m->vt && m->vt->destroy)
            m->vt->destroy(m);
    }

    /* =========================
       Defaults matching base.py
       ========================= */

    /* Default phase() implementation: returns 0.0, matching TransverseProfile.phase default. */
    double TransverseProfile_phase_default(const TransverseProfile *self,
                                           const double r_beam_um[3],
                                           double wavelength_um);

#ifdef __cplusplus
}
#endif

#endif /* LASERION_BASE_H */
