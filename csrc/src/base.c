#include "base.h"

double TransverseProfile_phase_default(const TransverseProfile *self,
                                       const double r_beam_um[3],
                                       double wavelength_um)
{
    (void)self;
    (void)r_beam_um;
    (void)wavelength_um;
    return 0.0;
}
