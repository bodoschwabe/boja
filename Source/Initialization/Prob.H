#ifndef _Prob_H_
#define _Prob_H_

#include "Nyx.H"

using namespace amrex;

void prob_errtags_default(amrex::Vector<amrex::AMRErrorTag>& errtags);

void prob_initdata_state(const int i,
                         const int j,
                         const int k,
                         amrex::Array4<amrex::Real> const& state);

void prob_initdata_state_on_box(const Box& bx,
                                Array4<amrex::Real> const& state);

#endif
