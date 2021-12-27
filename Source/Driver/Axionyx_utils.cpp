#include <Nyx.H>

using namespace amrex;

void
Nyx::compute_axionyx_quantities (Real& stringdens, Vector<Vector<Real> >& test_val)
{
  if( (Nyx::nstep_spectrum!=-1) && (parent->levelSteps(0)%Nyx::nstep_spectrum == 0) )
    prepare_and_compute_powerspectrum(state[Axion_Type].curTime());
  stringdens = get_level(parent->finestLevel()).vol_weight_sum("axion_string", state[Axion_Type].curTime(), false)
    *pow(state[Axion_Type].curTime()/get_level(parent->finestLevel()).geom.CellSizeArray()[0],2)/6.0;
  if (stringdens == 0.0 && parent->finestLevel()>0)
    stringdens = get_level(parent->finestLevel()-1).vol_weight_sum("axion_string", state[Axion_Type].curTime(), false)
      *pow(state[Axion_Type].curTime()/get_level(parent->finestLevel()-1).geom.CellSizeArray()[0],2)/6.0;

  MultiFab& Ax_new = get_level(0).get_new_data(Axion_Type);
  test_val.resize(Nyx::test_posx.size());
  for(int i=0; i<Nyx::test_posx.size(); i++){
    test_val[i].resize(Ax_new.nComp());
    for (MFIter mfi(Ax_new,false); mfi.isValid(); ++mfi){
      const Box& bx = mfi.validbox();
      const Dim3 lo = amrex::lbound(bx);
      const Dim3 hi = amrex::ubound(bx);
      if(lo.x<=Nyx::test_posx[i] && lo.y<=Nyx::test_posy[i] && lo.z<=Nyx::test_posz[i] && Nyx::test_posx[i]<=hi.x && Nyx::test_posy[i]<=hi.y && Nyx::test_posz[i]<=hi.z){
	for(int n=0; n<Ax_new.nComp(); n++)
	  test_val[i][n] = Ax_new.array(mfi)(Nyx::test_posx[i],Nyx::test_posy[i],Nyx::test_posz[i],n);
	break;
      }}
    ParallelDescriptor::ReduceRealSum(test_val[i].dataPtr(), test_val[i].size(), ParallelDescriptor::IOProcessorNumber());
  }
}
