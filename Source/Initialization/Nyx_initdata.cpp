#include <iomanip>
#include <Nyx.H>
#include <Prob.H>
#include "compute_laplace_operator.H"

using namespace amrex;

void
Nyx::initData ()
{
    BL_PROFILE("Nyx::initData()");

    int  do_readin_ics    = 0;
    std::string readin_ics_fname;
    ParmParse pp("nyx");
    pp.query("do_readin_ics", do_readin_ics);
    pp.query("readin_ics_fname", readin_ics_fname);

    if (verbose && ParallelDescriptor::IOProcessor())
        std::cout << "Initializing the data at level " << level << '\n';

    if (do_readin_ics)
	initialize_axion_string_simulation (readin_ics_fname);
    else{

      MultiFab&  Ax_new = get_new_data(Axion_Type);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(Ax_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
	{
	  const Box& bx = mfi.tilebox();
	  const auto fab_Ax_new=Ax_new.array(mfi);
	  prob_initdata_state_on_box(bx, fab_Ax_new);
	}
      advance_initial_conditions();      
      Ax_new.FillBoundary(geom.periodicity());
    }
    
    amrex::Gpu::Device::synchronize();

    if (verbose && ParallelDescriptor::IOProcessor())
        std::cout << "Done initializing the level " << level << " data\n";
}

void Nyx::initialize_axion_string_simulation (std::string readin_ics_fname)
{
  MultiFab& Ax_new = get_new_data(Axion_Type);
  if (level == 0){
    Nyx::string_init_time = getInitialTimeFromHeader(readin_ics_fname);
    setInitialTime();
    icReadAndPrepareFab(readin_ics_fname, 0, Ax_new);
    if (ParallelDescriptor::IOProcessor())
      std::cout << "Loading Jaxions initial conditions on root grid ...done\n";
  }else{
    FillCoarsePatch(Ax_new, 0, Nyx::string_init_time, Axion_Type, 0, Ax_new.nComp());
    if (ParallelDescriptor::IOProcessor())
      std::cout << "Interpolating Jaxions initial conditions onto level "<<level<<" ...done\n";
  }
  Ax_new.FillBoundary(geom.periodicity());
}

void Nyx::icReadAndPrepareFab(std::string mfDirName, int nghost, MultiFab &mf)
{
  if (level > 0 && nghost > 0)
    {
      std::cout << "Are sure you want to do what you are doing?" << std::endl;
      amrex::Abort();
    }

  MultiFab mf_read;

  if (!mfDirName.empty() && mfDirName[mfDirName.length()-1] != '/')
    mfDirName += '/';
  std::string Level = amrex::Concatenate("Level_", level, 1);
  mfDirName.append(Level);
  mfDirName.append("/Cell");

  VisMF::Read(mf_read,mfDirName.c_str());

  if (ParallelDescriptor::IOProcessor())
    std::cout << "mf read" << '\n';

  if (mf_read.contains_nan())
    {
      for (int i = 0; i < mf_read.nComp(); i++)
        {
          if (mf_read.contains_nan(i, 1))
            {
	      std::cout << "Found NaNs in read_mf in component " << i << ". " << std::endl;
	      amrex::Abort("Nyx::init_particles: Your initial conditions contain NaNs!");
            }
        }
    }

  const auto& ba      = parent->boxArray(level);
  const auto& dm      = parent->DistributionMap(level);
  const auto& ba_read = mf_read.boxArray();
  int      nc      = mf_read.nComp();

  mf.define(ba, dm, nc, nghost);

  mf.MultiFab::ParallelCopy(mf_read,0,0,nc,0,0);

  if (! ((ba.contains(ba_read) && ba_read.contains(ba))) )
    {
      if (ParallelDescriptor::IOProcessor()){
	std::cout << "ba      :" << ba << std::endl;
	std::cout << "ba_read :" << ba_read << std::endl;
	std::cout << "Read mf and hydro mf DO NOT cover the same domain!"
                  << std::endl;
      }
      ParallelDescriptor::Barrier();
      if (ParallelDescriptor::IOProcessor()){
	amrex::Abort();
      }
    }

  mf_read.clear();

  mf.FillBoundary();
  mf.EnforcePeriodicity(geom.periodicity());

  if (mf.contains_nan())
    {
      for (int i = 0; i < mf.nComp(); i++)
        {
          if (mf.contains_nan(i, 1, nghost))
            {
	      std::cout << "Found NaNs in component " << i << ". " << std::endl;
	      amrex::Abort("Nyx::init_particles: Your initial conditions contain NaNs!");
            }
        }
    }
}

Real Nyx::getInitialTimeFromHeader (std::string readin_ics_fname)
{
  std::string mfDirName(readin_ics_fname);
  int LineInWhichTimeIsStored = 8;
  std::ifstream file(mfDirName+"/Header");
  std::string str_time;
  for (int i = 1; i <= LineInWhichTimeIsStored; i++)
    std::getline(file, str_time);
  return float(std::floor(10000.0*stof(str_time)))/10000.0;
}

void Nyx::setInitialTime ()
{
  parent->setCumTime(Nyx::string_init_time);
  // We set dt to be large for this new level to avoid screwing up computeNewDt. 
  Real dummy_dt = 1.e100;
  setTimeLevel(parent->cumTime(), dummy_dt, dummy_dt);
}

void Nyx::advance_initial_conditions()
{
  for(int t=0; t<20; t++){

    for (int k = 0; k < NUM_STATE_TYPE; k++) {
      state[k].allocOldData();
      state[k].swapTimeLevels(0.0);
    }

    MultiFab&  mf_old = get_old_data(Axion_Type);
    MultiFab&  mf_new = get_new_data(Axion_Type);
    Real time = state[Axion_Type].curTime();
    
    for (MFIter mfi(mf_old,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
	const Box& bx  = mfi.tilebox();
	auto const arr_in   = mf_old[mfi].const_array();
	auto const arr_out  = mf_new[mfi].array();
	Real diff = 0.01;
	
	ParallelFor(bx,
		    [=] AMREX_GPU_DEVICE (int i, int j, int k)
		    {
		      arr_out(i,j,k,Nyx::AxRe) = arr_in(i,j,k,Nyx::AxRe)+compute_laplace_operator(arr_in,i,j,k,Nyx::AxRe,diff,neighbors);
		      arr_out(i,j,k,Nyx::AxIm) = arr_in(i,j,k,Nyx::AxIm)+compute_laplace_operator(arr_in,i,j,k,Nyx::AxIm,diff,neighbors);
		      
		      Real phase = atan2(arr_out(i,j,k,Nyx::AxIm),arr_out(i,j,k,Nyx::AxRe));
		      arr_out(i,j,k,Nyx::AxvRe ) = cos(phase);
		      arr_out(i,j,k,Nyx::AxvIm ) = sin(phase);
		      arr_out(i,j,k,Nyx::AxRe  ) = time*arr_out(i,j,k,Nyx::AxvRe);
		      arr_out(i,j,k,Nyx::AxIm  ) = time*arr_out(i,j,k,Nyx::AxvIm);
		    });
      }
    mf_new.FillBoundary(geom.periodicity());
  }
}
