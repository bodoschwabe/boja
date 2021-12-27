#include "Nyx.H"
#include <AMReX_MultiFab.H>
#include "compute_laplace_operator.H"

using namespace amrex;

Real
Nyx::advance (Real time, Real dt, int  iteration, int  ncycle)
{
    if (verbose && ParallelDescriptor::IOProcessor() ){
	std::cout << "Advancing the axions at level " << level <<  "...\n";
    }

    for (int k = 0; k < NUM_STATE_TYPE; k++) {
      state[k].allocOldData();
      state[k].swapTimeLevels(dt);
    }

    const Real* dx      = geom.CellSize();
    const Real invdeltasq  = 1.0 / dx[0] / dx[0];
    const Real dt_half = 0.5*dt;

    MultiFab&  Ax_old = get_old_data(Axion_Type);
    MultiFab&  Ax_new = get_new_data(Axion_Type);
    Ax_old.FillBoundary(geom.periodicity());

#ifdef DEBUG
    if (Ax_old.contains_nan(0, Ax_old.nComp(), 0))
      {
        for (int i = 0; i < Ax_old.nComp(); i++)
          {
            if (Ax_old.contains_nan(i,1,0))
              {
		std::cout << "Testing component i for NaNs: " << i << std::endl;
		amrex::Abort("Ax_old has NaNs in this component::advance_AxComplex_FD()");
              }
          }
      }

    if (Ax_old.contains_nan(0, Ax_old.nComp(), Ax_old.nGrow()))
      {
        for (int i = 0; i < Ax_old.nComp(); i++)
          {
            if (Ax_old.contains_nan(i,1,Ax_old.nGrow()))
              {
		std::cout << "Testing component i for NaNs: " << i << std::endl;
		amrex::Abort("Ax_old has ghost NaNs in this component::advance_AxComplex_FD()");
              }
          }
      }
#endif

    rescale_field(Ax_old);
    kick_AxComplex_FD(time, dt_half, Ax_old, Ax_new, invdeltasq, neighbors);
    // project_velocity(Ax_old, Ax_new);
    drift_AxComplex_FD(dt, Ax_old, Ax_new);
    // project_velocity(Ax_new, Ax_new);
    kick_AxComplex_FD(time+dt, dt_half, Ax_new, Ax_new, invdeltasq, neighbors);
    rescale_field(Ax_new);

#ifdef DEBUG
    if (Ax_new.contains_nan(0, Ax_new.nComp(), 0))
      {
        for (int i = 0; i < Ax_new.nComp(); i++)
          {
            if (Ax_new.contains_nan(i,1,0))
              {
		std::cout << "Testing component i for NaNs: " << i << std::endl;
		amrex::Abort("Ax_new has NaNs in this component::advance_AxComplex_FD()");
              }
          }
      }

    if (Ax_new.contains_nan(0, Ax_new.nComp(), Ax_new.nGrow()))
      {
        for (int i = 0; i < Ax_new.nComp(); i++)
          {
            if (Ax_new.contains_nan(i,1,Ax_new.nGrow()))
              {
		std::cout << "Testing component i for NaNs: " << i << std::endl;
		amrex::Abort("Ax_new has NaNs in this component::advance_AxComplex_FD()");
              }
          }
      }
#endif
    return dt;
}


void Nyx::kick_AxComplex_FD(Real time, Real dt, MultiFab&  mf_old, MultiFab&  mf_new, const Real invdeltasq, int neighbors)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(mf_old,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
      const Box& bx  = mfi.tilebox();
      auto const arr_old  = mf_old[mfi].const_array();
      auto const arr_new  = mf_new[mfi].array();
      Real lprs = 1.0;
      if(Nyx::prsstring)
	lprs = pow(Nyx::msa/time,2)/2.0;
      else{
	if(Nyx::string_stop_time<=0.0) amrex::Abort("physstring needs string_stop_time>0");
	lprs = pow(Nyx::msa/Nyx::string_stop_time,2)/2.0;
      }
      
      if(Nyx::fixmodulus[level])
	ParallelFor(bx,
		  [=] AMREX_GPU_DEVICE (int i, int j, int k)
		  {
		    arr_new(i,j,k,Nyx::AxvRe) = arr_old(i,j,k,Nyx::AxvRe)*(time-dt)/(time+dt)-time*dt/(time+dt)
		      *(arr_old(i,j,k,Nyx::AxRe)*compute_laplace_operator(arr_old,i,j,k,Nyx::AxIm,invdeltasq,neighbors)
			-arr_old(i,j,k,Nyx::AxIm)*compute_laplace_operator(arr_old,i,j,k,Nyx::AxRe,invdeltasq,neighbors)		    
			-string_massterm*pow(time/Nyx::string_time1,Nyx::string_n+2.0)/Nyx::string_time1/Nyx::string_time1
                        *arr_old(i,j,k,Nyx::AxIm))*arr_old(i,j,k,Nyx::AxIm);

		    arr_new(i,j,k,Nyx::AxvIm) = arr_old(i,j,k,Nyx::AxvIm)*(time-dt)/(time+dt)+time*dt/(time+dt)
		      *(arr_old(i,j,k,Nyx::AxRe)*compute_laplace_operator(arr_old,i,j,k,Nyx::AxIm,invdeltasq,neighbors)
			-arr_old(i,j,k,Nyx::AxIm)*compute_laplace_operator(arr_old,i,j,k,Nyx::AxRe,invdeltasq,neighbors)
			-string_massterm*pow(time/Nyx::string_time1,Nyx::string_n+2.0)/Nyx::string_time1/Nyx::string_time1
                        *arr_old(i,j,k,Nyx::AxIm))*arr_old(i,j,k,Nyx::AxRe);
		  });
      else
	ParallelFor(bx,
		  [=] AMREX_GPU_DEVICE (int i, int j, int k)
		  {
		    arr_new(i,j,k,Nyx::AxvRe) = arr_old(i,j,k,Nyx::AxvRe)*(time-dt)/(time+dt)+time*dt/(time+dt)
		      *(compute_laplace_operator(arr_old,i,j,k,Nyx::AxRe,invdeltasq,neighbors)
			-lprs*time*time*(arr_old(i,j,k,Nyx::AxRe)*arr_old(i,j,k,Nyx::AxRe)+arr_old(i,j,k,Nyx::AxIm)*arr_old(i,j,k,Nyx::AxIm)-1)*arr_old(i,j,k,Nyx::AxRe)
			+2.0*string_massterm*pow(time/Nyx::string_time1,Nyx::string_n+2.0)/Nyx::string_time1/Nyx::string_time1*arr_old(i,j,k,Nyx::AxIm)*arr_old(i,j,k,Nyx::AxIm)
			*pow(arr_old(i,j,k,Nyx::AxRe)*arr_old(i,j,k,Nyx::AxRe)+arr_old(i,j,k,Nyx::AxIm)*arr_old(i,j,k,Nyx::AxIm),-1.5));
		    
		    arr_new(i,j,k,Nyx::AxvIm) = arr_old(i,j,k,Nyx::AxvIm)*(time-dt)/(time+dt)+time*dt/(time+dt)
		      *(compute_laplace_operator(arr_old,i,j,k,Nyx::AxIm,invdeltasq,neighbors)
			-lprs*time*time*(arr_old(i,j,k,Nyx::AxRe)*arr_old(i,j,k,Nyx::AxRe)+arr_old(i,j,k,Nyx::AxIm)*arr_old(i,j,k,Nyx::AxIm)-1)*arr_old(i,j,k,Nyx::AxIm)
			-2.0*string_massterm*pow(time/Nyx::string_time1,Nyx::string_n+2.0)/Nyx::string_time1/Nyx::string_time1*arr_old(i,j,k,Nyx::AxRe)*arr_old(i,j,k,Nyx::AxIm)
			*pow(arr_old(i,j,k,Nyx::AxRe)*arr_old(i,j,k,Nyx::AxRe)+arr_old(i,j,k,Nyx::AxIm)*arr_old(i,j,k,Nyx::AxIm),-1.5));
		  });
      }
  mf_new.FillBoundary(geom.periodicity());
}


void Nyx::drift_AxComplex_FD(Real dt, MultiFab&  mf_old, MultiFab&  mf_new)
{
  for (MFIter mfi(mf_new,TilingIfNotGPU()); mfi.isValid(); ++mfi){
    auto const arr_old = mf_old.array(mfi);
    auto const arr_new = mf_new.array(mfi);
    const Box& bx = mfi.tilebox();
    if(Nyx::fixmodulus[level])
      ParallelFor(bx,
		  [=] AMREX_GPU_DEVICE (int i, int j, int k)
		  {
		    Real phase = (arr_old(i,j,k,Nyx::AxRe)*arr_new(i,j,k,Nyx::AxvIm)-arr_old(i,j,k,Nyx::AxIm)*arr_new(i,j,k,Nyx::AxvRe))*dt;
		    arr_new(i,j,k,Nyx::AxRe) = arr_old(i,j,k,Nyx::AxRe)*std::cos(phase)-arr_old(i,j,k,Nyx::AxIm)*std::sin(phase);
		    arr_new(i,j,k,Nyx::AxIm) = arr_old(i,j,k,Nyx::AxRe)*std::sin(phase)+arr_old(i,j,k,Nyx::AxIm)*std::cos(phase);
		  });
    else
      ParallelFor(bx,
		  [=] AMREX_GPU_DEVICE (int i, int j, int k)
		  {
		    arr_new(i,j,k,Nyx::AxRe) = arr_old(i,j,k,Nyx::AxRe) + arr_new(i,j,k,Nyx::AxvRe)*dt;
		    arr_new(i,j,k,Nyx::AxIm) = arr_old(i,j,k,Nyx::AxIm) + arr_new(i,j,k,Nyx::AxvIm)*dt;
		  });
  }
  mf_new.FillBoundary(geom.periodicity());
}


void Nyx::rescale_field(MultiFab& mf)
{

  if(!Nyx::fixmodulus[level])
    return;

  for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi){
    auto const arr = mf.array(mfi);
    const Box& bx = mfi.tilebox();
    ParallelFor(bx,
		[=] AMREX_GPU_DEVICE (int i, int j, int k)
		{
		  Real amp = std::sqrt(arr(i,j,k,Nyx::AxRe)*arr(i,j,k,Nyx::AxRe)+arr(i,j,k,Nyx::AxIm)*arr(i,j,k,Nyx::AxIm));
		  arr(i,j,k,Nyx::AxRe) /= amp;
		  arr(i,j,k,Nyx::AxIm) /= amp;
		});
  }
}


void Nyx::project_velocity(MultiFab& mf_old, MultiFab& mf_new)
{

  if(!Nyx::fixmodulus[level])
    return;

  for (MFIter mfi(mf_old,TilingIfNotGPU()); mfi.isValid(); ++mfi){
    auto const arr_old = mf_old.array(mfi);
    auto const arr_new = mf_new.array(mfi);
    const Box& bx = mfi.tilebox();
    ParallelFor(bx,
		[=] AMREX_GPU_DEVICE (int i, int j, int k)
		{
		  arr_new(i,j,k,Nyx::AxvRe) = -(arr_old(i,j,k,Nyx::AxRe)*arr_new(i,j,k,Nyx::AxvIm)-arr_old(i,j,k,Nyx::AxIm)*arr_new(i,j,k,Nyx::AxvRe))*arr_old(i,j,k,Nyx::AxIm);
		  arr_new(i,j,k,Nyx::AxvIm) =  (arr_old(i,j,k,Nyx::AxRe)*arr_new(i,j,k,Nyx::AxvIm)-arr_old(i,j,k,Nyx::AxIm)*arr_new(i,j,k,Nyx::AxvRe))*arr_old(i,j,k,Nyx::AxRe);
		});
  }
}
