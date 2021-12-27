#include <iomanip>
#include <algorithm>
#include <vector>
#include <iostream>
#include <string>

using std::cout;
using std::cerr;
using std::endl;
using std::istream;
using std::ostream;
using std::pair;
using std::string;

#include <AMReX_CONSTANTS.H>
#include <Nyx.H>
#include <AMReX_VisMF.H>
#include <AMReX_TagBox.H>
#include <AMReX_Utility.H>
#include <AMReX_Print.H>

#ifdef BL_USE_MPI
#include <MemInfo.H>
#endif

using namespace amrex;

static int sum_interval = -1;
static int  max_temp_dt = -1;

static Real fixed_dt    = -1.0;
static Real initial_dt  = -1.0;
static Real dt_cutoff   =  0;

int Nyx::load_balance_int = -1;
amrex::Real Nyx::load_balance_start_z = 15;
int Nyx::load_balance_wgt_strategy = 0;
int Nyx::load_balance_wgt_nmax = -1;
int Nyx::load_balance_strategy = DistributionMapping::SFC;

bool Nyx::dump_old = false;
int Nyx::verbose      = 0;

Real Nyx::cfl = 0.8;
Real Nyx::init_shrink = 1.0;
Real Nyx::change_max  = 1.1;

BCRec Nyx::phys_bc;
int Nyx::do_reflux = 1;
int Nyx::NUM_STATE = -1;

int Nyx::nsteps_from_plotfile = -1;

ErrorList Nyx::err_list;
Vector<AMRErrorTag> Nyx::errtags;

int Nyx::minimize_memory = 0;
int Nyx::shrink_to_fit = 0;

int Nyx::State_for_Time = Axion_Type;

int Nyx::NUM_AX = -1;
int Nyx::nstep_spectrum = -1;
int Nyx::neighbors = 2;
int Nyx::AxRe   = -1;
int Nyx::AxIm   = -1;
int Nyx::AxvRe  = -1;
int Nyx::AxvIm  = -1;
Real Nyx::msa = 1;
int  Nyx::jaxions_ic = 0;
bool Nyx::prsstring = 1;
Real Nyx::string_dt = 1.0;
Real Nyx::string_init_time = -1.0;
Real Nyx::string_stop_time = -1.0;
Real Nyx::sigmalut = 0.4;
Real Nyx::string_time1 = 1.0;
Real Nyx::string_n = 7.3;
bool Nyx::string_massterm = 1;
Vector<int> Nyx::test_posx;
Vector<int> Nyx::test_posy;
Vector<int> Nyx::test_posz;
Vector<int> Nyx::fixmodulus;

#ifdef _OPENMP
#include <omp.h>
#endif

// The default for how many digits to use for each column in the runlog
// We have a second parameter (rlp_terse) for regression-testing those quantities 
//    which show more variation than others
int Nyx::runlog_precision = 6;
int Nyx::runlog_precision_terse = 6;

int Nyx::write_parameters_in_plotfile = true;
int Nyx::write_skip_prepost = 0;
int Nyx::write_hdf5 = 0;

// this will be reset upon restart
Real         Nyx::previousCPUTimeUsed = 0.0;

Real         Nyx::startCPUTime = 0.0;

// Note: Nyx::variableSetUp is in Nyx_setup.cpp
void
Nyx::variable_cleanup ()
{
    desc_lst.clear();
}

void
Nyx::read_params ()
{
    BL_PROFILE("Nyx::read_params()");

    ParmParse pp_nyx("nyx");

    pp_nyx.query("v", verbose);
    pp_nyx.get("init_shrink", init_shrink);
    pp_nyx.get("cfl", cfl);
    pp_nyx.query("change_max", change_max);
    pp_nyx.query("fixed_dt", fixed_dt);
    pp_nyx.query("initial_dt", initial_dt);
    pp_nyx.query("max_temp_dt", max_temp_dt);
    pp_nyx.query("sum_interval", sum_interval);
    pp_nyx.get("dt_cutoff", dt_cutoff);

    pp_nyx.query("nstep_spectrum", nstep_spectrum);
    pp_nyx.query("neighbors", neighbors);
    pp_nyx.query("jaxions_ic",jaxions_ic);
    pp_nyx.query("msa", msa);
    pp_nyx.query("prsstring",prsstring);
    pp_nyx.query("string_dt", string_dt);
    pp_nyx.query("string_stop_time", string_stop_time);
    pp_nyx.query("sigmalut", sigmalut);
    pp_nyx.query("string_time1", string_time1);
    pp_nyx.query("string_n", string_n);
    pp_nyx.query("string_massterm", string_massterm);

    if (pp_nyx.contains("test_posx"))
      {
        int x = pp_nyx.countval("test_posx");
        test_posx.resize(x);
        pp_nyx.queryarr("test_posx",test_posx,0,x);
      }
    if (pp_nyx.contains("test_posy"))
      {
        int y = pp_nyx.countval("test_posy");
        test_posy.resize(y);
        pp_nyx.queryarr("test_posy",test_posy,0,y);
      }
    if (pp_nyx.contains("test_posz"))
      {
        int z = pp_nyx.countval("test_posz");
        test_posz.resize(z);
        pp_nyx.queryarr("test_posz",test_posz,0,z);
      }
    if(test_posx.size()!=test_posy.size() || test_posy.size()!= test_posz.size())
      amrex::Abort("test_posx, test_posy, test_posz have to have the same size in inputs file");
    if (pp_nyx.contains("fixmodulus"))
      {
        int nlevs = pp_nyx.countval("fixmodulus");
        fixmodulus.resize(nlevs);
        pp_nyx.queryarr("fixmodulus",fixmodulus,0,nlevs);
      }
    else
      amrex::Abort("Need to specify on which levels to fix modulus.");

    // Get boundary conditions
    Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
    pp_nyx.getarr("lo_bc", lo_bc, 0, AMREX_SPACEDIM);
    pp_nyx.getarr("hi_bc", hi_bc, 0, AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        phys_bc.setLo(i, lo_bc[i]);
        phys_bc.setHi(i, hi_bc[i]);
    }

    //
    // Check phys_bc against possible periodic geometry
    // if periodic, must have internal BC marked.
    //
    if (DefaultGeometry().isAnyPeriodic())
    {
        //
        // Do idiot check.  Periodic means interior in those directions.
        //
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
        {
            if (DefaultGeometry().isPeriodic(dir))
            {
                if (lo_bc[dir] != Interior)
                {
                    std::cerr << "Nyx::read_params:periodic in direction "
                              << dir
                              << " but low BC is not Interior" << std::endl;
                    amrex::Error();
                }
                if (hi_bc[dir] != Interior)
                {
                    std::cerr << "Nyx::read_params:periodic in direction "
                              << dir
                              << " but high BC is not Interior" << std::endl;
                    amrex::Error();
                }
            }
        }
    }
    else
    {
        //
        // Do idiot check.  If not periodic, should be no interior.
        //
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
        {
            if (lo_bc[dir] == Interior)
            {
                std::cerr << "Nyx::read_params:interior bc in direction "
                          << dir
                          << " but not periodic" << std::endl;
                amrex::Error();
            }
            if (hi_bc[dir] == Interior)
            {
                std::cerr << "Nyx::read_params:interior bc in direction "
                          << dir
                          << " but not periodic" << std::endl;
                amrex::Error();
            }
        }
    }

    pp_nyx.query("runlog_precision",runlog_precision);
    pp_nyx.query("runlog_precision_terse",runlog_precision_terse);

    pp_nyx.query("write_parameter_file",write_parameters_in_plotfile);
    if(pp_nyx.query("write_hdf5",write_hdf5))
        write_skip_prepost = write_hdf5;
    else
        pp_nyx.query("write_skip_prepost",write_skip_prepost);

#ifndef AMREX_USE_HDF5
    if(write_hdf5 == 1)
        amrex::Error("Must compile with USE_HDF5 = TRUE for write_hdf5 = 1");
#endif

#ifdef AMREX_USE_HDF5_ASYNC
    // Complete all previous async writes on amrex::Finalize()
    amrex::ExecOnFinalize(H5VLasync_waitall);
#endif

    pp_nyx.query("load_balance_int",          load_balance_int);
    pp_nyx.query("load_balance_wgt_strategy", load_balance_wgt_strategy);
    load_balance_wgt_nmax = amrex::ParallelDescriptor::NProcs();
    pp_nyx.query("load_balance_wgt_nmax",     load_balance_wgt_nmax);

    std::string theStrategy;

    if (pp_nyx.query("load_balance_strategy", theStrategy))
    {
        if (theStrategy == "SFC")
        {
            load_balance_strategy=DistributionMapping::Strategy::SFC;
        }
        else if (theStrategy == "KNAPSACK")
        {
            load_balance_strategy=DistributionMapping::Strategy::KNAPSACK;
        }
        else if (theStrategy == "ROUNDROBIN")
        {
            load_balance_strategy=DistributionMapping::Strategy::ROUNDROBIN;
        }
        else
        {
            std::string msg("Unknown strategy: ");
            msg += theStrategy;
            amrex::Warning(msg.c_str());
        }
    }
}

Nyx::Nyx ()
{
    BL_PROFILE("Nyx::Nyx()");
    fine_mask = 0;
}

Nyx::Nyx (Amr&            papa,
          int             lev,
          const Geometry& level_geom,
          const BoxArray& bl,
          const DistributionMapping& dm,
          Real            time)
    :
    AmrLevel(papa,lev,level_geom,bl,dm,time)
{
    BL_PROFILE("Nyx::Nyx(Amr)");

    MultiFab::RegionTag amrlevel_tag("AmrLevel_Level_" + std::to_string(lev));

    build_metrics();
    fine_mask = 0;
}

Nyx::~Nyx ()
{
    delete fine_mask;
}

void
Nyx::restart (Amr&     papa,
              istream& is,
              bool     b_read_special)
{
    BL_PROFILE("Nyx::restart()");
    AmrLevel::restart(papa, is, b_read_special);

    build_metrics();

    // get the elapsed CPU time to now;
    if (level == 0 && ParallelDescriptor::IOProcessor())
    {
      // get elapsed CPU time
      std::ifstream CPUFile;
      std::string FullPathCPUFile = parent->theRestartFile();
      FullPathCPUFile += "/CPUtime";
      CPUFile.open(FullPathCPUFile.c_str(), std::ios::in);

      CPUFile >> previousCPUTimeUsed;
      CPUFile.close();

      std::cout << "read CPU time: " << previousCPUTimeUsed << "\n";
    }
}

void
Nyx::build_metrics ()
{
}

void
Nyx::setTimeLevel (Real time,
                   Real dt_old,
                   Real dt_new)
{
    if (verbose && ParallelDescriptor::IOProcessor()) {
       std::cout << "Setting the current time in the state data to "
                 << parent->cumTime() << std::endl;
    }
    AmrLevel::setTimeLevel(time, dt_old, dt_new);
}

void
Nyx::init (AmrLevel& old)
{
    BL_PROFILE("Nyx::init(old)");

    MultiFab::RegionTag amrInit_tag("Init_" + std::to_string(level));
    Nyx* old_level = (Nyx*) &old;
    //
    // Create new grid data by fillpatching from old.
    //
    Real dt_new = parent->dtLevel(level);

    Real cur_time  = old_level->state[State_for_Time].curTime();
    Real prev_time = old_level->state[State_for_Time].prevTime();

    Real dt_old = cur_time - prev_time;
    setTimeLevel(cur_time, dt_old, dt_new);


    MultiFab&  Ax_new = get_new_data(Axion_Type);
    FillPatch(old, Ax_new, 0, cur_time, Axion_Type, 0, NUM_AX);

    amrex::Gpu::Device::streamSynchronize();

}

//
// This version inits the data on a new level that did not
// exist before regridding.
//
void
Nyx::init ()
{
    BL_PROFILE("Nyx::init()");
    Real dt        = parent->dtLevel(level);

    Real cur_time  = get_level(level-1).state[State_for_Time].curTime();
    Real prev_time = get_level(level-1).state[State_for_Time].prevTime();

    Real dt_old    = (cur_time - prev_time) / (Real)parent->MaxRefRatio(level-1);

    setTimeLevel(cur_time, dt_old, dt);

    MultiFab&  Ax_new = get_new_data(Axion_Type);
    FillCoarsePatch(Ax_new, 0, cur_time, Axion_Type, 0, Ax_new.nComp());

    // We set dt to be large for this new level to avoid screwing up
    // computeNewDt.
    parent->setDtLevel(1.e100, level);
}

Real
Nyx::initial_time_step ()
{
    BL_PROFILE("Nyx::initial_time_step()");
    Real dummy_dt = 0;
    Real init_dt = 0;

    if (initial_dt > 0)
    {
        init_dt = initial_dt;
    }
    else
    {
        init_dt = init_shrink * est_time_step(dummy_dt);
    }

    return init_dt;
}

Real
Nyx::est_time_step (Real /*dt_old*/)
{
    BL_PROFILE("Nyx::est_time_step()");
    if (fixed_dt > 0)
        return fixed_dt;

    // This is just a dummy value to start with
    Real est_dt = 1.0e+200;

    const Real* dx = geom.CellSize();
    Real cur_time = state[Axion_Type].curTime();
    Real dt = string_dt*dx[0]/std::sqrt(msa*msa+string_massterm*pow(cur_time/string_time1,string_n+2.0)/string_time1/string_time1+12);
    if (verbose && ParallelDescriptor::IOProcessor())
      std::cout << "...estdt from strings :  "<< dt <<'\n';
    est_dt = std::min(est_dt, dt);

    if (verbose && ParallelDescriptor::IOProcessor())
        std::cout << "Nyx::est_time_step at level "
                  << level
                  << ":  estdt = "
                  << est_dt << '\n';

    return est_dt;
}

void
Nyx::computeNewDt (int                      finest_level,
                   int                    /*sub_cycle*/,
                   Vector<int>&             n_cycle,
                   const Vector<IntVect>& /*ref_ratio*/,
                   Vector<Real>&            dt_min,
                   Vector<Real>&            dt_level,
                   Real                     stop_time,
                   int                      post_regrid_flag)
{
    BL_PROFILE("Nyx::computeNewDt()");
    //
    // We are at the start of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
    if (level > 0)
        return;

    int i;

    Real dt_0 = 1.0e+100;
    int n_factor = 1;
    for (i = 0; i <= finest_level; i++)
    {
        Nyx& adv_level = get_level(i);
        dt_min[i] = adv_level.est_time_step(dt_level[i]);
    }

    if (fixed_dt <= 0.0)
    {
        if (post_regrid_flag == 1)
        {
            //
            // Limit dt's by pre-regrid dt
            //
            for (i = 0; i <= finest_level; i++)
            {
                dt_min[i] = std::min(dt_min[i], dt_level[i]);
            }
            //
            // Find the minimum over all levels
            //
            for (i = 0; i <= finest_level; i++)
            {
                n_factor *= n_cycle[i];
                dt_0 = std::min(dt_0, n_factor * dt_min[i]);
            }
        }
        else
        {
            bool sub_unchanged=true;
            if ((parent->maxLevel() > 0) && (level == 0) &&
                (parent->subcyclingMode() == "Optimal") &&
                (parent->okToRegrid(level) || parent->levelSteps(0) == 0) )
            {
                Vector<int> new_cycle(finest_level+1);
                for (i = 0; i <= finest_level; i++)
                    new_cycle[i] = n_cycle[i];
                // The max allowable dt
                Vector<Real> dt_max(finest_level+1);
                for (i = 0; i <= finest_level; i++)
                {
                    dt_max[i] = dt_min[i];
                }
                // find the maximum number of cycles allowed.
                Vector<int> cycle_max(finest_level+1);
                cycle_max[0] = 1;
                for (i = 1; i <= finest_level; i++)
                {
                    cycle_max[i] = parent->MaxRefRatio(i-1);
                }
                // estimate the amout of work to advance each level.
                Vector<Real> est_work(finest_level+1);
                for (i = 0; i <= finest_level; i++)
                {
                    est_work[i] = parent->getLevel(i).estimateWork();
                }

                // This value will be used only if the subcycling pattern is changed.

                dt_0 = parent->computeOptimalSubcycling(finest_level+1, new_cycle.dataPtr(), dt_max.dataPtr(), 
                                                        est_work.dataPtr(), cycle_max.dataPtr());

                for (i = 0; i <= finest_level; i++)
                {
                    if (n_cycle[i] != new_cycle[i])
                    {
                        sub_unchanged = false;
                        n_cycle[i] = new_cycle[i];
                    }
                }

            }

            if (sub_unchanged)
            //
            // Limit dt's by change_max * old dt
            //
            {
                for (i = 0; i <= finest_level; i++)
                {
                    if (verbose && ParallelDescriptor::IOProcessor())
                    {
                        if (dt_min[i] > change_max*dt_level[i])
                        {
                            std::cout << "Nyx::compute_new_dt : limiting dt at level "
                                      << i << '\n';
                            std::cout << " ... new dt computed: " << dt_min[i]
                                      << '\n';
                            std::cout << " ... but limiting to: "
                                      << change_max * dt_level[i] << " = " << change_max
                                      << " * " << dt_level[i] << '\n';
                        }
                    }

                    dt_min[i] = std::min(dt_min[i], change_max * dt_level[i]);
                }
                //
                // Find the minimum over all levels
                //
                for (i = 0; i <= finest_level; i++)
                {
                    n_factor *= n_cycle[i];
                    dt_0 = std::min(dt_0, n_factor * dt_min[i]);
                }
            }
            else
            {
                if (verbose && ParallelDescriptor::IOProcessor())
                {
                   std::cout << "Nyx: Changing subcycling pattern. New pattern:\n";
                   for (i = 1; i <= finest_level; i++)
                    std::cout << "   Lev / n_cycle: " << i << " " << n_cycle[i] << '\n';
                }
            }
        }
    }
    else
    {
        dt_0 = fixed_dt;
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001 * dt_0;
    Real cur_time = state[State_for_Time].curTime();
    if (stop_time >= 0.0)
    {
        if ((cur_time + dt_0) > (stop_time - eps))
            dt_0 = stop_time - cur_time;
    }

    n_factor = 1;
    for (i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0 / n_factor;
    }
}

void
Nyx::computeInitialDt (int                      finest_level,
                       int                    /*sub_cycle*/,
                       Vector<int>&             n_cycle,
                       const Vector<IntVect>& /*ref_ratio*/,
                       Vector<Real>&            dt_level,
                       Real                     stop_time)
{
    BL_PROFILE("Nyx::computeInitialDt()");
    //
    // Grids have been constructed, compute dt for all levels.
    //
    if (level > 0)
        return;

    int i;
    Real dt_0 = 1.0e+100;
    int n_factor = 1;
    if (parent->subcyclingMode() == "Optimal")
    {
        Vector<int> new_cycle(finest_level+1);
        for (i = 0; i <= finest_level; i++)
            new_cycle[i] = n_cycle[i];
        Vector<Real> dt_max(finest_level+1);
        for (i = 0; i <= finest_level; i++)
        {
            dt_max[i] = get_level(i).initial_time_step();
        }
        // Find the maximum number of cycles allowed
        Vector<int> cycle_max(finest_level+1);
        cycle_max[0] = 1;
        for (i = 1; i <= finest_level; i++)
        {
            cycle_max[i] = parent->MaxRefRatio(i-1);
        }
        // estimate the amout of work to advance each level.
        Vector<Real> est_work(finest_level+1);
        for (i = 0; i <= finest_level; i++)
        {
            est_work[i] = parent->getLevel(i).estimateWork();
        }

        dt_0 = parent->computeOptimalSubcycling(finest_level+1, new_cycle.dataPtr(), dt_max.dataPtr(), 
                                                est_work.dataPtr(), cycle_max.dataPtr());

        for (i = 0; i <= finest_level; i++)
        {
            n_cycle[i] = new_cycle[i];
        }
        if (verbose && ParallelDescriptor::IOProcessor() && finest_level > 0)
        {
           std::cout << "Nyx: Initial subcycling pattern:\n";
           for (i = 0; i <= finest_level; i++)
               std::cout << "Level " << i << ": " << n_cycle[i] << '\n';
        }
    }
    else
    {
        for (i = 0; i <= finest_level; i++)
        {
            dt_level[i] = get_level(i).initial_time_step();
            n_factor *= n_cycle[i];
            dt_0 = std::min(dt_0, n_factor * dt_level[i]);
        }
    }
    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001 * dt_0;
    Real cur_time = state[State_for_Time].curTime();
    if (stop_time >= 0)
    {
        if ((cur_time + dt_0) > (stop_time - eps))
            dt_0 = stop_time - cur_time;
    }

    n_factor = 1;
    for (i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0 / n_factor;
    }
}

bool
Nyx::writePlotNow ()
{
  return false;
}

// bool
// Nyx::doAnalysisNow ()
// {
//     BL_PROFILE("Nyx::doAnalysisNow()");
//     return false;
// }

void
Nyx::post_timestep (int iteration)
{
    BL_PROFILE("Nyx::post_timestep()");

    MultiFab::RegionTag amrPost_tag("Post_" + std::to_string(level));

    //
    // Integration cycle on fine level grids is complete
    // do post_timestep stuff here.
    //
    int finest_level = parent->finestLevel();
    const int ncycle = parent->nCycle(level);
    BL_PROFILE_VAR("Nyx::post_timestep()::do_reflux",do_reflux);

    if (level < finest_level)
        average_down();

    amrex::Gpu::streamSynchronize();
    BL_PROFILE_VAR_STOP(do_reflux);
    BL_PROFILE_VAR("Nyx::post_timestep()::sum_write",sum_write);

    if (level == 0)
    {
        int nstep = parent->levelSteps(0);

        write_info();

#ifdef BL_USE_MPI
        // Memory monitoring:
        MemInfo* mInfo = MemInfo::GetInstance();
        char info[32];
        snprintf(info, sizeof(info), "Step %4d", nstep);
        mInfo->LogSummary(info);
#endif
    }

    amrex::Gpu::streamSynchronize();
    BL_PROFILE_VAR_STOP(sum_write);
    BL_PROFILE_VAR("Nyx::post_timestep()::compute_temp",compute_temp);

    amrex::Gpu::Device::streamSynchronize();
    BL_PROFILE_VAR_STOP(compute_temp);

}

// void
// Nyx::typical_values_post_restart (const std::string& restart_file)
// {
//     if (level > 0)
//         return;

//     if (use_typical_steps)
//     {
//       if (ParallelDescriptor::IOProcessor())
//         {
//           std::string FileName = restart_file + "/first_max_steps";
//           std::ifstream File;
//           File.open(FileName.c_str(),std::ios::in);
//           if (!File.good())
//             amrex::FileOpenFailed(FileName);
//           File >> old_max_sundials_steps;
//         }
//       ParallelDescriptor::Bcast(&old_max_sundials_steps, 1, ParallelDescriptor::IOProcessorNumber());

//       if (ParallelDescriptor::IOProcessor())
//         {
//           std::string FileName = restart_file + "/second_max_steps";
//           std::ifstream File;
//           File.open(FileName.c_str(),std::ios::in);
//           if (!File.good())
//             amrex::FileOpenFailed(FileName);
//           File >> new_max_sundials_steps;
//         }
//       ParallelDescriptor::Bcast(&new_max_sundials_steps, 1, ParallelDescriptor::IOProcessorNumber());
//     }
// }

void
Nyx::post_restart ()
{
    // BL_PROFILE("Nyx::post_restart()");

    // if (level == 0)
    //     typical_values_post_restart(parent->theRestartFile());

    // // if (inhomo_reion) init_zhi();

    // Real cur_time = state[State_for_Time].curTime();
}


void
Nyx::postCoarseTimeStep (Real cumtime)
{
   BL_PROFILE("Nyx::postCoarseTimeStep()");
   MultiFab::RegionTag amrPost_tag("Post_" + std::to_string(level));

   AmrLevel::postCoarseTimeStep(cumtime);

   if (verbose>1)
   {
       amrex::Print() << "End of postCoarseTimeStep, printing:" <<std::endl;
       MultiFab::printMemUsage();
       amrex::Arena::PrintUsage();
   }
}

void
Nyx::post_regrid (int lbase,
                  int new_finest)
{
    BL_PROFILE("Nyx::post_regrid()");
    delete fine_mask;
    fine_mask = 0;
}

void
Nyx::post_init (Real /*stop_time*/)
{
    BL_PROFILE("Nyx::post_init()");
    if (level > 0) {
        return;
    }

    // If we restarted from a plotfile, we need to reset the level_steps counter
    if ( ! parent->theRestartPlotFile().empty()) {
        parent->setLevelSteps(0,nsteps_from_plotfile);
    }

    //
    // Average data down from finer levels
    // so that conserved data is consistent between levels.
    //
    int finest_level = parent->finestLevel();
    for (int k = finest_level - 1; k >= 0; --k) {
        get_level(k).average_down();
    }

    write_info();

}

int
Nyx::okToContinue ()
{
    if (level > 0) {
        return 1;
    }

    int test = 1;
    if (parent->dtLevel(0) < dt_cutoff) {
        test = 0;
    }

    return test;
}


void
Nyx::average_down ()
{
    BL_PROFILE("Nyx::average_down()");
    if (level == parent->finestLevel()) return;
    average_down(Axion_Type);
}

void
Nyx::average_down (int state_index)
{
    BL_PROFILE("Nyx::average_down(si)");
    
    if (level == parent->finestLevel()) return;

    Nyx& fine_lev = get_level(level+1);

    const Geometry& fgeom = fine_lev.geom;
    const Geometry& cgeom =          geom;

    MultiFab& S_crse = get_new_data(state_index);
    MultiFab& S_fine = fine_lev.get_new_data(state_index);

    const int num_comps = S_fine.nComp();

    amrex::average_down(S_fine,S_crse,fgeom,cgeom,0,num_comps,fine_ratio);
}

void
Nyx::errorEst (TagBoxArray& tags,
               int          clearval,
               int          tagval,
               Real         time,
               int        /*n_error_buf*/,
               int        /*ngrow*/)
{
    BL_PROFILE("Nyx::errorEst()");

    for (int j=0; j<errtags.size(); ++j) {
      std::unique_ptr<MultiFab> mf;
      if (errtags[j].Field() != std::string()) {
        mf = std::unique_ptr<MultiFab>(derive(errtags[j].Field(), time, errtags[j].NGrow()));
      }
      errtags[j](tags,mf.get(),clearval,tagval,time,level,geom);
    }
}

std::unique_ptr<MultiFab>
Nyx::derive (const std::string& name,
             Real               time,
             int                ngrow)
{
    BL_PROFILE("Nyx::derive()");
    return AmrLevel::derive(name, time, ngrow);
}

void
Nyx::derive (const std::string& name,
             Real               time,
             MultiFab&          mf,
             int                dcomp)
{
    BL_PROFILE("Nyx::derive()");
    const auto& derive_dat = AmrLevel::derive(name, time, mf.nGrow());
    MultiFab::Copy(mf, *derive_dat, 0, dcomp, 1, mf.nGrow());
}

Real
Nyx::getCPUTime()
{

  int numCores = ParallelDescriptor::NProcs();
#ifdef _OPENMP
  numCores = numCores*omp_get_max_threads();
#endif

  Real T = numCores*(ParallelDescriptor::second() - startCPUTime) +
    previousCPUTimeUsed;

  return T;
}

void
Nyx::InitErrorList() {
}


//static Box the_same_box (const Box& b) { return b; }

void
Nyx::InitDeriveList() {
}


void
Nyx::LevelDirectoryNames(const std::string &dir,
                         const std::string &secondDir,
                         std::string &LevelDir,
                         std::string &FullPath)
{
    LevelDir = amrex::Concatenate("Level_", level, 1);
    //
    // Now for the full pathname of that directory.
    //
    FullPath = dir;
    if( ! FullPath.empty() && FullPath.back() != '/') {
        FullPath += '/';
    }
    FullPath += secondDir;
    FullPath += "/";
    FullPath += LevelDir;
}


void
Nyx::CreateLevelDirectory (const std::string &dir)
{
  AmrLevel::CreateLevelDirectory(dir);
}
