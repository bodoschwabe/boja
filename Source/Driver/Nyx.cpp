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

static Real fixed_dt    = -1.0;
bool Nyx::dump_old = false;
int Nyx::verbose      = 0;
Real Nyx::change_max  = 1.1;
BCRec Nyx::phys_bc;
int Nyx::nsteps_from_plotfile = -1;
ErrorList Nyx::err_list;
Vector<AMRErrorTag> Nyx::errtags;

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
    pp_nyx.query("fixed_dt", fixed_dt);
    pp_nyx.query("change_max", change_max);
    
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

    fine_mask = 0;
}

Nyx::~Nyx ()
{
    delete fine_mask;
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

    Real cur_time  = old_level->state[Axion_Type].curTime();
    Real prev_time = old_level->state[Axion_Type].prevTime();

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

    Real cur_time  = get_level(level-1).state[Axion_Type].curTime();
    Real prev_time = get_level(level-1).state[Axion_Type].prevTime();

    Real dt_old    = (cur_time - prev_time) / (Real)parent->MaxRefRatio(level-1);

    setTimeLevel(cur_time, dt_old, dt);

    MultiFab&  Ax_new = get_new_data(Axion_Type);
    FillCoarsePatch(Ax_new, 0, cur_time, Axion_Type, 0, Ax_new.nComp());

    // We set dt to be large for this new level to avoid screwing up
    // computeNewDt.
    parent->setDtLevel(1.e100, level);
}

Real
Nyx::est_time_step ()
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
        dt_min[i] = adv_level.est_time_step();
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
    Real cur_time = state[Axion_Type].curTime();
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
            dt_max[i] = get_level(i).est_time_step();
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
            dt_level[i] = get_level(i).est_time_step();
            n_factor *= n_cycle[i];
            dt_0 = std::min(dt_0, n_factor * dt_level[i]);
        }
    }
    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001 * dt_0;
    Real cur_time = state[Axion_Type].curTime();
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
