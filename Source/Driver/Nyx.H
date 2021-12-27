
#ifndef _Nyx_H_
#define _Nyx_H_

#include <AMReX_BC_TYPES.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_ErrorList.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_ParmParse.H>

#define ALIGN 16
#include <Distribution.H>
#include <AlignedAllocator.h>
#include <Dfft.H>

#include <iostream>

#ifdef BL_HDF5
#include <hdf5.h>
#endif

using std::istream;
using std::ostream;

enum StateType {
  Axion_Type = 0,
  NUM_STATE_TYPE
};

static constexpr int max_prob_param = 5;

//
// AmrLevel-derived class for hyperbolic conservation equations for stellar
// media
//

class Nyx
    :
    public amrex::AmrLevel
{
public:
    //
    //Default constructor.  Builds invalid object.
    //
    Nyx();

    //
    //The basic constructor.
    //
    Nyx(amrex::Amr& papa, int lev, const amrex::Geometry& level_geom,
        const amrex::BoxArray& bl, const amrex::DistributionMapping& dm,
        amrex::Real time);

    //
    //The destructor.
    //
    virtual ~Nyx() override;

    //
    //Define data descriptors.
    //
    static void variable_setup();

    //
    //Define tagging functions.
    //
    static void error_setup();
    //
    //Cleanup data descriptors at end of run.
    //
    static void variable_cleanup();

    //
    //Initialize grid data at problem start-up.
    //
    virtual void initData() override;

    // //
    // //Initialize grid data from a plotfile at problem start-up.
    // //
    // void init_from_plotfile();

    // //
    // //Initialize grid data from a plotfile at problem start-up.
    // //
    // void ReadPlotFile(bool first, const std::string& plot_file_name, bool& rhoe_infile);

    // //
    // // Initial time in code units
    // //
    // static amrex::Real initial_time;

    // //
    // // End time in code units
    // //
    // static amrex::Real final_time;


    void icReadAndPrepareFab(std::string mfDirName, int nghost, amrex::MultiFab &mf);

  void initialize_axion_string_simulation (std::string readin_ics_fname);
  amrex::Real getInitialTimeFromHeader (std::string readin_ics_fname);
  void setInitialTime ();
  void advance_AxComplex_FD (amrex::Real time, amrex::Real dt);
  void kick_AxComplex_FD (amrex::Real time, amrex::Real dt, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new, const amrex::Real invdeltasq, int neighbors);
  void drift_AxComplex_FD (amrex::Real dt, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new);
  void rescale_field (amrex::MultiFab& mf);
  void project_velocity (amrex::MultiFab& mf_old, amrex::MultiFab& mf_new);
  void advance_initial_conditions ();
  void compute_axionyx_quantities (amrex::Real& stringdens, amrex::Vector<amrex::Vector<amrex::Real> >& test_val);
#ifdef AMREX_USE_MPI
  void prepare_and_compute_powerspectrum (amrex::Real time);
  amrex::MultiFab construct_regular_mf (amrex::Real time);
  const hacc::Distribution initialize_fft_distribution (amrex::MultiFab &mf);
  std::vector<complex_t, hacc::AlignedAllocator<complex_t, ALIGN> > initialize_fft_input_field (amrex::MultiFab &mf);
  void actually_compute_powerspectrum (hacc::Dfft& dfft, std::vector<complex_t, hacc::AlignedAllocator<complex_t, ALIGN> >& a, amrex::Real time);
#endif

    //
    //Initialize data on this level from another Nyx (during regrid).
    //
    virtual void init(amrex::AmrLevel& old) override;

    //
    // Initialize data on this level after regridding if old level did not
    // previously exist
    //
    virtual void init() override;

    //
    // Advance grids at this level in time.
    //
    virtual amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration, int ncycle) override;

    //
    // Estimate time step.
    //
    amrex::Real est_time_step();

    // //
    // // Compute initial time step.
    // //
    // amrex::Real initial_time_step();

    //
    // Compute initial `dt'.
    //
    virtual void computeInitialDt(int finest_level, int sub_cycle,
                                  amrex::Vector<int>& n_cycle,
                                  const amrex::Vector<amrex::IntVect>& ref_ratio,
                                  amrex::Vector<amrex::Real>& dt_level, amrex::Real stop_time) override;
    //
    // Compute new `dt'.
    //
    virtual void computeNewDt(int finest_level, int sub_cycle,
                              amrex::Vector<int>& n_cycle,
                              const amrex::Vector<amrex::IntVect>& ref_ratio,
                              amrex::Vector<amrex::Real>& dt_min, amrex::Vector<amrex::Real>& dt_level,
                              amrex::Real stop_time, int post_regrid_flag) override;

    //
    // Do work after timestep().
    //
    virtual void post_timestep(int iteration) override;

    //
    // Do work after `regrid()`.
    //
    virtual void post_regrid(int lbase, int new_finest) override;

    //
    // Do work after `init()`.
    //
    virtual void post_init(amrex::Real stop_time) override;

    //
    // Error estimation for regridding.
    //
    virtual void errorEst(amrex::TagBoxArray& tb, int clearval, int tagval, amrex::Real time,
                          int n_error_buf=0, int ngrow=0) override;

    //
    // Called in grid_places after other tagging routines to modify
    //   the list of tagged points
    //
    virtual void manual_tags_placement (amrex::TagBoxArray&    tags,
                                        const amrex::Vector<amrex::IntVect>& bf_lev) override;

    // Returns a amrex::MultiFab containing the derived data for this level. The user
    // is responsible for deleting this pointer when done with it. If
    // `ngrow` > 0 the amrex::MultiFab is built on the appropriately grown amrex::BoxArray.
    std::unique_ptr<amrex::MultiFab> derive(const std::string& name, amrex::Real time, int ngrow) override;

    // This version of `derive()` fills the dcomp'th component of mf with the
    // derived quantity.
    void derive(const std::string& name, amrex::Real time, amrex::MultiFab& mf, int dcomp) override;

    amrex::Real vol_weight_sum              (const std::string& name, amrex::Real time, bool masked);
    amrex::Real vol_weight_sum              (amrex::MultiFab& mf, bool masked);

  static int NUM_AX;
  static int nstep_spectrum;
  static int neighbors;
  static int AxRe;
  static int AxIm;
  static int AxvRe;
  static int AxvIm;
  static amrex::Real msa;
  static int jaxions_ic;
  static bool prsstring;
  static amrex::Real string_dt;
  static amrex::Real string_init_time;
  static amrex::Real string_stop_time;
  static amrex::Real sigmalut;
  static amrex::Real string_time1;
  static amrex::Real string_n;
  static bool string_massterm;
  static amrex::Vector<int> test_posx;
  static amrex::Vector<int> test_posy;
  static amrex::Vector<int> test_posz;
  static amrex::Vector<int>fixmodulus;

    //
    // This amrex::MultiFab is used for the level coarser than this level to mask out
    // this level.  We only build this when it is needed.
    // This amrex::MultiFab has to live on this level even though it is at the resolution
    // of the next coarser level because it must be updated whenever this level changes.
    //
    amrex::MultiFab* fine_mask;
    amrex::MultiFab* build_fine_mask();

protected:

    static void read_params();

    Nyx& get_level(int lev);

    void average_down();
    void average_down(int state_indx);

    // void build_metrics();

    // void set_small_values_given_average (amrex::Real average_dens, amrex::Real average_temp, amrex::Real a,
    //                                      amrex::Real & small_dens_inout, amrex::Real & small_temp_inout,
    //                                      amrex::Real &small_pres_inout, amrex::Real gamma_minus_1,
    //                                      amrex::Real h_species);

    void write_info();

    //
    // Static data members.
    //
    static bool dump_old;
    static int verbose;
    static amrex::Real change_max;
    static amrex::ErrorList err_list;
    static amrex::Vector<amrex::AMRErrorTag> errtags;
    static amrex::BCRec phys_bc;
    static int nsteps_from_plotfile;
};

//
// Inlines.
//

inline
Nyx&
Nyx::get_level(int my_level)
{
    return *(Nyx *) &parent->getLevel(my_level);
}

#endif /*_Nyx_H_*/
