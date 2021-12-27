#include <AMReX_LevelBld.H>
#include <AMReX_buildInfo.H>

#include <Nyx.H>
#include <AxComplexDerive.H>
#include <bc_fill.H>

using namespace amrex;
using std::string;

static Box the_same_box(const Box& b)
{
    return b;
}

static Box grow_box_by_one(const Box& b)
{
    return amrex::grow(b, 1);
}

static Box grow_box_by_two(const Box& b)
{
  return amrex::grow(b, 2);
}

typedef StateDescriptor::BndryFunc BndryFunc;

//
// Components are:
//  Interior, Inflow, Outflow,  Symmetry,     SlipWall,     NoSlipWall
//
static int scalar_bc[] =
{
    INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_EVEN, REFLECT_EVEN, REFLECT_EVEN
};

static
void
set_scalar_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        bc.setLo(i, scalar_bc[lo_bc[i]]);
        bc.setHi(i, scalar_bc[hi_bc[i]]);
    }
}

void
Nyx::variable_setup()
{

    BL_ASSERT(desc_lst.size() == 0);

    if (ParallelDescriptor::IOProcessor()) {
          const char* amrex_hash  = amrex::buildInfoGetGitHash(2);
          std::cout << "\n" << "AMReX git describe: " << amrex_hash << "\n";
          const char* nyx_hash  = amrex::buildInfoGetGitHash(1);
          std::cout << "\n" << "Nyx git describe:   " << nyx_hash << "\n";
    }

    read_params();

    AxRe       = 0;
    AxIm       = 1;
    AxvRe      = 2;
    AxvIm      = 3;
    NUM_AX     = 4;

    // Note that the default is state_data_extrap = false,
    // store_in_checkpoint = true.  We only need to put these in
    // explicitly if we want to do something different,
    // like not store the state data in a checkpoint directory
    bool state_data_extrap = false;
    bool store_in_checkpoint;

    BCRec bc;

    StateDescriptor::BndryFunc bndryfunc(nyx_bcfill);
    bndryfunc.setRunOnGPU(true);

    ParmParse pp("nyx");
    int interpolator = 1;
    pp.query("interpolator", interpolator);
    Interpolater* fdminterp;
    switch(interpolator){
    case 1:
      fdminterp = &cell_bilinear_interp;
      break;
    case 2:
      fdminterp = &quartic_interp;
      break;
    default:
      amrex::Error("Dont know what interpolator to use.");
    }
    store_in_checkpoint = true;
    desc_lst.addDescriptor(Axion_Type, IndexType::TheCellType(),
                           StateDescriptor::Point, neighbors, NUM_AX,
                           fdminterp, state_data_extrap,
                           store_in_checkpoint);

    set_scalar_bc(bc, phys_bc);

    desc_lst.setComponent(Axion_Type, 0, "AxRe", bc,
                          bndryfunc);
    desc_lst.setComponent(Axion_Type, 1, "AxIm", bc,
                          bndryfunc);
    desc_lst.setComponent(Axion_Type, 2, "AxvRe", bc,
                          bndryfunc);
    desc_lst.setComponent(Axion_Type, 3, "AxvIm", bc,
                          bndryfunc);

    derive_lst.add("axion_string", IndexType::TheCellType(), 1,
                   derstring, grow_box_by_one);
    derive_lst.addComponent("axion_string", desc_lst, Axion_Type, AxRe,1);
    derive_lst.addComponent("axion_string", desc_lst, Axion_Type, AxIm,1);
    derive_lst.add("axion_velocity", IndexType::TheCellType(), 1,
                   deraxvel, grow_box_by_one);
    derive_lst.addComponent("axion_velocity", desc_lst, Axion_Type, 0,NUM_AX);
    derive_lst.add("axion_gradients", IndexType::TheCellType(), 1,
                   deraxgrad, grow_box_by_two);
    derive_lst.addComponent("axion_gradients", desc_lst, Axion_Type, AxRe,1);
    derive_lst.addComponent("axion_gradients", desc_lst, Axion_Type, AxIm,1);
    derive_lst.add("axion_dens_gradient", IndexType::TheCellType(), 1,
                   deraxgraddens, grow_box_by_two);
    derive_lst.addComponent("axion_dens_gradient", desc_lst, Axion_Type, AxRe,1);
    derive_lst.addComponent("axion_dens_gradient", desc_lst, Axion_Type, AxIm,1);
    derive_lst.add("axion_phase", IndexType::TheCellType(), 1,
                   deraxphas, the_same_box);
    derive_lst.addComponent("axion_phase", desc_lst, Axion_Type, AxRe,1);
    derive_lst.addComponent("axion_phase", desc_lst, Axion_Type, AxIm,1);
    derive_lst.add("axion_dens", IndexType::TheCellType(), 1,
                   deraxdens, the_same_box);
    derive_lst.addComponent("axion_dens", desc_lst, Axion_Type, AxRe,1);
    derive_lst.addComponent("axion_dens", desc_lst, Axion_Type, AxIm,1);
    derive_lst.add("axion_energy", IndexType::TheCellType(), 1,
                   deraxdens, grow_box_by_one);
    derive_lst.addComponent("axion_energy", desc_lst, Axion_Type, 0,NUM_AX);


    //
    // DEFINE ERROR ESTIMATION QUANTITIES
    //
    error_setup();
}

