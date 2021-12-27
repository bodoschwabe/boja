#include <iomanip>
#include <Nyx.H>

using namespace amrex;

void
Nyx::write_info ()
{
    int ndatalogs = parent->NumDataLogs();

    if (ndatalogs > 0)
    {

        Real stringdens;
        Vector<Vector<Real> > test_val;
        compute_axionyx_quantities(stringdens, test_val);
        MultiFab&  Ax_new = get_level(0).get_new_data(Axion_Type);

        Real time  = state[Axion_Type].curTime();
        Real dt    = parent->dtLevel(0);
        int  nstep = parent->levelSteps(0);

        if (ParallelDescriptor::IOProcessor())
        {
            std::ostream& data_loga = parent->DataLog(0);
            if (time == Nyx::string_init_time)
            {
                data_loga << std::setw( 8) <<  "#  nstep";
                data_loga << std::setw(14) <<  "       time    ";
                data_loga << std::setw(14) <<  "       dt      ";
                data_loga << std::setw(22) <<  " string_dens";
                for(int i=0; i<Nyx::test_posx.size(); i++){
                  data_loga << std::setw(14) <<  "    test_mRe   ";
                  data_loga << std::setw(14) <<  "    test_mIm   ";
                  data_loga << std::setw(14) <<  "    test_vRe   ";
                  data_loga << std::setw(14) <<  "    test_vIm   ";
                }
		data_loga << '\n';
	    }

	    data_loga << std::setw( 8) <<  nstep;
	    data_loga << std::setw(14) <<  std::setprecision(10) << time;
	    data_loga << std::setw(14) <<  std::setprecision(10) << dt;
	    data_loga << std::setw(22) <<  std::setprecision(6)   << stringdens;
	    for(int i=0; i<Nyx::test_posx.size(); i++)
	      for(int n=0 ; n<Ax_new.nComp(); n++)
		data_loga << std::setw(14) <<  std::setprecision(6) << test_val[i][n];
	    data_loga << '\n';
            }
    }
}
