#include <unistd.h>
#include <iomanip>
#include <Nyx.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_buildInfo.H>

extern std::string inputs_name;

void mt_write(std::ofstream& output);

using namespace amrex;

std::string
Nyx::thePlotFileType () const
{
    //
    // Increment this whenever the writePlotFile() format changes.
    //
    static const std::string the_plot_file_type("HyperCLaw-V1.1");
    return the_plot_file_type;
}

void
Nyx::setPlotVariables ()
{
    AmrLevel::setPlotVariables();

    ParmParse pp("nyx");
    bool plot_rank = false;
    if (pp.query("plot_rank", plot_rank))
    {
        if (plot_rank)
        {
            //
            // Write the processor ID for each grid into the plotfile
            //
            std::string proc_string = "Rank";
            parent->addDerivePlotVar(proc_string);
        }
    }
}

void
Nyx::writePlotFilePre (const std::string& /*dir*/, ostream& /*os*/)
{
    if(write_hdf5 == 1 && (parent->maxLevel() > 0))
        amrex::Error("Calling single-level hdf5 interface for multilevel code (max_level > 0)");
    if(write_skip_prepost == 1)
    {
        amrex::Print()<<"Skip writePlotFilePre"<<std::endl;
    }
}

void
Nyx::writePlotFile (const std::string& dir,
                    ostream&           os,
                    VisMF::How         how)
{

#ifdef AMREX_USE_HDF5
    if(write_hdf5==1 && parent->finestLevel() == 0)
    {
    Real cur_time = state[State_for_Time].curTime();

    std::string dir_final = dir;
    if(!amrex::AsyncOut::UseAsyncOut())
    {
        auto start_position_to_erase = dir_final.find(".temp");
        dir_final.erase(start_position_to_erase,5);
    }

    int i, n;
    //
    // The list of indices of State to write to plotfile.
    // first component of pair is state_type,
    // second component of pair is component # within the state_type
    //
    std::vector<std::pair<int,int> > plot_var_map;
    for (int typ = 0; typ < desc_lst.size(); typ++)
    {
        for (int comp = 0; comp < desc_lst[typ].nComp();comp++)
        {
            if (parent->isStatePlotVar(desc_lst[typ].name(comp))
                && desc_lst[typ].getType() == IndexType::TheCellType())
            {
                plot_var_map.push_back(std::pair<int,int>(typ, comp));
            }
        }
    }

    int num_derive = 0;
    std::list<std::string> derive_names;

    for (std::list<DeriveRec>::const_iterator it = derive_lst.dlist().begin();
         it != derive_lst.dlist().end(); ++it)
    {
        if (parent->isDerivePlotVar(it->name()))
        {
	  if (it->name() == "Rank") {
	    derive_names.push_back(it->name());
	    num_derive++;
	  } else {
	    derive_names.push_back(it->name());
	    num_derive++;
	  }
        }
    }

    int n_data_items = plot_var_map.size() + num_derive;

    //
    // We combine all of the multifabs -- state, derived, etc -- into one
    // multifab -- plotMF.
    // NOTE: we are assuming that each state variable has one component,
    // but a derived variable is allowed to have multiple components.
    int cnt = 0;
    const int nGrow = 0;
    MultiFab plotMF(grids, dmap, n_data_items, nGrow);
    MultiFab* this_dat = 0;
    Vector<std::string> varnames;
    //
    // Cull data from state variables -- use no ghost cells.
    //
    for (i = 0; i < plot_var_map.size(); i++)
    {
        int typ = plot_var_map[i].first;
        int comp = plot_var_map[i].second;
        varnames.push_back(desc_lst[typ].name(comp));
        this_dat = &state[typ].newData();
        MultiFab::Copy(plotMF, *this_dat, comp, cnt, 1, nGrow);
        cnt++;
    }
    //
    // Cull data from derived variables.
    //
    if (derive_names.size() > 0)
    {
      for (std::list<DeriveRec>::const_iterator it = derive_lst.dlist().begin();
           it != derive_lst.dlist().end(); ++it)
        {
            varnames.push_back(it->name());
            const auto& derive_dat = derive(it->name(), cur_time, nGrow);
            MultiFab::Copy(plotMF, *derive_dat, 0, cnt, 1, nGrow);
            cnt++;
        }
    }

    WriteSingleLevelPlotfileHDF5(dir_final,
                          plotMF, varnames,
                          Geom(), cur_time, nStep());
//                          const std::string &versionName,
//                          const std::string &levelPrefix,
//                          const std::string &mfPrefix,
//                          const Vector<std::string>& extra_dirs)
    }
    else
#endif
        AmrLevel::writePlotFile(dir, os, how);

}

void
Nyx::writePlotFilePost (const std::string& dir, ostream& /*os*/)
{

    Real cur_time = state[State_for_Time].curTime();

    if(write_skip_prepost == 1)
    {
        amrex::Print()<<"Skip writePlotFilePost"<<std::endl;
    }
    else
    {

        // // Write comoving_a into its own file in the particle directory
        // if (ParallelDescriptor::IOProcessor())
        // {
        //     std::string FileName = dir + "/comoving_a";
        //     std::ofstream File;
        //     File.open(FileName.c_str(), std::ios::out|std::ios::trunc);
        //     if ( ! File.good()) {
        //         amrex::FileOpenFailed(FileName);
        //     }
        //     File.precision(15);
        //     File.close();
        // }

        // if (ParallelDescriptor::IOProcessor() && use_typical_steps)
        // {
        //     std::string FileName = dir + "/first_max_steps";
        //     std::ofstream File;
        //     File.open(FileName.c_str(), std::ios::out|std::ios::trunc);
        //     if ( ! File.good()) {
        //         amrex::FileOpenFailed(FileName);
        //     }
        //     File.precision(15);
        //     File << old_max_sundials_steps << '\n';
        // }

        // if (ParallelDescriptor::IOProcessor() && use_typical_steps)
        // {
        //     std::string FileName = dir + "/second_max_steps";
        //     std::ofstream File;
        //     File.open(FileName.c_str(), std::ios::out|std::ios::trunc);
        //     if ( ! File.good()) {
        //         amrex::FileOpenFailed(FileName);
        //     }
        //     File.precision(15);
        //     File << old_max_sundials_steps << '\n';
        // }
    
    if (level == 0 && ParallelDescriptor::IOProcessor())
    {
      writeJobInfo(dir);
      writeGridsFile(dir);
    }

    // Write out all parameters into the plotfile
    if (write_parameters_in_plotfile) {
        write_parameter_file(dir);
    }
    }
    if(verbose) {

    if (level == 0)
    {
      if (cur_time == 0) {
        amrex::Print().SetPrecision(15) << "Output file " << dir << " at step " << std::to_string(nStep()) << std::endl;
      } else {
        amrex::Print().SetPrecision(15) << "Output file " << dir << " at step " << std::to_string(nStep()) << std::endl;
      }
    }
  }
}

void
Nyx::writeJobInfo (const std::string& dir)
{
        // job_info file with details about the run
        std::ofstream jobInfoFile;
        std::string FullPathJobInfoFile = dir;
        FullPathJobInfoFile += "/job_info";
        jobInfoFile.open(FullPathJobInfoFile.c_str(), std::ios::out);

        std::string PrettyLine = std::string(78, '=') + "\n";
        std::string OtherLine = std::string(78, '-') + "\n";
        std::string SkipSpace = std::string(8, ' ');

        // job information
        jobInfoFile << PrettyLine;
        jobInfoFile << " Nyx Job Information\n";
        jobInfoFile << PrettyLine;

        jobInfoFile << "inputs file: " << inputs_name << "\n\n";

        jobInfoFile << "number of MPI processes: " << ParallelDescriptor::NProcs() << "\n";
#ifdef _OPENMP
        jobInfoFile << "number of threads:       " << omp_get_max_threads() << "\n";
#endif
        jobInfoFile << "\n";
        jobInfoFile << "CPU time used since start of simulation (CPU-hours): " <<
          getCPUTime()/3600.0;

        jobInfoFile << "\n\n";

        // plotfile information
        jobInfoFile << PrettyLine;
        jobInfoFile << " Plotfile Information\n";
        jobInfoFile << PrettyLine;

        time_t now = time(0);

        // Convert now to tm struct for local timezone
        tm* localtm = localtime(&now);
        jobInfoFile   << "output data / time: " << asctime(localtm);

        char currentDir[FILENAME_MAX];
        if (getcwd(currentDir, FILENAME_MAX)) {
          jobInfoFile << "output dir:         " << currentDir << "\n";
        }

        jobInfoFile << "\n\n";

        // build information
        jobInfoFile << PrettyLine;
        jobInfoFile << " Build Information\n";
        jobInfoFile << PrettyLine;

        jobInfoFile << "build date:    " << buildInfoGetBuildDate() << "\n";
        jobInfoFile << "build machine: " << buildInfoGetBuildMachine() << "\n";
        jobInfoFile << "build dir:     " << buildInfoGetBuildDir() << "\n";
        jobInfoFile << "AMReX dir:     " << buildInfoGetAMReXDir() << "\n";

        jobInfoFile << "\n";

        jobInfoFile << "COMP:          " << buildInfoGetComp() << "\n";
        jobInfoFile << "COMP version:  " << buildInfoGetCompVersion() << "\n";

        jobInfoFile << "\n";

        jobInfoFile << "C++ compiler:  " << buildInfoGetCXXName() << "\n";
        jobInfoFile << "C++ flags:     " << buildInfoGetCXXFlags() << "\n";

        jobInfoFile << "\n";

        jobInfoFile << "Fortran comp:  " << buildInfoGetFName() << "\n";
        jobInfoFile << "Fortran flags: " << buildInfoGetFFlags() << "\n";

        jobInfoFile << "\n";

        jobInfoFile << "Link flags:    " << buildInfoGetLinkFlags() << "\n";
        jobInfoFile << "Libraries:     " << buildInfoGetLibraries() << "\n";

        jobInfoFile << "\n";

        const char* githash1 = buildInfoGetGitHash(1);
        const char* githash2 = buildInfoGetGitHash(2);
        if (strlen(githash1) > 0) {
          jobInfoFile << "Nyx    git hash: " << githash1 << "\n";
        }
        if (strlen(githash2) > 0) {
          jobInfoFile << "AMReX git hash:  " << githash2 << "\n";
        }

        jobInfoFile << "\n\n";

        // grid information
        jobInfoFile << PrettyLine;
        jobInfoFile << " Grid Information\n";
        jobInfoFile << PrettyLine;

        int f_lev = parent->finestLevel();

        for (int i = 0; i <= f_lev; i++)
          {
            jobInfoFile << " level: " << i << "\n";
            jobInfoFile << "   number of boxes = " << parent->numGrids(i) << "\n";
            jobInfoFile << "   maximum zones   = ";
            for (int n = 0; n < AMREX_SPACEDIM; n++)
              {
                jobInfoFile << parent->Geom(i).Domain().length(n) << " ";
                //jobInfoFile << parent->Geom(i).ProbHi(n) << " ";
              }
            jobInfoFile << "\n\n";
          }

        jobInfoFile << " Boundary conditions\n";
        Vector<int> lo_bc_out(AMREX_SPACEDIM), hi_bc_out(AMREX_SPACEDIM);
        ParmParse pp("nyx");
        pp.getarr("lo_bc",lo_bc_out,0,AMREX_SPACEDIM);
        pp.getarr("hi_bc",hi_bc_out,0,AMREX_SPACEDIM);


        // these names correspond to the integer flags setup in the
        // Nyx_setup.cpp
        const char* names_bc[] =
          { "interior", "inflow", "outflow",
            "symmetry", "slipwall", "noslipwall" };


        jobInfoFile << "   -x: " << names_bc[lo_bc_out[0]] << "\n";
        jobInfoFile << "   +x: " << names_bc[hi_bc_out[0]] << "\n";
        if (AMREX_SPACEDIM >= 2) {
          jobInfoFile << "   -y: " << names_bc[lo_bc_out[1]] << "\n";
          jobInfoFile << "   +y: " << names_bc[hi_bc_out[1]] << "\n";
        }
        if (AMREX_SPACEDIM == 3) {
          jobInfoFile << "   -z: " << names_bc[lo_bc_out[2]] << "\n";
          jobInfoFile << "   +z: " << names_bc[hi_bc_out[2]] << "\n";
        }

        jobInfoFile << "\n\n";


        // runtime parameters
        jobInfoFile << PrettyLine;
        jobInfoFile << " Inputs File Parameters\n";
        jobInfoFile << PrettyLine;

        ParmParse::dumpTable(jobInfoFile, true);

        jobInfoFile.close();
}

void
Nyx::writeGridsFile (const std::string& dir)
{
  std::string myFname = dir;
  if (!myFname.empty() && myFname[myFname.size()-1] != '/')
    myFname += '/';
  myFname += "grids_file";

  std::ofstream gridsFile(myFname.c_str());

  int f_lev = parent->finestLevel();
  gridsFile << f_lev << '\n';

  for (int lev = 1; lev <= f_lev; lev++)
    {
      const auto& ba = get_level(lev).boxArray();
      gridsFile << ba.size() << '\n';
      // ba.coarsen(2);
      for (int i=0; i < ba.size(); i++)
	gridsFile << ba[i] << "\n";
    }
  gridsFile.close();
}

void
Nyx::write_parameter_file (const std::string& dir)
{
    if (level == 0)
    {
        if (ParallelDescriptor::IOProcessor())
        {
            std::string FileName = dir + "/the_parameters";
            std::ofstream File;
            File.open(FileName.c_str(), std::ios::out|std::ios::trunc);
            if ( ! File.good()) {
                amrex::FileOpenFailed(FileName);
            }
            File.precision(15);
            ParmParse::dumpTable(File,true);
            File.close();
        }
    }
}

void
Nyx::writeMultiFabAsPlotFile(const std::string& pltfile,
                             const MultiFab&    mf,
                             std::string        componentName)
{
    std::ofstream os;
    if (ParallelDescriptor::IOProcessor())
    {
        if( ! amrex::UtilCreateDirectory(pltfile, 0755)) {
          amrex::CreateDirectoryFailed(pltfile);
        }
        std::string HeaderFileName = pltfile + "/Header";
        os.open(HeaderFileName.c_str(), std::ios::out|std::ios::trunc|std::ios::binary);
        // The first thing we write out is the plotfile type.
        os << thePlotFileType() << '\n';
        // Just one component ...
        os << 1 << '\n';
        // ... with name
        os << componentName << '\n';
        // Dimension
        os << AMREX_SPACEDIM << '\n';
        // Time
        os << "0\n";
        // One level
        os << "0\n";
        for (int i = 0; i < AMREX_SPACEDIM; i++)
            os << Geom().ProbLo(i) << ' ';
        os << '\n';
        for (int i = 0; i < AMREX_SPACEDIM; i++)
            os << Geom().ProbHi(i) << ' ';
        os << '\n';
        // Only one level -> no refinement ratios
        os << '\n';
        // Geom
        os << parent->Geom(0).Domain() << ' ';
        os << '\n';
        os << parent->levelSteps(0) << ' ';
        os << '\n';
        for (int k = 0; k < AMREX_SPACEDIM; k++)
            os << parent->Geom(0).CellSize()[k] << ' ';
        os << '\n';
        os << (int) Geom().Coord() << '\n';
        os << "0\n"; // Write bndry data.
    }
    // Build the directory to hold the MultiFab at this level.
    // The name is relative to the directory containing the Header file.
    //
    static const std::string BaseName = "/Cell";
    std::string Level = "Level_0";
    //
    // Now for the full pathname of that directory.
    //
    std::string FullPath = pltfile;
    if ( ! FullPath.empty() && FullPath[FullPath.size()-1] != '/') {
        FullPath += '/';
    }
    FullPath += Level;
    //
    // Only the I/O processor makes the directory if it doesn't already exist.
    //
    if (ParallelDescriptor::IOProcessor()) {
        if ( ! amrex::UtilCreateDirectory(FullPath, 0755)) {
            amrex::CreateDirectoryFailed(FullPath);
        }
    }
    //
    // Force other processors to wait until directory is built.
    //
    ParallelDescriptor::Barrier();

    if (ParallelDescriptor::IOProcessor())
    {
        Real cur_time = state[State_for_Time].curTime();
        os << level << ' ' << grids.size() << ' ' << cur_time << '\n';
        os << parent->levelSteps(level) << '\n';

        for (int i = 0; i < grids.size(); ++i)
        {
            RealBox gridloc = RealBox(grids[i], geom.CellSize(), geom.ProbLo());
            for (int n = 0; n < AMREX_SPACEDIM; n++)
                os << gridloc.lo(n) << ' ' << gridloc.hi(n) << '\n';
        }
        //
        // The full relative pathname of the MultiFabs at this level.
        // The name is relative to the Header file containing this name.
        // It's the name that gets written into the Header.
        //
        std::string PathNameInHeader = Level;
        PathNameInHeader += BaseName;
        os << PathNameInHeader << '\n';
    }

    //
    // Use the Full pathname when naming the MultiFab.
    //
    std::string TheFullPath = FullPath;
    TheFullPath += BaseName;
    VisMF::Write(mf, TheFullPath);
    ParallelDescriptor::Barrier();
}

void
Nyx::checkPoint (const std::string& dir,
                 std::ostream&      os,
                 VisMF::How         how,
                 bool               dump_old)
{

  // for (int s = 0; s < desc_lst.size(); ++s) {
  //     if (dump_old && state[s].hasOldData()) {
  //         MultiFab& old_MF = get_old_data(s);
  //         amrex::prefetchToHost(old_MF);
  //     }
  //     MultiFab& new_MF = get_new_data(s);
  //     amrex::prefetchToHost(new_MF);
  // }

  AmrLevel::checkPoint(dir, os, how, dump_old);

  // for (int s = 0; s < desc_lst.size(); ++s) {
  //     if (dump_old && state[s].hasOldData()) {
  //         MultiFab& old_MF = get_old_data(s);
  //         amrex::prefetchToDevice(old_MF);
  //     }
  //     MultiFab& new_MF = get_new_data(s);
  //     amrex::prefetchToDevice(new_MF);
  // }

  if (level == 0 && ParallelDescriptor::IOProcessor())
  {
      writeJobInfo(dir);
      // writeGridsFile(dir);
  }

  // if (level == 0 && ParallelDescriptor::IOProcessor())
  //   {
  //     {
  //       // store elapsed CPU time
  //       std::ofstream CPUFile;
  //       std::string FullPathCPUFile = dir;
  //       FullPathCPUFile += "/CPUtime";
  //       CPUFile.open(FullPathCPUFile.c_str(), std::ios::out);

  //       CPUFile << std::setprecision(15) << getCPUTime();
  //       CPUFile.close();
  //     }
  //   }
}

// int
// Nyx::updateInSitu ()
// {
// #if defined(BL_USE_SENSEI_INSITU) || defined(AMREX_USE_ASCENT)
//     BL_PROFILE("Nyx::UpdateInSitu()");

// #if defined(BL_USE_SENSEI_INSITU)
//     if (insitu_bridge && insitu_bridge->update(this))
//     {
//         amrex::ErrorStream() << "Amr::updateInSitu : Failed to update." << std::endl;
//         amrex::Abort();
//     }
// #endif

// #ifdef REEBER
//     amrex::Vector<Halo>& reeber_halos);
//     halo_find(parent->dtLevel(level), reeber_halos);
//     halo_print(reeber_halos);
// #endif

// #endif
//     return 0;
// }

void
Nyx::checkPointPre (const std::string& /*dir*/,
                    std::ostream&      /*os*/)
{}

void
Nyx::checkPointPost (const std::string& dir,
                     std::ostream&      /*os*/)
{}
