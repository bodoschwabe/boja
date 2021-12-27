# About

AxioNyx is a publicly available extension of the Nyx code enabling cosmological simulations of scalar fields like axions, axion-like particles and the inflaton. For scientific research examples please consult the publication list. For brevity, we collectively refer to them as axion fields.

The Nyx code itself solves equations of compressible hydrodynamics on an adaptive grid
hierarchy coupled with an N-body treatment of dark matter. The gas dynamics in
Nyx uses a finite volume methodology on a set of 3-D Eulerian grids;
dark matter is represented as discrete particles moving under the influence of
gravity. Particles are evolved via a particle-mesh method, using Cloud-in-Cell
deposition/interpolation scheme. Both baryonic and dark matter contribute to
the gravitational field. In addition, Nyx includes physics needed to
accurately model the intergalactic medium: in optically thin limit and assuming
ionization equilibrium, the code calculates heating and cooling processes of the
primordial-composition gas in an ionizing ultraviolet background radiation field.
Additional physics capabilities are under development.

While Nyx can run on any Linux system in general, we particularly focus on supercomputer systems.
Nyx is parallelized with MPI + X, where X can be OpenMP on multicore architectures and
CUDA/HIP/DPC++ on hybrid CPU/GPU architectures.
In the OpenMP regime, Nyx has been successfully run at parallel concurrency
of up to 2,097,152 on NERSC's Cori-KNL. With Cuda implementation, it was run on up to
13,824 GPUs on OLCF's Summit.

More information on Nyx can be found at the [main web page](http://amrex-astro.github.io/Nyx/) and
the [online documentation](https://amrex-astro.github.io/Nyx/docs_html/).

AxioNyx extends Nyx in various ways. The axion field is implemented as an additional AxionState that is evolved using finite difference methods. On the coarsest level it can also be evolved using a spectral solver. Both the Schroedinger equation appropriate for non-relativistic, gravity dominated dynamics and the Klein-Gordan equation for cosmic axion string network simulations have been implemented. For structure formation simulations it is often more efficient to use a hybrid method combining N-body methods to model the large-scale gravitational potential around and the mass accretion onto pre-selected halos with simulations of the Schroedinger-Poisson equation to capture wave-like effects inside these halos. This method requires particles to carry phase information for the reconstruction of the wave function. They are implemented as an addition FDMphaseParticleContainer.

# Setting up a simulation

The AxioNyx source code currently lives in a [github repository](https://github.com/axionyx/axionyx_2.0.git) that can be cloned by using git:

```
git clone https://github.com/axionyx/axionyx_2.0.git --recursive
```
With ```--recursive``` the ```àmrex4axionyx_2.0``` repository is automatically cloned as a submodule to Axionyx. 

See the Nyx [Getting Started section](https://amrex-astro.github.io/Nyx/docs_html/NyxGettingStarted.html) for more information.

There are a variety of different simulation setups already implemented in 
```
axionyx_2.0/Exec/FDM/
```
Generally any new project needs a GNUMakefile specifying directories and compiler options. While the directories typically do not have to be changed, compiler options depend on the specific problem. If set equal to TRUE, new functionality is added to the executable, while setting it to FALSE excludes this functionality. The main compiler options are

| option | description |
|--|--|
| USE_MPI | include MPI parallelization |
| USE_OMP | include OpenMP parallelization |
| USE_CUDA | if run on GPUs |
| NO_MPI_CHECKING | MPI checking unfortunately doesn't seem to properly work yet, so we have to specify compiler flags manually in axionyx_2.0/subprojects/amrex/Tools/GNUMake/Make.local and set this flag to TRUE |
|DEBUG | A new project should always be run in debug mode first as it checks for pointers running out of arrays, initialises values to NAN if not defined and checks assertions all over the code. Once everything runs properly, always switch of debug module to significantly increase performance of the code. |
| USE_PARTICLES | TRUE if any kind of particle (N-body or else) is used.|
| USE_GRAV | if gravity should be used |
| NO_HYDRO | if hydro should not be used |
| USE_FDM | This initialises and AxionState with (AxDens,AxRe,AxIm,AxPhas) that is evolved by spectral or finite difference implementations of the Schroedinger equation. |
| USE_FDM_PARTICLES | This initialises the FDMPhaseParticleContainer that holds the N-body particles with phase information needed for the reconstruction of the wave function in the hybrid method. |
| USE_AXCOMPLEX | Initialises and AxionState with (AxRe,AxIm,AxvRe,AxvIm) representing the cosmic axion string network evolved by a spectral or finite difference implementation of the Klein-Gordon equation with QCD potential. |
| USE_AXREAL | After cosmic axion strings have collapsed only the complex phase needs to be further evolved. This is not yet fully implemented and should always be turned off. |
| USE_INF | This is required when inflaton N-body and hybrid simulations are performed as it changes the unit system (value of G, hbaroverm). |
| USE_INFGRAV | For computation of the gravitational wave spectrum in inflaton simulations. |

Additionally a project typically has a Make.package file adding a Prob.cpp to the compilation. Within Prob.cpp initial conditions can be specified analogous to previous projects. After defining the simulation setup AxioNyx is compiled as described in detail in the next section.

# Compilation

## 1. Compilation on a Mac computer 

### 1.1 gcc compiler
Make sure you have the most recent gcc compiler. By default, Apple uses clang which is a no-no. Install for example gcc-10, we recommend you do it with homebrew. You might also need to add 

```
    export OMPI_CXX=g++-10
    export OMPI_CC=gcc-10
    export OMPI_FC=gfortran-10
    export OMPI_F90=gfortran-10
```

to your ~/.bashrc file and source it. (source ~/.bashrc) Additionally, you might need 

```
    alias gcc='gcc-10'
    alias cc='gcc-10'
    alias g++='g++-10'
    alias c++='c++-10'
```

in ~/.bash_profile. After this, you should see

```
    gcc --version

    gcc-10 (Homebrew GCC 10.2.0_3) 10.2.0
    Copyright (C) 2020 Free Software Foundation, Inc.
    ...
```

### 1.2 FFTW3

AxioNyx depends on FFTW3, which can be downloaded
[here](http://www.fftw.org/download.html)

and installed locally, e.g. in $HOME/local using 

```
    ./configure --enable-mpi --enable-openmp --enable-shared --prefix=$HOME/local
    make
    make install
```

This ensures that both MPI and OpenMP are supported. Additionally, one can specify a gcc compiler at the end of the configuration command with e.g.

```
    CC=gcc-10
```

In AxioNyx GNUmakefile the FFTW3 directory location should then be specified and libraries linked with 

```
    FFTW_DIR ?= /Users/yourname/local
    LIBRARIES += -L$(FFTW_DIR)/lib -lfftw3_mpi -lfftw3_omp -lfftw3
    INCLUDE_LOCATIONS       += $(FFTW_DIR)/include
```

 You might need 

``` 
     NO_MPI_CHECKING=FALSE
```


## 2. Compilation on NeSI - Mahuika

Here, you will be using intel compilers, which you load by 

```
    module load intel/2018b 
```

In GNUmakefile, you then need to replace existing files to:  

```
    COMP    = intel
    LIBRARIES += -mkl    
 ```
 
 You might need 
 
 ```
     NO_MPI_CHECKING=FALSE
```

```INCLUDE_LOCATIONS``` does not need to be specified, you also don't need to install FFTW3. 
 
## 3. Compilation on NeSI - Maui

### 3.1 Compilers 
Compilers to load:

```
    module swap PrgEnv-cray PrgEnv-intel
    module load gcc/7.3.0
    module load craype-hugepages2M
```

### 3.2 In the GNUmakefile  

```
    COMP    = intel
    LIBRARIES += -mkl
```

IMPORTANT: Insert this at the end of GNUmakefile

```
    MKLPATH = /opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/include/fftw
    FFLAGS += -qopt-zmm-usage=high -I$(MKLPATH)
    F90FLAGS += -qopt-zmm-usage=high -I$(MKLPATH)
    CXXFLAGS += -qopt-zmm-usage=high -I$(MKLPATH)
    CFLAGS += -qopt-zmm-usage=high -I$(MKLPATH)
```

IMPORTANT no.2:

```
    NO_MPI_CHECKING=FALSE
```

### 3.3 AMReX

If you use AMReX form this git repo: 

```
    MatejaGosenca/amrex
```

you do not need to do anything, otherwise you need to change these two files: 

```
    ../../../amrex/Tools/GNUMake/sites/Make.nesi
```
and 
```
    ../../../amrex/Tools/GNUMake/Make.machines
```
as specified in ```MatejaGosenca/amrex```.

## 4. Compilation on HLRN

Currently we load the following modules
```
export FFTW_DIR="/sw/numerics/fftw3/impi/intel/3.3.8/skl/lib"
export LD_LIBRARY_PATH=$FFTW_DIR:$LD_LIBRARY_PATH

module load intel impi/2019.5 fftw3/impi/intel/3.3.8 hdf5-parallel/impi/intel/1.10.5 gcc/9.3.0
```
The GNUMakefile and make files in amrex are already configured for use on the HLRN. However, it is important to create a file called ```Make.local``` in ```/amrex/Tools/GNUMake``` with the following content:
```
CXX := mpicxx
CC := mpicc
FC := mpif90
F90 := mpif90

```
## 5. Compilation on the gwdg
Currently we load the following modules 

```
module load fftw/3.3.8 openjdk/11.0.2 hdf5/1.10.7 intel-mpi/2019.9.304 gcc/9.3.0

export FFTW_DIR='/opt/sw/rev/20.12/cascadelake/intel-2020.4/fftw-3.3.8-gc6ipq/lib'
export LD_LIBRARY_PATH=$FFTW_DIR:$LD_LIBRARY_PATH

```
The GNUMakefile and make files in amrex are already configured for use on the gwdg. However, it is important to create a file called ```Make.local``` in ```/amrex/Tools/GNUMake``` with the following content:
```
CXX := mpicxx
CC := mpicc
FC := mpif90
F90 := mpif90

``` 

# Running a simulation

Create a new directory containing your compiled executable, an inputs file defining your simulation parameters and a runscript to submit jobs on the cluster. Apart from input parameters from AMReX and Nyx well explained in the respective documentations, AxioNyx introduces new parameters 

## All configurations
|parameter| description | default |
|--|--|--|
| nyx.levelmethod | Which integration method to use on each level (finite difference = 0, Gauss Beam = 1, N-body = 2, speudo spectral = 3, classical wave approximation = 4| has to be provided |
| nyx.order | nth order spectral method 2 or 6 | 2 |
| nyx.neighbors | 1 or 2 neighbor cells for laplace stencil | 2 |
| nyx.nstep_spectrum | output power spectrum every nth step | -1 (no output) |
| nyx.interpolator | how to interpolate between levels (bilinear = 1, quartic = 2) | 1 |

## FDM
| parameter | description | default |
|--|--|--|
| nyx.m_tt | FDM mass in 10^-22 eV | 2.5 |
| nyx.sigma_fdm | Width of the Gaussian beam kernel in standard deviation per cell size | 1.0 |
| nyx.mixed_cosmology | whether to simulated mixed FDM/CDM cosmology | 0 |
| nyx.ratio_fdm | fraction of FDM/(FDM+CDM) in mixed cosmology simulations | 1.0 |
| nyx.beam_cfl | rescale Gauss beam time step | 0.2 |
| nyx.vonNeumann_dt | rescale FDM grid time step | 0.0 (no rescaling) |
| nyx.fdm_halo | refine pre-selected halo | false |
| nyx.fdm_halo_pos | initial position of that halo | has to be provided |
| nyx.fdm_halo_box | width of box tagged for refinement around that halo | has to be provided  |
| nyx.shiftvel | subtract velocity from all FDMphaseParticles (was needed for agora simulation) | false |

## Axion strings
| parameter | description | default |
|--|--|--|
| nyx.msa | axion mass per length scale | 1.0 |
| nyx.jaxions_ic | whether to construct initial conditions for string network simulations | 0 |
| nyx.prsstring | whether to use PRS strings | 1 |
| nyx.string_dt | rescale axion string time step | 1.0 |
| nyx.sigmalut | threshold for LUT correction in units 1/dx of root grid | 0.4 |
| nyx.string_massterm | whether to add mass term | 1 |
| nyx.string_time1 | reference time in mass term | 1.0 |
| nyx.string_n | exponent in mass term | 7.3 |
| nyx.test_posx/y/z | output data of cells with these indices in runlog file | zero length vector |

## Early Universe
| parameter | description | default |
|--|--|--|
| nyx.gw_interval | whether the gravitational wave spectrum is not computed at all time steps | 0 |
|nyx.gw_dt | time step between greavitational wave computation | 0 |
|nyx.gw_log| where to output the gravitational wave spectrum (make sure this directory exists) | "" |

## runscripts
For the HLRN this could look similar to this 
```
#!/bin/bash                                                                                                                                                                                                
#SBATCH -t 1:00:00                                                                                                                                                                                         
#SBATCH --nodes 1                                                                                                                                                                                          
#SBATCH -p standard96:test                                                                                                                                                                                                                                                                                                                                                                       

module load intel impi/2019.5 fftw3/impi/intel/3.3.8 hdf5-parallel/impi/intel/1.10.5 gcc/9.3.0

export FFTW_DIR="/sw/numerics/fftw3/impi/intel/3.3.8/skl/lib"
export LD_LIBRARY_PATH=$FFTW_DIR:$LD_LIBRARY_PATH
export SLURM_CPU_BIND=none  # important when using "mpirun" from Intel-MPI!                                                                                                                                                                                                                                                                                                                 

mpirun -np 64 ./Nyx3d.gnu.MPI.ex inputs
```

For more info on how to run on  the [HLRN](https://www.hlrn.de/doc/display/PUB/Usage+Guide) or [NeSI](https://support.nesi.org.nz/hc/en-gb/articles/360000684396-Submitting-your-first-job) computing clusters look at there respective documentation.

# Miscellaneous

## Simulations with FDM Particles

When running simulations with FDM Particles (e.g. in hybrid simulations) , the timestep is determined by taking the minimum between the phase criterion and the particle criterion (see ```estTimestepFDM``` and ```estTimestep``` in ```/Source/Particle/NyxParticles.cpp```). However, in our hybrid simulations where initially only the N-body particles are evolved, ```estTimestepFDM``` is not required. Because of this we added an ```if```-clause such that by default the phase timestep is not considered. Ideally, one should specify the level at which the reconstruction of the wave function is done such that above the last N-body level the phase time step is taken into account.  
