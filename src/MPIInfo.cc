//
//  MPIInfo.cpp
//  main
//
//

#include "../include/MPIInfo.h"

using namespace dealii;
using MPIInit = Utilities::MPI::MPI_InitFinalize;

MPIInfo::MPIInfo(const bool mpiSupport,
                 int argc, char* argv[])
: __MPISupport(mpiSupport)
, __mpiInitPtr(__MPISupport 
               ? std::make_unique<MPIInit>(argc,
                                           argv,
                                           /*max_num_threads=*/1)
               : nullptr)
, __mpiCommPtr(__MPISupport ? std::make_unique<MPI_Comm>(MPI_COMM_WORLD): nullptr)
, __rank(__MPISupport ? Utilities::MPI::this_mpi_process(*__mpiCommPtr) : 0)
, __nRanks(__MPISupport ? Utilities::MPI::n_mpi_processes(*__mpiCommPtr): 1)
{}



MPI_Comm* MPIInfo::mpiCommPtr() noexcept
{
    return __MPISupport ? __mpiCommPtr.get() : nullptr;
}


const MPI_Comm* MPIInfo::mpiCommPtr() const noexcept
{
    return __MPISupport ? __mpiCommPtr.get() : nullptr;
}


bool MPIInfo::isMPI() const
{
    return __MPISupport;
}

unsigned int MPIInfo::rank() const
{
    return __rank;
}

unsigned int MPIInfo::nRanks() const
{
    return __nRanks;
}

bool MPIInfo::isCurrentRank(const unsigned int rank) const
{
    return rank == __rank;
}


void MPIInfo::summary(std::ostream& stream)
{
    if (__MPISupport) {
        stream << "MPI mode" << std::endl
        << "\tnumber of ranks: " << __nRanks
        << "\tcurrent rank: " << __rank << std::endl;
        
    } else {
        stream << "Non-MPI mode" << std::endl;
    }
}
