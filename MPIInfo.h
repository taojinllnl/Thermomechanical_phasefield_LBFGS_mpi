//
//  MPIInfo.h
//  main
//
//

#ifndef MPIInfo_h
#define MPIInfo_h

#include <deal.II/base/mpi.h>

#include <memory>
#include <ostream>

class MPIInfo
{
private:
    const bool                                                __MPISupport;

    std::unique_ptr<dealii::Utilities::MPI::MPI_InitFinalize> __mpiInitPtr;
    std::unique_ptr<MPI_Comm>                                 __mpiCommPtr;
    
    const unsigned int                                        __rank;
    const unsigned int                                        __nRanks;
    
public:
    virtual ~MPIInfo() = default;
    
    MPIInfo(const bool mpiSupport,
            int argc, char* argv[]);
    
    
    MPI_Comm* mpiCommPtr() noexcept;
    
    const MPI_Comm* mpiCommPtr() const noexcept;
    
    bool isMPI() const;
    unsigned int rank() const;
    unsigned int nRanks() const;
    
    
    void summary(std::ostream& stream);
    
    
};


#endif /* MPIInfo_h */
