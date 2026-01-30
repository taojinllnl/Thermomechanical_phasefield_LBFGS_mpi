//
//  Logger.cpp
//  main
//
//

#include "Logger.h"



Logger::Logger(const MPIInfo&       mpiInfo,
               const std::string&   dir,
               const std::string&   filename,
               const unsigned int   selectedRank)
: __mpiInfo(mpiInfo)
, __isOn(mpiInfo.rank() == selectedRank)
{
    if(__isOn) {
        std::string streamPath;
        if(__mpiInfo.isMPI())
            
            streamPath = dir + "mpi_" + std::to_string(__mpiInfo.rank()) + "_" + std::to_string(__mpiInfo.nRanks()) + "_" + filename;
        else
            streamPath = dir + "serial_"+  filename;
        
        __ofstream = std::make_unique<std::ofstream>(streamPath);
    }
}



Logger::~Logger()
{
    if (__ofstream) {
        __ofstream->close();
    }
    
}
