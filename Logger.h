//
//  Logger.hpp
//  main
//
//

#ifndef Logger_hpp
#define Logger_hpp

#include <fstream>
#include <memory>

#include "MPIInfo.h"


class Logger
{
private:
    const MPIInfo&   __mpiInfo;
    
    const bool       __isOn;
    
    mutable std::unique_ptr<std::ofstream>             __ofstream;
public:
    virtual ~Logger();
    
    
    
    Logger()                = delete;
    Logger(const Logger&)   = delete;
    Logger(Logger&&)        = delete;
    
    
    Logger(const MPIInfo&       mpiInfo,
           const std::string&   dir,
           const std::string&   filename,
           const unsigned int   selectedRank = 0);
    
    
    // << operator
    template <typename T>
    friend Logger& operator<<(Logger& logger, const T& content);
  
    // << operator for manipulator, i.e. std::endl, std::flush
    friend Logger& operator<<(Logger& logger,
                              std::ostream& (*manip)(std::ostream&));
    
    
};





template <typename T>
inline Logger&
operator<<(Logger& logger, const T& content) 
{
    if(logger.__isOn)
        *(logger.__ofstream) << content;
        
    return logger;
}

inline Logger&
operator<<(Logger& logger,
           std::ostream& (*manip)(std::ostream&))
    {
        if(logger.__isOn)
            manip(*(logger.__ofstream));
            
        return logger;
    }



#endif /* Logger_hpp */
