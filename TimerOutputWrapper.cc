//
//  TimerOutputWrapper.cpp
//  main
//
//

#include "TimerOutputWrapper.h"

using namespace dealii;
using OutputFrequency   =  dealii::TimerOutput::OutputFrequency;
using OutputType        =  dealii::TimerOutput::OutputType;
using OutputData        =  dealii::TimerOutput::OutputData;

TimerOutputWrapper
::TimerOutputWrapper(std::ostream&  stream,
                     const MPIInfo& mpiInfo,
                     const OutputFrequency output_frequency,
                     const OutputType output_type)
: __mpiInfo(mpiInfo)
{
    if (__mpiInfo.isMPI()) {
        __timerPtr = std::make_unique<TimerOutput>(*mpiInfo.mpiComm(),
                                                   stream,
                                                   output_frequency,
                                                   output_type);
    } else {
        __timerPtr = std::make_unique<TimerOutput>(stream,
                                                   output_frequency,
                                                   output_type);
    }
    
}

TimerOutputWrapper
::TimerOutputWrapper(ConditionalOStream &stream,
                     const MPIInfo& mpiInfo,
                     const OutputFrequency output_frequency,
                     const OutputType output_type)
: __mpiInfo(mpiInfo)
{
    if (__mpiInfo.isMPI()) {
        __timerPtr = std::make_unique<TimerOutput>(*mpiInfo.mpiComm(),
                                                   stream,
                                                   output_frequency,
                                                   output_type);
    } else {
        __timerPtr = std::make_unique<TimerOutput>(stream,
                                                   output_frequency,
                                                   output_type);
    }
}



TimerOutput&
TimerOutputWrapper
::timer()
{
    return *__timerPtr;
}




void     
TimerOutputWrapper
::enter_subsection (const std::string& section_name)
{
    __timerPtr->enter_subsection(section_name);
}


void
TimerOutputWrapper
::leave_subsection (const std::string& section_name)
{
    __timerPtr->leave_subsection(section_name);
}

std::map< std::string, double >     
TimerOutputWrapper
::get_summary_data (const OutputData kind) const
{
    return __timerPtr->get_summary_data(kind);
}

void     
TimerOutputWrapper
::print_summary () const
{
    __timerPtr->print_summary();
}

void     
TimerOutputWrapper
::print_wall_time_statistics (const double print_quantile) const
{
    __timerPtr->print_wall_time_statistics(*__mpiInfo.mpiComm(),
                                           print_quantile);
}

void     
TimerOutputWrapper
::disable_output ()
{
    __timerPtr->disable_output();
}

void     
TimerOutputWrapper
::enable_output ()
{
    __timerPtr->enable_output();
}

void     
TimerOutputWrapper
::reset ()
{
    __timerPtr->reset();
}
