//
//  TimerOutputWrapper.cpp
//  main
//
//

#include "../include/TimerOutputWrapper.h"

using namespace dealii;
using OutputFrequency   =  dealii::TimerOutput::OutputFrequency;
using OutputType        =  dealii::TimerOutput::OutputType;
using OutputData        =  dealii::TimerOutput::OutputData;

template <typename LATraits>
TimerOutputWrapper<LATraits>
::TimerOutputWrapper(std::ostream&  stream,
                     const MPIInfo& mpiInfo,
                     const OutputFrequency output_frequency,
                     const OutputType output_type)
: __mpiInfo(mpiInfo)
{
    if (__mpiInfo.isMPI()) {
        __timerPtr = std::make_unique<TimerOutput>(*mpiInfo.mpiCommPtr(),
                                                   stream,
                                                   output_frequency,
                                                   output_type);
    } else {
        __timerPtr = std::make_unique<TimerOutput>(stream,
                                                   output_frequency,
                                                   output_type);
    }
    
}

template <typename LATraits>
TimerOutputWrapper<LATraits>
::TimerOutputWrapper(ConditionalOStream &stream,
                     const MPIInfo& mpiInfo,
                     const OutputFrequency output_frequency,
                     const OutputType output_type)
: __mpiInfo(mpiInfo)
{
    if (__mpiInfo.isMPI()) {
        __timerPtr = std::make_unique<TimerOutput>(*mpiInfo.mpiCommPtr(),
                                                   stream,
                                                   output_frequency,
                                                   output_type);
    } else {
        __timerPtr = std::make_unique<TimerOutput>(stream,
                                                   output_frequency,
                                                   output_type);
    }
}


template <typename LATraits>
TimerOutput&
TimerOutputWrapper<LATraits>
::timer()
{
    return *__timerPtr;
}



template <typename LATraits>
void
TimerOutputWrapper<LATraits>
::enter_subsection (const std::string& section_name)
{
    if constexpr (std::is_same_v<la::Traits<la::TagSerial>, LATraits>) {
        __timerPtr->enter_subsection(section_name);
    } else {
        __scopeMap[section_name] = std::make_unique<Scope>(timer(), section_name);
    }
}

template <typename LATraits>
void
TimerOutputWrapper<LATraits>
::leave_subsection (const std::string& section_name)
{
    if constexpr (std::is_same_v<la::Traits<la::TagSerial>, LATraits>) {
        __timerPtr->leave_subsection(section_name);
    } else {
        auto it = __scopeMap.find(section_name);
        if (it != __scopeMap.end())
            __scopeMap.erase(it);
    }
}

template <typename LATraits>
std::map< std::string, double >
TimerOutputWrapper<LATraits>
::get_summary_data (const OutputData kind) const
{
    return __timerPtr->get_summary_data(kind);
}

template <typename LATraits>
void
TimerOutputWrapper<LATraits>
::print_summary () const
{
    __timerPtr->print_summary();
}

template <typename LATraits>
void
TimerOutputWrapper<LATraits>
::print_wall_time_statistics (const double print_quantile) const
{
    __timerPtr->print_wall_time_statistics(*__mpiInfo.mpiCommPtr(),
                                           print_quantile);
}

template <typename LATraits>
void
TimerOutputWrapper<LATraits>
::disable_output ()
{
    __timerPtr->disable_output();
}

template <typename LATraits>
void
TimerOutputWrapper<LATraits>
::enable_output ()
{
    __timerPtr->enable_output();
}

template <typename LATraits>
void
TimerOutputWrapper<LATraits>
::reset ()
{
    __timerPtr->reset();
}




template class TimerOutputWrapper<la::Traits<la::TagSerial>>;
template class TimerOutputWrapper<la::Traits<la::TagPETSc>>;
template class TimerOutputWrapper<la::Traits<la::TagTrilinos>>;
