//
//  TimerOutputWrapper.h
//  main
//
//

#ifndef TimerOutputWrapper_h
#define TimerOutputWrapper_h

#include <deal.II/base/timer.h>

#include <ostream>
#include <memory>

#include "MPIInfo.h"


class TimerOutputWrapper
{
private:
    
    using OutputFrequency   =  dealii::TimerOutput::OutputFrequency;
    using OutputType        =  dealii::TimerOutput::OutputType;
    using OutputData        =  dealii::TimerOutput::OutputData;
    
    const MPIInfo&                          __mpiInfo;
    
    std::unique_ptr<dealii::TimerOutput>    __timerPtr;
    
    
public:
    virtual ~TimerOutputWrapper() = default;
    
    explicit TimerOutputWrapper(std::ostream&  stream,
                                const MPIInfo& mpiInfo,
                                const OutputFrequency output_frequency,
                                const OutputType output_type);
    
    explicit TimerOutputWrapper(dealii::ConditionalOStream &stream,
                                const MPIInfo& mpiInfo,
                                const OutputFrequency output_frequency,
                                const OutputType output_type);
    
    
    
    dealii::TimerOutput& timer();
    
    
    
    void enter_subsection (const std::string &section_name);
    void leave_subsection (const std::string &section_name="");
    std::map<std::string, double> get_summary_data (const OutputData kind) const;
    void print_summary () const;
    void print_wall_time_statistics (const double print_quantile=0.) const;
    void disable_output ();
    void enable_output ();
    void reset ();
    
};

#endif /* TimerOutputWrapper_h */
