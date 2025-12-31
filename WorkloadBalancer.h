//
//  WorkloadBalancer.h
//  main
//
//

#ifndef WorkloadBalancer_h
#define WorkloadBalancer_h

#include <vector>
#include <string>

#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>

#include "MPIInfo.h"

template <typename Tria>
class WorkloadBalancer
{
private:
    const MPIInfo& __mpiInfo;
    
public:
    virtual ~WorkloadBalancer() = default;
    
    WorkloadBalancer(const MPIInfo& mpiInfo);
    
    void balance();
};


#endif /* WorkloadBalancer_h */
