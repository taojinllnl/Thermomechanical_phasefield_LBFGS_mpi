//
//  VersionAdapter.h
//  main
//
//

#ifndef VersionAdapter_h
#define VersionAdapter_h

#include <deal.II/lac/affine_constraints.h>


#include "BlockDesc.h"

namespace common
{

class VersionAdapter
{
public:
    static void cstReinit(dealii::AffineConstraints<double>& constraints,
                          const dealii::IndexSet& locally_owned_dofs,
                          const dealii::IndexSet& locally_relevant_dofs);
};


}

#endif /* VersionAdapter_h */
