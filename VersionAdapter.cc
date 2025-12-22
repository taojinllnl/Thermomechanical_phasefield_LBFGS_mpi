//
//  VersionAdapter.cpp
//  main
//
//

#include "VersionAdapter.h"


using namespace dealii;


void VersionAdapter
::cstReinit(AffineConstraints<double>& constraints,
            const IndexSet& locally_owned_dofs,
            const IndexSet& locally_relevant_dofs)
{
#  if DEAL_II_VERSION_GTE(9, 6, 0)
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
#  else
    constraints.reinit(locally_relevant_dofs);
#  endif
}

