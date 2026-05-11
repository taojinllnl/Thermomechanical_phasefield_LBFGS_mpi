//
//  CstHelper.cpp
//  main
//
//

#include "../../../include/Common/cst/CstHelper.h"


using namespace ::dealii;
using namespace ::bcs;



ConstraintEntry::ConstraintEntry(const unsigned int dofAtPnt,
                                 const double       value)
  : value(value)
  , dofAtPnt(dofAtPnt)
  , dof(numbers::invalid_dof_index)
{}


ConstraintEntry::ConstraintEntry(const ConstraintEntry &view)
  : value(view.value)
  , dofAtPnt(view.dofAtPnt)
  , dof(view.dof)
{}



::dealii::types::global_dof_index
ConstraintEntry::getGlobalDoF() const
{
  return dof;
}

double
ConstraintEntry::getValue() const
{
  return value;
}

bool
ConstraintEntry::hasValidDoF() const
{
  return dof != ::dealii::numbers::invalid_dof_index;
}



void
ConstraintEntry::setDof(const ::dealii::types::global_dof_index new_dof)
{
  AssertThrow(new_dof != ::dealii::numbers::invalid_dof_index,
              ExcMessage("CstView::setDof() got invalid_dof_index."));
  dof = new_dof;
}



void
ConstraintEntry::reset()
{
  dof = ::dealii::numbers::invalid_dof_index;
}



ConstrainedPnt ::ConstrainedPnt(const std::array<double, 3> &pntCoorinates,
                                const std::vector<ConstraintEntry> &csts,
                                const double                        tol)
  : pntCoorinates(std::make_shared<std::array<double, 3>>(pntCoorinates))
  , csts(csts)
  , nCsts(csts.size())
  , tol(tol)
{}


ConstrainedPnt ::ConstrainedPnt(const ConstrainedPnt &cstPnt)
  : pntCoorinates(cstPnt.pntCoorinates)
  , csts(cstPnt.csts)
  , nCsts(cstPnt.nCsts)
  , tol(cstPnt.tol)
{}

const std::vector<ConstraintEntry> &
ConstrainedPnt ::cstEntrys() const
{
  return csts;
}



ConstrainedFunc ::ConstrainedFunc(const SelFunc &selectorFunc,
                                  const std::vector<ConstraintEntry> &csts)
  : nCsts(0)
  , nPnts(0)
  , selectorFunc(std::make_shared<SelFunc>(selectorFunc))
  , valuesFunc(nullptr)
  , csts(csts)
  , dofsFound()
  , cstsFound()
  , pointsFound()
{}


ConstrainedFunc ::ConstrainedFunc(const SelFunc &selectorFunc,
                                  const std::vector<ConstraintEntry> &csts,
                                  const ValuesAtPntFunc &valuesFunc)
  : nCsts(0)
  , nPnts(0)
  , selectorFunc(std::make_shared<SelFunc>(selectorFunc))
  , valuesFunc(std::make_shared<ValuesAtPntFunc>(valuesFunc))
  , csts(csts)
  , dofsFound()
  , cstsFound()
  , pointsFound()
{}

ConstrainedFunc ::ConstrainedFunc(const ConstrainedFunc &cstFunc)
  : nCsts(0)
  , nPnts(0)
  , selectorFunc(cstFunc.selectorFunc)
  , valuesFunc(cstFunc.valuesFunc)
  , csts(cstFunc.csts)
  , dofsFound()
  , cstsFound()
  , pointsFound()
{}


const std::vector<ConstraintEntry> &
ConstrainedFunc ::cstEntrys() const
{
  return cstsFound;
}



void
CstHelper ::cstReinit(AffineConstraints<double> &constraints,
                      const IndexSet            &locally_owned_dofs,
                      const IndexSet            &locally_relevant_dofs)
{
#if DEAL_II_VERSION_GTE(9, 6, 0)
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
#else
  constraints.reinit(locally_relevant_dofs);
#endif
}



void
CstHelper ::addPntCst(AffineConstraints<double>              &cst,
                      const ::dealii::types::global_dof_index dof,
                      const double                            value)
{
  if (!cst.is_constrained(dof))
    {
      cst.add_line(dof);
      cst.set_inhomogeneity(dof, value);
    }
}
