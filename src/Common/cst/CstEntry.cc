//
//  CstEntry.cpp
//  main
//
//

#include "../../../include/Common/cst/CstEntry.h"


using namespace ::dealii;
using namespace ::bcs;



template <typename Tria>
CstEntry<Tria>::CstEntry(const LocalDoF dofAtPnt, const double value)
  : dofAtPnt(dofAtPnt)
  , value(value)
{}


template <typename Tria>
CstEntry<Tria>::CstEntry(const CstEntry &entry)
  : dofAtPnt(entry.dofAtPnt)
  , value(entry.value)
{}


template <typename Tria>
CstEntry<Tria>::CstEntry(CstEntry &&entry)
  : dofAtPnt(entry.dofAtPnt)
  , value(entry.value)
{}



template <typename Tria>
double
CstEntry<Tria>::getValue() const
{
  return value;
}

template <typename Tria>
typename CstEntry<Tria>::LocalDoF
CstEntry<Tria>::getDofAtPnt() const
{
  return dofAtPnt;
}


template <typename Tria>
void
CstEntry<Tria>::setValue(const double value_)
{
  value = value_;
}



template struct bcs::CstEntry<RTria<1, 1>>;
template struct bcs::CstEntry<RTria<1, 2>>;
template struct bcs::CstEntry<RTria<1, 3>>;

template struct bcs::CstEntry<RTria<2, 2>>;
template struct bcs::CstEntry<RTria<2, 3>>;

template struct bcs::CstEntry<RTria<3, 3>>;



#if defined(DISTRIBUTED_TRIA) && DISTRIBUTED_TRIA
template struct bcs::CstEntry<DTria<1, 1>>;
template struct bcs::CstEntry<DTria<1, 2>>;
template struct bcs::CstEntry<DTria<1, 3>>;

template struct bcs::CstEntry<DTria<2, 2>>;
template struct bcs::CstEntry<DTria<2, 3>>;

template struct bcs::CstEntry<DTria<3, 3>>;
#endif



template <typename Tria>
CstEntryResult<Tria>::CstEntryResult(const LocalDoF dofAtPnt,
                                     const double   value)
  : CstEntry<Tria>(dofAtPnt, value)
  , dof(::dealii::numbers::invalid_dof_index)
{}

template <typename Tria>
CstEntryResult<Tria>::CstEntryResult(const CstEntryResult &entry)
  : CstEntry<Tria>(entry)
  , dof(entry.dof)
{}


template <typename Tria>
CstEntryResult<Tria>::CstEntryResult(CstEntryResult &&entry)
  : CstEntry<Tria>(std::move(entry))
  , dof(entry.dof)
{
  entry.resetDoF();
}


template <typename Tria>
CstEntryResult<Tria>::CstEntryResult(const CstEntry<Tria> &entry)
  : CstEntry<Tria>(entry)
  , dof(::dealii::numbers::invalid_dof_index)
{}

template <typename Tria>
CstEntryResult<Tria>::CstEntryResult(CstEntry<Tria> &&entry)
  : CstEntry<Tria>(std::move(entry))
  , dof(::dealii::numbers::invalid_dof_index)
{}



template <typename Tria>
types::global_dof_index
CstEntryResult<Tria>::getGlobalDoF() const
{
  AssertThrow(hasValidDoF(),
              ExcMessage("No valid DoF assigned into CstEntry<Tria>"));
  return dof;
}



template <typename Tria>
void
CstEntryResult<Tria>::setGlobalDof(
  const ::dealii::types::global_dof_index new_dof)
{
  AssertThrow(new_dof != ::dealii::numbers::invalid_dof_index,
              ExcMessage("CstView::setDof() got invalid_dof_index."));
  dof = new_dof;
}



template <typename Tria>
bool
CstEntryResult<Tria>::hasValidDoF() const
{
  return dof != ::dealii::numbers::invalid_dof_index;
}



template <typename Tria>
void
CstEntryResult<Tria>::resetDoF()
{
  dof = ::dealii::numbers::invalid_dof_index;
}



template struct bcs::CstEntryResult<RTria<1, 1>>;
template struct bcs::CstEntryResult<RTria<1, 2>>;
template struct bcs::CstEntryResult<RTria<1, 3>>;

template struct bcs::CstEntryResult<RTria<2, 2>>;
template struct bcs::CstEntryResult<RTria<2, 3>>;

template struct bcs::CstEntryResult<RTria<3, 3>>;



#if defined(DISTRIBUTED_TRIA) && DISTRIBUTED_TRIA
template struct bcs::CstEntryResult<DTria<1, 1>>;
template struct bcs::CstEntryResult<DTria<1, 2>>;
template struct bcs::CstEntryResult<DTria<1, 3>>;

template struct bcs::CstEntryResult<DTria<2, 2>>;
template struct bcs::CstEntryResult<DTria<2, 3>>;

template struct bcs::CstEntryResult<DTria<3, 3>>;
#endif
