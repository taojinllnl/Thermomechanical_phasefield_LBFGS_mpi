//
//  CstSelector.cpp
//  main
//
//


#include "../../../include/Common/cst/CstSelectorBase.h"

using namespace ::dealii;
using namespace ::bcs;


template <typename Tria>
CstSelectorBase<Tria>::CstSelectorBase(const CstSelectorBase &selector)
  : __cstsInput(selector.__cstsInput)
  , __cstsOutput()
  , __pntCache({{INF, INF, INF}})
  , __valuesCache()
  , __cstPntRecordsFound()
  , __dofsFound()
{
  clear();
}

template <typename Tria>
CstSelectorBase<Tria>::CstSelectorBase(CstSelectorBase &&selector)
  : __cstsInput(selector.__cstsInput)
  , __cstsOutput()
  , __pntCache({{INF, INF, INF}})
  , __valuesCache()
  , __cstPntRecordsFound()
  , __dofsFound()
{
  clear();
}


template <typename Tria>
CstSelectorBase<Tria>::CstSelectorBase(const CstEntry<Tria> &cstInput)
  : __cstsInput(1, cstInput)
  , __cstsOutput()
  , __pntCache({{INF, INF, INF}})
  , __valuesCache()
  , __dofsCache()
  , __cstPntRecordsFound()
  , __dofsFound()
{
  clear();
}


template <typename Tria>
CstSelectorBase<Tria>::CstSelectorBase(
  const std::vector<CstEntry<Tria>> &cstInput)
  : __cstsInput(cstInput)
  , __cstsOutput()
  , __pntCache({{INF, INF, INF}})
  , __valuesCache()
  , __cstPntRecordsFound()
  , __dofsFound()
{
  clear();
}



template <typename Tria>
CstSelectorBase<Tria>::CstSelectorBase(
  const std::initializer_list<CstEntry<Tria>> &cstInput)
  : __cstsInput(cstInput)
  , __cstsOutput()
  , __pntCache({{INF, INF, INF}})
  , __valuesCache()
  , __cstPntRecordsFound()
  , __dofsFound()
{
  clear();
}


template <typename Tria>
bool
CstSelectorBase<Tria>::isDoFValid(const GlobalDoF dof)
{
  return dof != numbers::invalid_dof_index;
}



template <typename Tria>
bool
CstSelectorBase<Tria>::hasNoCachedDoF(const CstEntry<Tria> &entryRef) const
{
  return !isDoFValid(__dofsCache[entryRef.getDofAtPnt()]);
}


template <typename Tria>
void
CstSelectorBase<Tria>::newCstInput(const std::vector<CstEntry<Tria>> &cstInput)
{
  clear();
  __cstsInput = std::vector<CstEntry<Tria>>(cstInput);
}


template <typename Tria>
const std::vector<CstEntry<Tria>> &
CstSelectorBase<Tria>::cstInput() const
{
  return __cstsInput;
}


template <typename Tria>
const std::vector<CstEntryResult<Tria>> &
CstSelectorBase<Tria>::cstOutput() const
{
  return __cstsOutput;
}


template <typename Tria>
const std::vector<typename CstSelectorBase<Tria>::CstPointRecord> &
CstSelectorBase<Tria>::cstPointRecordsFound() const
{
  return __cstPntRecordsFound;
}



template <typename Tria>
void
CstSelectorBase<Tria>::clearCache()
{
  __pntCache[0] = INF;
  __pntCache[1] = INF;
  __pntCache[2] = INF;

  __valuesCache.clear();
  __dofsCache.clear();
  hasInit = false;
}

template <typename Tria>
void
CstSelectorBase<Tria>::clearFound()
{
  __dofsFound.clear();
  __cstPntRecordsFound.clear();

  hasInit = false;
}


template <typename Tria>
void
CstSelectorBase<Tria>::clear()
{
  __cstsOutput.clear();

  clearCache();
  clearFound();
}

template <typename Tria>
std::size_t
CstSelectorBase<Tria>::nCstPnts() const
{
  return __cstPntRecordsFound.size();
}

template <typename Tria>
std::size_t
CstSelectorBase<Tria>::nCstEntries() const
{
  return __cstsOutput.size();
}



template <typename Tria>
std::size_t
CstSelectorBase<Tria>::expectedNumberOfCstPoints() const
{
  return UNKNOW_SIZE;
}



template <typename Tria>
std::size_t
CstSelectorBase<Tria>::expectedNumberOfCstEntries() const
{
  return UNKNOW_SIZE;
}


template <typename Tria>
void
CstSelectorBase<Tria>::assignValues(const double,
                                    const double,
                                    const double,
                                    std::vector<double> &values)
{
  for (const CstEntry<Tria> &entry : __cstsInput)
    {
      const unsigned int dofAtPnt = entry.getDofAtPnt();

      AssertThrow(dofAtPnt < values.size(),
                  ExcMessage("values size is smaller than dofAtPnt."));
      values[dofAtPnt] = entry.getValue();
    }
}


template <typename Tria>
void
CstSelectorBase<Tria>::init(const std::size_t nDoFs)
{
  __pntCache[0] = INF;
  __pntCache[1] = INF;
  __pntCache[2] = INF;

  __valuesCache.assign(nDoFs, INF);
  __dofsCache.assign(nDoFs, numbers::invalid_dof_index);

  __dofsFound.clear();
  __cstPntRecordsFound.clear();

  __cstsOutput.clear();

  hasInit = true;
}


template <typename Tria>
std::vector<typename CstSelectorBase<Tria>::GlobalDoF> &
CstSelectorBase<Tria>::addDoFs2Cache(const std::size_t nDoFs)
{
  if (nDoFs != __dofsCache.size())
    {
      __dofsCache.assign(nDoFs, numbers::invalid_dof_index);
    }
  return __dofsCache;
}


template <typename Tria>
void
CstSelectorBase<Tria>::addDoF2Cache(const std::size_t     nDoFs,
                                    const GlobalDoF       dof,
                                    const CstEntry<Tria> &entryRef)
{
  const auto dofAtPnt = entryRef.getDofAtPnt();
  AssertThrow(dofAtPnt < nDoFs,
              ExcMessage(
                "The dofAtPnt is larger than nDoFs in addDoF2Cache()."));
  __dofsCache[dofAtPnt] = dof;
}



template <typename Tria>
void
CstSelectorBase<Tria>::createOutputByPoint(
  const ::dealii::Point<spacedim> &point,
  const std::size_t                nDoFs)
{
  AssertThrow(hasInit, ExcMessage("CstSelectorBase<Tria> is not initialized."));


  bool hasNewDoF = false;

  __pntCache[0] = point[0];

  if constexpr (spacedim >= 2)
    __pntCache[1] = point[1];
  else
    __pntCache[1] = 0.0;

  if constexpr (spacedim >= 3)
    __pntCache[2] = point[2];
  else
    __pntCache[2] = 0.0;

  // dofs have already been added into __dofsCache via addDoF2Cache() or
  // addDoFs2Cache()
  for (const GlobalDoF dof : __dofsCache)
    {
      // skip unset dofs
      if (!isDoFValid(dof))
        continue;

      // found new dof
      // skip the dofs which have been found
      if (__dofsFound.find(dof) == __dofsFound.end())
        {
          hasNewDoF = true;
          break;
        }
    }

  // obtain the values into __valuesCache by derived classes
  if (hasNewDoF)
    {
      __valuesCache.assign(nDoFs, INF);

      assignValues(__pntCache[0], __pntCache[1], __pntCache[2], __valuesCache);

      AssertThrow(
        __valuesCache.size() >= nDoFs,
        ExcMessage(
          "CstFunc::valuesFunc returned/resized values with size smaller than nDoFs."));
    }


  // store the entry indices
  std::vector<std::size_t> entryIndicesOnThisPoint;

  // loop over __cstsInput as template to generate __cstPntRecordsFound and
  // __cstsOutput
  for (const CstEntry<Tria> &inputEntry : __cstsInput)
    {
      const unsigned int dofAtPnt = inputEntry.getDofAtPnt();

      AssertThrow(dofAtPnt < __dofsCache.size(),
                  ExcMessage("dofAtPnt is out of range of __dofsCache."));

      const GlobalDoF dof = __dofsCache[dofAtPnt];

      // skip unset dofs
      if (!isDoFValid(dof))
        continue;

      // skip the dofs which have been found
      if (__dofsFound.find(dof) != __dofsFound.end())
        continue;

      AssertThrow(
        dofAtPnt < __valuesCache.size(),
        ExcMessage(
          "valuesFunc returned insufficient values for cstEntry.dofAtPnt."));

      AssertThrow(
        std::isfinite(__valuesCache[dofAtPnt]),
        ExcMessage(
          "Constraint value was not assigned or set an infinite value for dofAtPnt = " +
          std::to_string(dofAtPnt) + "."));

      const bool inserted = __dofsFound.insert(dof).second;

      AssertThrow(
        inserted,
        ExcMessage(
          "Internal error: DoF was expected to be new but insertion failed."));

      // create a new CstEntryResult by inputEntry and set its dof and value
      CstEntryResult<Tria> temp(inputEntry);
      temp.setGlobalDof(dof);
      temp.setValue(__valuesCache[dofAtPnt]);

      __cstsOutput.emplace_back(temp);
      entryIndicesOnThisPoint.emplace_back(__cstsOutput.size() - 1);
    }

  if (!entryIndicesOnThisPoint.empty())
    {
      CstPointRecord record;
      record.point        = __pntCache;
      record.entryIndices = std::move(entryIndicesOnThisPoint);

      __cstPntRecordsFound.emplace_back(std::move(record));
    }

  __pntCache[0] = INF;
  __pntCache[1] = INF;
  __pntCache[2] = INF;

  __valuesCache.assign(nDoFs, INF);
  __dofsCache.assign(nDoFs, numbers::invalid_dof_index);
}



template <typename Tria>
void
CstSelectorBase<Tria>::updateValues(const std::size_t nDoFs)
{
  AssertThrow(
    hasInit,
    ExcMessage(
      "CstSelectorBase<Tria> is not initialized. Call preparePntCst() before updateValues()."));

  AssertThrow(nDoFs > 0,
              ExcMessage("nDoFs must be larger than zero in updateValues()."));

  // In MPI mode, many ranks may not own any constrained DoF.
  // In that case, this selector has nothing to update on this rank.
  if (__cstPntRecordsFound.empty())
    {
      AssertThrow(
        __cstsOutput.empty(),
        ExcMessage(
          "Internal inconsistency: cstPointRecordsFound() is empty but cstOutput() is not empty."));
      return;
    }


  for (const CstPointRecord &record : __cstPntRecordsFound)
    {
      __valuesCache.assign(nDoFs, INF);

      assignValues(record.point[0],
                   record.point[1],
                   record.point[2],
                   __valuesCache);

      AssertThrow(
        __valuesCache.size() >= nDoFs,
        ExcMessage(
          "assignValues()/valuesFunc returned or resized values to a size smaller than nDoFs."));

      for (const std::size_t entryIndex : record.entryIndices)
        {
          AssertThrow(
            entryIndex < __cstsOutput.size(),
            ExcMessage("Invalid entry index in CstPointRecord::entryIndices."));

          CstEntryResult<Tria> &entry = __cstsOutput[entryIndex];

          AssertThrow(
            entry.hasValidDoF(),
            ExcMessage(
              "CstEntryResult has invalid global DoF during updateValues()."));

          const unsigned int dofAtPnt = entry.getDofAtPnt();

          AssertThrow(
            dofAtPnt < __valuesCache.size(),
            ExcMessage(
              "dofAtPnt is out of range of values in updateValues()."));

          AssertThrow(
            std::isfinite(__valuesCache[dofAtPnt]),
            ExcMessage(
              "Constraint value was not assigned or is not finite for dofAtPnt = " +
              std::to_string(dofAtPnt) + " in updateValues()."));

          entry.setValue(__valuesCache[dofAtPnt]);
        }
    }
  __valuesCache.assign(nDoFs, INF);
}


template class bcs::CstSelectorBase<RTria<1, 1>>;
;
template class bcs::CstSelectorBase<RTria<1, 2>>;
;
template class bcs::CstSelectorBase<RTria<1, 3>>;
;

template class bcs::CstSelectorBase<RTria<2, 2>>;
;
template class bcs::CstSelectorBase<RTria<2, 3>>;
;

template class bcs::CstSelectorBase<RTria<3, 3>>;
;


#if defined(DISTRIBUTED_TRIA) && DISTRIBUTED_TRIA
template class bcs::CstSelectorBase<DTria<1, 1>>;
;
template class bcs::CstSelectorBase<DTria<1, 2>>;
;
template class bcs::CstSelectorBase<DTria<1, 3>>;
;

template class bcs::CstSelectorBase<DTria<2, 2>>;
;
template class bcs::CstSelectorBase<DTria<2, 3>>;
;

template class bcs::CstSelectorBase<DTria<3, 3>>;
;
#endif
