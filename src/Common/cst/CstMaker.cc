//
//  CstMaker.cpp
//  main
//
//

#include "../../../include/Common/cst/CstMaker.h"


using namespace ::dealii;
using namespace ::bcs;


template <typename Tria>
void
CstMaker<Tria>::cstReinit(AffineConstraints<double> &constraints,
                          const IndexSet            &locally_owned_dofs,
                          const IndexSet            &locally_relevant_dofs)
{
#if DEAL_II_VERSION_GTE(9, 6, 0)
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
#else
  constraints.reinit(locally_relevant_dofs);
#endif
}


template <typename Tria>
void
CstMaker<Tria>::extractDoFsAtVetex(
  std::vector<::dealii::types::global_dof_index> &dofs,
  const RVertexIter                              &vertex,
  const DoFHandler                               &dof_handler,
  const std::size_t                               nDoFsPerVertex)
{
  using namespace ::dealii;
  if (dofs.size() != nDoFsPerVertex)
    {
      dofs.resize(nDoFsPerVertex);
    }
  DoFAccessor<0, dim, spacedim, false> vertex_dofs(
    &(dof_handler.get_triangulation()),
    vertex->level(),
    vertex->index(),
    &dof_handler);
  const unsigned int n_dofs = dof_handler.get_fe().dofs_per_vertex;

  for (unsigned int i = 0; i < n_dofs; ++i)
    {
      dofs[i] = vertex_dofs.vertex_dof_index(0, i);
    }
}



template <typename Tria>
CstMaker<Tria>::CstMaker(const common::MPIInfo &mpiInfo)
  : __mpiInfo(mpiInfo)
{}


template <typename Tria>
void
CstMaker<Tria>::eliminateSelectors()
{
  __cstSelectors.clear();
  isReady           = false;
  hasBoundSelectors = false;
  hasFoundCsts      = false;
}

template <typename Tria>
CstMaker<Tria> &
CstMaker<Tria>::addCstSelector(const CstSelectorBase<Tria> &selector)
{
  AssertThrow(
    !isReady,
    ExcMessage(
      "Cannot add CstSelector after preparePntCst(). Clear/rebuild CstMaker first."));
  __cstSelectors.emplace_back(selector.clone());
  return *this;
}

template <typename Tria>
bool
CstMaker<Tria>::isAddedSelectors() const
{
  return __cstSelectors.size() > 0;
}



template <typename Tria>
void
CstMaker<Tria>::updatePntCstValues(const DoFHandler &dofHandler,
                                   const bool        verifyAllFound,
                                   const bool        verifyNoRepeatedEntry)
{
  AssertThrow(isReady,
              ExcMessage(
                "preparePntCst() must be called before updatePntCstValues()."));

  const unsigned int nDoFs = dofHandler.get_fe().dofs_per_vertex;

  AssertThrow(nDoFs > 0,
              ExcMessage(
                "dofs_per_vertex should not be zero in updatePntCstValues()."));

  if (verifyAllFound)
    __verifyAllFound();

  if (verifyNoRepeatedEntry)
    __verifyNoRepeat();

  for (auto &selectorPtr : __cstSelectors)
    {
      AssertThrow(selectorPtr,
                  ExcMessage(
                    "nullptr selector in CstMaker::updatePntCstValues()."));

      selectorPtr->updateValues(nDoFs);
    }
}


template <typename Tria>
void
CstMaker<Tria>::applyPntCsts(Constraints &cst,
                             const bool   verifyAllFound,
                             const bool   verifyNoRepeatedEntry)
{
  AssertThrow(isReady,
              ExcMessage(
                "The csts on points have not been found before applying."));

  if (verifyAllFound)
    __verifyAllFound();
  if (verifyNoRepeatedEntry)
    __verifyNoRepeat();

  // in MPI mode: only the rank with constrained points will apply BCs.
  for (unsigned int i = 0; i < __cstSelectors.size(); ++i)
    {
      AssertThrow(__cstSelectors[i], ExcMessage("nullptr"));
      const CstSelectorBase<Tria> &selector = *__cstSelectors[i];

      const std::size_t nCsts = selector.nCstEntries();

      if (nCsts > 0)
        {
          for (unsigned int j = 0; j < nCsts; ++j)
            {
              const CstEntryResult<Tria> &cstEntry = selector.cstOutput()[j];

              if (!cstEntry.hasValidDoF())
                continue;

              const GlobalDoF dof = cstEntry.getGlobalDoF();

              if (!cst.is_constrained(dof))
                {
                  cst.add_line(dof);
                }
              cst.set_inhomogeneity(dof, cstEntry.getValue());
            }
        }
    } // loop over constrainted points
}



template <typename Tria>
void
CstMaker<Tria>::makeCstIfPrepareNeeded(bool             &dofChangedflag,
                                       Constraints      &cst,
                                       const Tria       &tria,
                                       const DoFHandler &dofHandler,
                                       const bool        updateValues,
                                       const bool        verifyAllFound,
                                       const bool        verifyNoRepeatedEntry)
{
  if (dofChangedflag)
    {
      preparePntCst(tria, dofHandler);
      dofChangedflag = false;
    }
  if (updateValues)
    {
      updatePntCstValues(dofHandler);
    }
  applyPntCsts(cst, verifyAllFound, verifyNoRepeatedEntry);
}


template <typename Tria>
void
CstMaker<Tria>::__verifyInput(const unsigned int nDoFs)
{
  using namespace ::dealii;

  AssertThrow(nDoFs > 0, ExcMessage("dofs_per_vertex should not be zero."));

  AssertThrow(__cstSelectors.size() > 0,
              ExcMessage("No CstSelectorBase<Tria> added in CstMaker<Tria>."));

  for (unsigned int ith = 0; ith < __cstSelectors.size(); ++ith)
    {
      AssertThrow(
        __cstSelectors[ith],
        ExcMessage(
          "[internal error] CstSelectorBase<Tria> is nullptr in CstMaker<Tria>."));

      const auto &inputEntries = __cstSelectors[ith]->cstInput();

      AssertThrow(inputEntries.size() > 0,
                  ExcMessage(std::to_string(ith) +
                             "-th constraint selector has empty cstInput()."));

      std::set<unsigned int> used_local_dof_ids;

      for (unsigned int k = 0; k < inputEntries.size(); ++k)
        {
          const CstEntry<Tria> &inputEntry = inputEntries[k];

          const unsigned int dofAtPnt = inputEntry.getDofAtPnt();

          // check dofAtPnt is larger than nDoFs
          AssertThrow(
            dofAtPnt < nDoFs,
            ExcMessage(
              std::to_string(ith) + "-th selector, " + std::to_string(k) +
              "-th CstEntry has dofAtPnt = " + std::to_string(dofAtPnt) +
              ", but dofs_per_vertex = " + std::to_string(nDoFs) + "."));

          // check repeated local dof
          AssertThrow(
            used_local_dof_ids.insert(dofAtPnt).second,
            ExcMessage(
              std::to_string(ith) +
              "-th constraint selector has repeated local dofAtPnt = " +
              std::to_string(dofAtPnt) +
              " in cstInput(). This is ambiguous inside one selector."));
        }
    }
}


template <typename Tria>
void
CstMaker<Tria>::__verifyAllFound()
{
  using namespace ::dealii;

  std::size_t global_total_entries = 0;

  for (unsigned int ith = 0; ith < __cstSelectors.size(); ++ith)
    {
      AssertThrow(
        __cstSelectors[ith],
        ExcMessage(
          "[internal error] nullptr selector in CstMaker::__verifyAllFound()."));

      const CstSelectorBase<Tria> &selector = *__cstSelectors[ith];

      const std::size_t local_n_pnts    = selector.nCstPnts();
      const std::size_t local_n_entries = selector.nCstEntries();

      std::size_t global_n_pnts    = local_n_pnts;
      std::size_t global_n_entries = local_n_entries;

      if constexpr (is_distributed)
        {
          global_n_pnts =
            Utilities::MPI::sum(local_n_pnts, *__mpiInfo.mpiCommPtr());

          global_n_entries =
            Utilities::MPI::sum(local_n_entries, *__mpiInfo.mpiCommPtr());
        }

      global_total_entries += global_n_entries;

      if (selector.expectedNumberOfCstPoints() !=
          CstSelectorBase<Tria>::UNKNOW_SIZE)
        {
          const std::size_t expected_n_pnts =
            selector.expectedNumberOfCstPoints();

          AssertThrow(
            global_n_pnts == expected_n_pnts,
            ExcMessage(
              std::to_string(ith) +
              "-th constraint selector did not find the expected number of constrained points. " +
              "Expected points = " + std::to_string(expected_n_pnts) +
              ", found points = " + std::to_string(global_n_pnts) + "."));
        }

      if (selector.expectedNumberOfCstEntries() !=
          CstSelectorBase<Tria>::UNKNOW_SIZE)
        {
          const std::size_t expected_n_entries =
            selector.expectedNumberOfCstEntries();

          AssertThrow(
            global_n_entries == expected_n_entries,
            ExcMessage(
              std::to_string(ith) +
              "-th constraint selector did not find the expected number of constrained entries. " +
              "Expected entries = " + std::to_string(expected_n_entries) +
              ", found entries = " + std::to_string(global_n_entries) + "."));
        }
      else
        {
          AssertThrow(
            global_n_entries > 0,
            ExcMessage(
              std::to_string(ith) +
              "-th constraint selector did not find any valid constrained DoF."));
        }
    }

  hasFoundCsts = (global_total_entries > 0);

  AssertThrow(hasFoundCsts,
              ExcMessage("No valid constrained DoF was found by CstMaker."));
}

template <typename Tria>
void
CstMaker<Tria>::__verifyNoRepeat()
{
  using namespace ::dealii;
  using global_dof_index = ::dealii::types::global_dof_index;

  std::set<global_dof_index> visited_dofs;

  bool has_local_repetition = false;

  global_dof_index repeated_dof = numbers::invalid_dof_index;

  unsigned int repeated_selector_id = 0;
  unsigned int repeated_entry_id    = 0;

  for (unsigned int ith = 0; ith < __cstSelectors.size(); ++ith)
    {
      AssertThrow(
        __cstSelectors[ith],
        ExcMessage(
          "[internal error] nullptr selector in CstMaker::__verifyNoRepeat()."));

      const CstSelectorBase<Tria> &selector = *__cstSelectors[ith];

      const auto &entries = selector.cstOutput();

      for (unsigned int k = 0; k < entries.size(); ++k)
        {
          const CstEntryResult<Tria> &cstEntry = entries[k];

          if (!cstEntry.hasValidDoF())
            continue;

          const global_dof_index dof = cstEntry.getGlobalDoF();

          if (!visited_dofs.insert(dof).second)
            {
              has_local_repetition = true;
              repeated_dof         = dof;
              repeated_selector_id = ith;
              repeated_entry_id    = k;
              break;
            }
        }

      if (has_local_repetition)
        break;
    }

  const unsigned int local_repetition = has_local_repetition ? 1u : 0u;

  unsigned int global_repetition = local_repetition;

  if constexpr (is_distributed)
    {
      global_repetition =
        Utilities::MPI::sum(local_repetition, *__mpiInfo.mpiCommPtr());
    }

  AssertThrow(global_repetition == 0,
              ExcMessage(
                "Repeated constrained DoF detected in CstMaker. " +
                (has_local_repetition ?
                   ("Repeated global DoF = " + std::to_string(repeated_dof) +
                    ", selector id = " + std::to_string(repeated_selector_id) +
                    ", entry id = " + std::to_string(repeated_entry_id) + ".") :
                   std::string(
                     "The repeated DoF was detected on another MPI rank."))));
}



template <typename Tria>
std::string
CstMaker<Tria>::cstListString(const std::string &name) const
{
  using namespace ::dealii;

  std::ostringstream local;

  local << std::scientific << std::setprecision(8);

  const unsigned int rank   = __mpiInfo.rank();
  const unsigned int nRanks = __mpiInfo.nRanks() - 1;

  local << "Rank " << rank << " / " << nRanks << "\n";
  local << "  Number of selectors: " << __cstSelectors.size() << "\n";

  for (unsigned int ithSelector = 0; ithSelector < __cstSelectors.size();
       ++ithSelector)
    {
      const auto &selectorPtr = __cstSelectors[ithSelector];

      if (!selectorPtr)
        {
          local << "  Selector " << ithSelector << ": nullptr\n";
          continue;
        }

      const CstSelectorBase<Tria> &selector = *selectorPtr;

      local << "  Selector " << ithSelector << ":\n";
      local << "    input entries  = " << selector.cstInput().size() << "\n";
      local << "    found points   = " << selector.nCstPnts() << "\n";
      local << "    output entries = " << selector.nCstEntries() << "\n";

      const auto &records = selector.cstPointRecordsFound();

      if (records.empty())
        {
          local << "    No constrained point found on this rank.\n";
          continue;
        }

      for (unsigned int pointID = 0; pointID < records.size(); ++pointID)
        {
          const auto &record = records[pointID];

          local << "    Point " << pointID << ":\n";
          local << "      coordinate = (" << record.point[0] << ", "
                << record.point[1] << ", " << record.point[2] << ")\n";

          local << "      number of constraints on this point = "
                << record.entryIndices.size() << "\n";

          for (unsigned int entryID = 0; entryID < record.entryIndices.size();
               ++entryID)
            {
              const std::size_t outputID = record.entryIndices[entryID];

              AssertThrow(outputID < selector.cstOutput().size(),
                          ExcMessage(
                            "Invalid entry index in cstListString()."));

              const CstEntryResult<Tria> &entry =
                selector.cstOutput()[outputID];


              local << "        Entry " << entryID << ":\n";
              local << "          dofAtPnt  = " << entry.getDofAtPnt() << "\n";

              if (entry.hasValidDoF())
                {
                  local << "          globalDoF = " << entry.getGlobalDoF()
                        << "\n";
                }
              else
                {
                  local << "          globalDoF = invalid\n";
                }

              local << "          value     = " << entry.getValue() << "\n";
            }
        }
    }

  const std::string localString = local.str();

  std::ostringstream global;
  global << "cstListString: " << name << std::endl;

  if constexpr (is_distributed)
    {
      const std::vector<std::string> allRankStrings =
        Utilities::MPI::all_gather(*__mpiInfo.mpiCommPtr(), localString);

      global << "========================================\n";
      global << "Constraint list: " << name << "\n";
      global << "MPI synchronized output from all ranks\n";
      global << "Number of ranks: " << allRankStrings.size() << "\n";
      global << "========================================\n";

      for (unsigned int r = 0; r < allRankStrings.size(); ++r)
        {
          global << "\n";
          global << "----------------------------------------\n";
          global << "Rank " << r << "\n";
          global << "----------------------------------------\n";
          global << allRankStrings[r];
        }
    }
  else
    {
      global << "========================================\n";
      global << "Constraint list: " << name << "\n";
      global << "Serial output\n";
      global << "========================================\n";
      global << localString;
    }

  return global.str();
}



template class bcs::CstMaker<RTria<1, 1>>;
template class bcs::CstMaker<RTria<1, 2>>;
template class bcs::CstMaker<RTria<1, 3>>;

template class bcs::CstMaker<RTria<2, 2>>;
template class bcs::CstMaker<RTria<2, 3>>;

template class bcs::CstMaker<RTria<3, 3>>;



#if defined(DISTRIBUTED_TRIA) && DISTRIBUTED_TRIA
template class bcs::CstMaker<DTria<1, 1>>;
template class bcs::CstMaker<DTria<1, 2>>;
template class bcs::CstMaker<DTria<1, 3>>;

template class bcs::CstMaker<DTria<2, 2>>;
template class bcs::CstMaker<DTria<2, 3>>;

template class bcs::CstMaker<DTria<3, 3>>;
#endif
