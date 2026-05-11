//
//  CstMaker.h
//  main
//
//

#ifndef CstMaker_h
#define CstMaker_h


#include "../MPIInfo.h"
#include "CstFunc.h"
#include "CstPnt.h"
#include "CstSelectorBase.h"


namespace bcs
{

  // The interface to apply the constraints for a list of nodes.
  template <typename Tria>
  class CstMaker
  {
  public:
    static const unsigned int dim      = Tria::dimension;
    static const unsigned int spacedim = Tria::space_dimension;

    static constexpr bool is_distributed = std::is_same_v<
      std::remove_cv_t<Tria>,
      ::dealii::parallel::distributed::Triangulation<dim, spacedim>>;

    using GlobalDoF = typename CstEntryResult<Tria>::GlobalDoF;

    using RVertexIter =
      typename ::dealii::Triangulation<dim, spacedim>::active_vertex_iterator;
    using DoFHandler  = ::dealii::DoFHandler<dim, spacedim>;
    using DoFList     = std::vector<::dealii::types::global_dof_index>;
    using Constraints = ::dealii::AffineConstraints<double>;


  private:
    const common::MPIInfo                              &__mpiInfo;
    std::vector<std::unique_ptr<CstSelectorBase<Tria>>> __cstSelectors;

    bool isReady           = false;
    bool hasBoundSelectors = false;
    bool hasFoundCsts      = false;


    void
    __verifyInput(const unsigned int nDoFs);


    void
    __verifyAllFound();

    void
    __verifyNoRepeat();


  public:
    CstMaker(const common::MPIInfo &mpiInfo);

    virtual ~CstMaker() = default;

    CstMaker(const CstMaker &) = delete;
    CstMaker(CstMaker &&)      = delete;
    CstMaker &
    operator=(const CstMaker &) = delete;
    CstMaker &
    operator=(CstMaker &&) = delete;


    void
    cstReinit(Constraints              &constraints,
              const ::dealii::IndexSet &locally_owned_dofs,
              const ::dealii::IndexSet &locally_relevant_dofs);


    void
    extractDoFsAtVetex(DoFList           &dofs,
                       const RVertexIter &vertex,
                       const DoFHandler  &dof_handler,
                       const std::size_t  nDoFsPerVertex);

    void
    eliminateSelectors();
    CstMaker &
    addCstSelector(const CstSelectorBase<Tria> &selector);
    bool
    isAddedSelectors() const;

    template <bool isAtBoundary = true>
    void
    preparePntCst(const Tria       &tria,
                  const DoFHandler &dofHandler,
                  const bool        verifyAllFound        = true,
                  const bool        verifyNoRepeatedEntry = true);


    void
    updatePntCstValues(const DoFHandler &dofHandler,
                       const bool        verifyAllFound        = true,
                       const bool        verifyNoRepeatedEntry = true);

    void
    applyPntCsts(Constraints &cst,
                 const bool   verifyAllFound        = true,
                 const bool   verifyNoRepeatedEntry = true);


    void
    makeCstIfPrepareNeeded(bool             &dofChangedflag,
                           Constraints      &cst,
                           const Tria       &tria,
                           const DoFHandler &dofHandler,
                           const bool        updateValues          = false,
                           const bool        verifyAllFound        = false,
                           const bool        verifyNoRepeatedEntry = false);


    std::string
    cstListString(const std::string &name) const;
  };



  template <typename Tria>
  template <bool isAtBoundary>
  void
  CstMaker<Tria>::preparePntCst(const Tria       &tria,
                                const DoFHandler &dofHandler,
                                const bool        verifyAllFound,
                                const bool        verifyNoRepeatedEntry)
  {
    using namespace ::dealii;
    const unsigned int nDoFs = dofHandler.get_fe().dofs_per_vertex;

    // reset the status of dof founding process
    hasFoundCsts = false;

    __verifyInput(nDoFs);

    // clear data and allocate memory in __cstSelectors
    for (auto &selectorPtr : __cstSelectors)
      {
        selectorPtr->init(nDoFs);
      }

    if constexpr (is_distributed)
      {
        const dealii::IndexSet &locallyOwnedDoFs =
          dofHandler.locally_owned_dofs();
        for (auto const &cell : dofHandler.active_cell_iterators())
          {
            // skip ghost cells or cells that are not at boundary
            // Note: at_boundary can accelerate the iterations but it cannot
            // gaurantee the found vertices are at boundary.
            if constexpr (isAtBoundary)
              {
                if (!cell->is_locally_owned() || !cell->at_boundary())
                  continue;
              }
            else
              {
                if (!cell->is_locally_owned())
                  continue;
              }


            // loop over vertices on the locally owned cells at boundary
            for (const auto vertex : cell->vertex_indices())
              {
                // obtain vertex
                const Point<spacedim> point = cell->vertex(vertex);

                // loop over prescribed constraints
                for (unsigned int j = 0; j < __cstSelectors.size(); ++j)
                  {
                    // j-th prescribed CstPnt / CstFunc
                    CstSelectorBase<Tria> &cstSelector = *__cstSelectors[j];

                    // the vertex is close enough to the constrained point
                    if (cstSelector.isSelectedPnt(point))
                      {
                        bool hasValidDoFOnPnt = false;
                        // extract constrained DoFs
                        for (const CstEntry<Tria> &cstEntry :
                             cstSelector.cstInput())
                          {
                            // skip, if the dof has been added to the cache
                            if (!cstSelector.hasNoCachedDoF(cstEntry))
                              {
                                continue;
                              }

                            // extract global dof
                            const GlobalDoF dof =
                              cell->vertex_dof_index(vertex,
                                                     cstEntry.getDofAtPnt());

                            // skip, if the dof is not valid
                            if (!CstSelectorBase<Tria>::isDoFValid(dof))
                              continue;

                            // verify if this dof is locally owned
                            if (!locallyOwnedDoFs.is_element(dof))
                              continue;

                            // The constrained point is owned by current rank.
                            hasFoundCsts     = true;
                            hasValidDoFOnPnt = true;
                            // put a dof into selector's cache
                            cstSelector.addDoF2Cache(nDoFs, dof, cstEntry);
                          }

                        if (hasValidDoFOnPnt)
                          cstSelector.createOutputByPoint(point, nDoFs);
                      }
                  } // loop over constrainted pnts
              }     // loop over vertices in cell
          }         // loop over cells
      }
    else
      {
        typename Triangulation<dim, spacedim>::active_vertex_iterator
          vertex_itr;
        vertex_itr = tria.begin_active_vertex();


        for (; vertex_itr != tria.end_vertex(); ++vertex_itr)
          {
            for (unsigned int j = 0; j < __cstSelectors.size(); ++j)
              {
                // j-th prescribed CstPnt
                auto &selectorPtr = __cstSelectors[j];

                const Point<spacedim> &vertex = vertex_itr->vertex();
                // the vertex is close enough to the constrained point
                if (selectorPtr->isSelectedPnt(vertex))
                  {
                    // extract constrained DoFs and inject to cache
                    extractDoFsAtVetex(selectorPtr->addDoFs2Cache(nDoFs),
                                       vertex_itr,
                                       dofHandler,
                                       nDoFs);

                    selectorPtr->createOutputByPoint(vertex_itr->vertex(),
                                                     nDoFs);
                    hasFoundCsts = true;
                  }
              }
          }
      }

    if (verifyAllFound)
      __verifyAllFound();
    if (verifyNoRepeatedEntry)
      __verifyNoRepeat();

    // change hasFoundCsts to true for all ranks
    if constexpr (is_distributed)
      {
        hasFoundCsts =
          ::common::MPIInfo::syncBool<true>(hasFoundCsts,
                                            *__mpiInfo.mpiCommPtr());
      }
    isReady = true;
  }

} // namespace bcs


#endif /* CstMaker_h */
