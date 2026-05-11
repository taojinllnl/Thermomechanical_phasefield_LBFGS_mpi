//
//  CstHelper.h
//  main
//
//

#ifndef CstHelper_h
#define CstHelper_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>

#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "../BlockDesc.h"


namespace bcs
{

  class ConstrainedPnt;
  class ConstrainedFunc;
  class CstHelper;
  class CstSelector;



  struct ConstraintEntry
  {
  public:
    using ValueFunc =
      std::function<double(const double, const double, const double)>;

    ConstraintEntry(const unsigned int dofAtPnt, const double value = 0.0);

    ConstraintEntry(const ConstraintEntry &view);

    ::dealii::types::global_dof_index
    getGlobalDoF() const;

    bool
    hasValidDoF() const;

    double
    getValue() const;

  private:
    double                            value;
    const unsigned int                dofAtPnt;
    ::dealii::types::global_dof_index dof;

    void
    setDof(const ::dealii::types::global_dof_index new_dof);


    void
    reset();


    friend class ConstrainedPnt;
    friend class ConstrainedFunc;
    friend class CstHelper;
  };

  class CstSelector
  {
  protected:
    std::vector<ConstraintEntry> _csts;

  public:
    virtual ~CstSelector() = default;


    friend class CstHelper;
  };

  class ConstrainedPnt : public CstSelector
  {
  private:
    std::shared_ptr<std::array<double, 3>> pntCoorinates{};
    std::vector<ConstraintEntry>           csts;

    const std::size_t nCsts;

    const std::vector<ConstraintEntry> &
    cstEntrys() const;

  public:
    const double tol;

    virtual ~ConstrainedPnt() = default;

    ConstrainedPnt(const std::array<double, 3>        &point,
                   const std::vector<ConstraintEntry> &csts,
                   const double                        tol = 1e-9);

    ConstrainedPnt(const ConstrainedPnt &cstPnt);

    template <int spacedim>
    bool
    isSelectedPnt(const ::dealii::Point<spacedim> &point);


    template <typename Tria>
    std::string
    debugString(const Tria        &tria,
                const std::string &name = "ConstrainedPnt") const;

    friend class CstHelper;
  };

  class ConstrainedFunc : public CstSelector
  {
  public:
    using SelFunc =
      std::function<bool(const double, const double, const double)>;


    using ValuesAtPntFunc = std::function<
      void(const double, const double, const double, std::vector<double> &)>;

  private:
    std::size_t nCsts;
    std::size_t nPnts;

    std::shared_ptr<SelFunc>         selectorFunc{};
    std::shared_ptr<ValuesAtPntFunc> valuesFunc{};

    std::vector<ConstraintEntry> csts;

    std::set<::dealii::types::global_dof_index> dofsFound;
    std::vector<ConstraintEntry>                cstsFound;
    std::vector<std::array<double, 3>>          pointsFound;


    template <int spacedim>
    void
    addEntry(const ::dealii::Point<spacedim> &point, const unsigned int nDoFs);

    const std::vector<ConstraintEntry> &
    cstEntrys() const;

  public:
    virtual ~ConstrainedFunc() = default;

    ConstrainedFunc(const SelFunc                      &selectorFunc,
                    const std::vector<ConstraintEntry> &csts);

    ConstrainedFunc(const SelFunc                      &selectorFunc,
                    const std::vector<ConstraintEntry> &csts,
                    const ValuesAtPntFunc              &valuesFunc);

    ConstrainedFunc(const ConstrainedFunc &cstFunc);

    template <int spacedim>
    bool
    isSelectedPnt(const ::dealii::Point<spacedim> &point);


    template <typename Tria>
    std::string
    debugString(const Tria        &tria,
                const std::string &name = "ConstrainedFunc") const;

    friend class CstHelper;
  };



  class CstHelper
  {
  private:
    template <typename CstType>
    static void
    __verifyInput(const std::vector<CstType> &cstPnts,
                  const unsigned int          nDoFs);


    template <typename Tria, typename CstType>
    static void
    __verifyAllFound(const Tria &tria, const std::vector<CstType> &cstPnts);

    template <typename Tria, typename CstType>
    static void
    __verifyNoRepeat(const Tria &tria, const std::vector<CstType> &cstPnts);


  public:
    static void
    cstReinit(::dealii::AffineConstraints<double> &constraints,
              const ::dealii::IndexSet            &locally_owned_dofs,
              const ::dealii::IndexSet            &locally_relevant_dofs);


    static void
    addPntCst(::dealii::AffineConstraints<double>    &cst,
              const ::dealii::types::global_dof_index dof,
              const double                            value = 0.0);


    template <typename CellIter>
    static void
    addPntCst(::dealii::AffineConstraints<double> &cst,
              const CellIter                      &cell,
              const unsigned int                   ithVertex,
              const unsigned int                   ithLocalDoFAtVertex,
              const double                         value = 0.0);

    template <typename Tria, typename CstType, bool isAtBoundary = true>
    static void
    addPntCsts(const Tria                                        &tria,
               const ::dealii::DoFHandler<Tria::dimension,
                                          Tria::space_dimension> &dofHandler,
               ::dealii::AffineConstraints<double>               &cst,
               const std::initializer_list<CstType>              &cstPnts,
               const bool verifyAllFound        = true,
               const bool verifyNoRepeatedEntry = true);

    template <int dim, int spacedim = dim>
    static void
    extractDoFs(std::vector<::dealii::types::global_dof_index> &dofs,
                const typename ::dealii::Triangulation<dim, spacedim>::
                  active_vertex_iterator                  &vertex,
                const ::dealii::DoFHandler<dim, spacedim> &dof_handler,
                const std::size_t                          nDoFsPerVertex);
  };



  template <int spacedim>
  bool
  ConstrainedPnt ::isSelectedPnt(const ::dealii::Point<spacedim> &point)
  {
    bool isCurrentPoint = false;

    if (pntCoorinates)
      {
        const std::array<double, 3> &cstPnt = *pntCoorinates;
        if constexpr (spacedim == 2)
          {
            isCurrentPoint = std::fabs(point[0] - cstPnt[0]) < tol &&
                             std::fabs(point[1] - cstPnt[1]) < tol;
          }
        else if constexpr (spacedim == 3)
          {
            isCurrentPoint = std::fabs(point[0] - cstPnt[0]) < tol &&
                             std::fabs(point[1] - cstPnt[1]) < tol &&
                             std::fabs(point[2] - cstPnt[2]) < tol;
          }
      }
    return isCurrentPoint;
  }

  template <int spacedim>
  bool
  ConstrainedFunc ::isSelectedPnt(const ::dealii::Point<spacedim> &point)
  {
    bool isCurrentPoint = false;

    if (selectorFunc)
      {
        const auto &func = *selectorFunc;
        if constexpr (spacedim == 2)
          {
            isCurrentPoint = func(point[0], point[1], 0);
          }
        else if constexpr (spacedim == 3)
          {
            isCurrentPoint = func(point[0], point[1], point[2]);
          }
      }

    return isCurrentPoint;
  }



  template <int spacedim>
  void
  ConstrainedFunc ::addEntry(const ::dealii::Point<spacedim> &point,
                             const unsigned int               nDoFs)
  {
    using namespace ::dealii;

    std::array<double, 3> pnt;

    if constexpr (spacedim == 2)
      {
        pnt = std::array<double, 3>{point[0], point[1], 0.0};
      }
    else if constexpr (spacedim == 3)
      {
        pnt = std::array<double, 3>{point[0], point[1], point[2]};
      }

    bool hasNewDoF = false;

    for (const ConstraintEntry &cstEntry : csts)
      {
        if (!cstEntry.hasValidDoF())
          continue;

        const auto dof = cstEntry.getGlobalDoF();

        if (dofsFound.find(dof) == dofsFound.end())
          {
            hasNewDoF = true;
            break;
          }
      }

    std::vector<double> values;

    if (hasNewDoF && valuesFunc)
      {
        values.resize(nDoFs);

        (*valuesFunc)(pnt[0], pnt[1], pnt[2], values);

        AssertThrow(
          values.size() >= nDoFs,
          ExcMessage(
            "ConstrainedFunc::valuesFunc returned/resized values with size smaller than nDoFs."));
      }

    bool insertedAnyDoF = false;

    for (ConstraintEntry &cstEntry : csts)
      {
        if (cstEntry.hasValidDoF())
          {
            const auto dof = cstEntry.getGlobalDoF();

            if (dofsFound.find(dof) == dofsFound.end())
              {
                if (valuesFunc)
                  {
                    AssertThrow(
                      cstEntry.dofAtPnt < values.size(),
                      ExcMessage(
                        "valuesFunc returned insufficient values for cstEntry.dofAtPnt."));

                    cstEntry.value = values[cstEntry.dofAtPnt];
                  }

                const bool inserted = dofsFound.insert(dof).second;

                AssertThrow(
                  inserted,
                  ExcMessage(
                    "Internal error: DoF was expected to be new but insertion failed."));

                cstsFound.emplace_back(cstEntry);
                ++nCsts;

                insertedAnyDoF = true;
              }
          }

        cstEntry.reset();
      }

    if (insertedAnyDoF)
      {
        pointsFound.emplace_back(pnt);
        ++nPnts;
      }
  }



  template <int dim, int spacedim>
  void
  CstHelper ::extractDoFs(
    std::vector<::dealii::types::global_dof_index> &dofs,
    const typename ::dealii::Triangulation<dim,
                                           spacedim>::active_vertex_iterator
                                              &vertex,
    const ::dealii::DoFHandler<dim, spacedim> &dof_handler,
    const std::size_t                          nDoFsPerVertex)
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



  template <typename CellIter>
  void
  CstHelper ::addPntCst(::dealii::AffineConstraints<double> &cst,
                        const CellIter                      &cell,
                        const unsigned int                   ithVertex,
                        const unsigned int ithLocalDoFAtVertex,
                        const double       value)
  {
    using namespace ::dealii::types;
    const global_dof_index dof =
      cell->vertex_dof_index(ithVertex, ithLocalDoFAtVertex);
    if (!cst.is_constrained(dof))
      {
        cst.add_line(dof);
        cst.set_inhomogeneity(dof, value);
      }
  }



  template <typename Tria, typename CstType, bool isAtBoundary>
  void
  CstHelper ::addPntCsts(
    const Tria &tria,
    const ::dealii::DoFHandler<Tria::dimension, Tria::space_dimension>
                                         &dofHandler,
    ::dealii::AffineConstraints<double>  &cst,
    const std::initializer_list<CstType> &cstPnts_,
    const bool                            verifyAllFound,
    const bool                            verifyNoRepeatedEntry)
  {
    using namespace ::dealii;
    static const unsigned int dim      = Tria::dimension;
    static const unsigned int spacedim = Tria::space_dimension;
    static constexpr bool     is_distributed =
      std::is_same_v<std::remove_cv_t<Tria>,
                     parallel::distributed::Triangulation<dim, spacedim>>;


    static const bool isConstrainedPnt =
      std::is_same_v<CstType, ConstrainedPnt>;
    static const bool isConstrainedFunc =
      std::is_same_v<CstType, ConstrainedFunc>;

    static_assert(
      isConstrainedPnt || isConstrainedFunc,
      "CstHelper::addPntCsts only supports ConstrainedPnt or ConstrainedFunc.");

    const unsigned int nDoFs = dofHandler.get_fe().dofs_per_vertex;

    // record if there is a constrained on current rank
    bool hasCst = false;

    // copy the given csts
    std::vector<CstType> cstPnts = cstPnts_;

    __verifyInput(cstPnts, nDoFs);

    if constexpr (is_distributed)
      {
        //        std::vector<bool> locally_owned_vertices =
        //        GridTools::get_locally_owned_vertices(tria);
        for (auto const &cell : dofHandler.active_cell_iterators())
          {
            // skip ghost cells or cells that are not at boundary
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
                // skip vertices that are not owned by current rank.
                // This operation is necessary because some vertices are shared
                // by cells owned by other ranks. Or, it may cause unexpected
                // results.
                //                if
                //                (!locally_owned_vertices[cell->vertex_index(vertex)])
                //                    continue;

                // obtain vertex
                const Point<spacedim> point = cell->vertex(vertex);

                // loop over prescribed constraints
                for (unsigned int j = 0; j < cstPnts.size(); ++j)
                  {
                    // j-th prescribed ConstrainedPnt / ConstrainedFunc
                    CstType &cstPoint = cstPnts[j];

                    //                    if constexpr (isConstrainedPnt)
                    //                    {
                    //                        // skip further operations, if
                    //                        this point has been handled.
                    //                        if(cstPoint.isFound)
                    //                            continue;
                    //                    }

                    // the vertex is close enough to the constrained point
                    if (cstPoint.isSelectedPnt(point))
                      {
                        //                        // the point is found
                        //                        if constexpr
                        //                        (isConstrainedPnt)
                        //                        {
                        //                            cstPoint.isFound = true;
                        //                        }

                        bool hasValidDoFOnPnt = false;
                        // extract constrained DoFs
                        for (ConstraintEntry &cstEntry : cstPoint.csts)
                          {
                            if (cstEntry.hasValidDoF())
                              {
                                continue;
                              }

                            const auto dof =
                              cell->vertex_dof_index(vertex, cstEntry.dofAtPnt);

                            if (!dofHandler.locally_owned_dofs().is_element(
                                  dof))
                              continue;

                            // The constrained point is owned by current rank.
                            hasCst           = true;
                            hasValidDoFOnPnt = true;

                            cstEntry.setDof(dof);
                          }

                        if constexpr (isConstrainedFunc)
                          {
                            // to filter that the dof is not owned by a rank but
                            // the point is owned by this rank
                            if (hasValidDoFOnPnt)
                              cstPoint.addEntry(point, nDoFs);
                          }
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
        std::vector<types::global_dof_index> nodeDoFs(nDoFs);


        for (; vertex_itr != tria.end_vertex(); ++vertex_itr)
          {
            for (unsigned int j = 0; j < cstPnts.size(); ++j)
              {
                // j-th prescribed ConstrainedPnt
                CstType &cstPoint = cstPnts[j];

                //                if constexpr (isConstrainedPnt)
                //                {
                //                    // skip further operations, if this point
                //                    has been handled. if(cstPoint.isFound)
                //                        continue;
                //                }


                const Point<spacedim> &vertex = vertex_itr->vertex();
                // the vertex is close enough to the constrained point
                if (cstPoint.isSelectedPnt(vertex))
                  {
                    //                    // the point is found
                    //                    if constexpr (isConstrainedPnt)
                    //                    {
                    //                        cstPoint.isFound = true;
                    //                    }

                    // The constrained point is owned by current rank.

                    bool hasValidDoFOnPnt = false;
                    // extract constrained DoFs
                    CstHelper::extractDoFs(nodeDoFs,
                                           vertex_itr,
                                           dofHandler,
                                           nDoFs);

                    for (ConstraintEntry &cstEntry : cstPoint.csts)
                      {
                        if (!cstEntry.hasValidDoF())
                          {
                            hasCst           = true;
                            hasValidDoFOnPnt = true;
                            cstEntry.setDof(nodeDoFs[cstEntry.dofAtPnt]);
                          }
                      }

                    if constexpr (isConstrainedFunc)
                      {
                        if (hasValidDoFOnPnt)
                          cstPoint.addEntry(vertex_itr->vertex(), nDoFs);
                      }
                  }
              }
          }
      }

    if (verifyAllFound)
      __verifyAllFound(tria, cstPnts);
    if (verifyNoRepeatedEntry)
      __verifyNoRepeat(tria, cstPnts);

    // in MPI mode: only the rank with constrained points will apply BCs.
    if (hasCst)
      {
        for (unsigned int i = 0; i < cstPnts.size(); ++i)
          {
            const CstType &cstPoint = cstPnts[i];

            const std::size_t nCsts = cstPoint.nCsts;


            if (nCsts > 0)
              {
                for (unsigned int j = 0; j < nCsts; ++j)
                  {
                    const ConstraintEntry &cstEntry = cstPoint.cstEntrys()[j];

                    if (cstEntry.hasValidDoF() &&
                        !cst.is_constrained(cstEntry.getGlobalDoF()))
                      {
                        cst.add_line(cstEntry.getGlobalDoF());

                        cst.set_inhomogeneity(cstEntry.getGlobalDoF(),
                                              cstEntry.getValue());
                      }
                  }
              }
          }
      } // loop over constrainted points
  }


  template <typename CstType>
  void
  CstHelper ::__verifyInput(const std::vector<CstType> &cstPnts,
                            const unsigned int          nDoFs)
  {
    using namespace ::dealii;


    for (unsigned int ith = 0; ith < cstPnts.size(); ++ith)
      {
        std::set<unsigned int> used_local_dof_ids;

        for (const auto &entry : cstPnts[ith].csts)
          {
            AssertThrow(entry.dofAtPnt < nDoFs,
                        ExcMessage(
                          "CstView::dofAtPnt is larger than dofs_per_vertex."));

            AssertThrow(used_local_dof_ids.insert(entry.dofAtPnt).second,
                        ExcMessage(
                          std::to_string(ith) +
                          "-th constraint selector has repeated dofAtPnt = " +
                          std::to_string(entry.dofAtPnt)));
          }
      }
  }


  template <typename Tria, typename CstType>
  void
  CstHelper ::__verifyAllFound(const Tria                 &tria,
                               const std::vector<CstType> &cstPnts)
  {
    using namespace ::dealii;

    static constexpr unsigned int dim      = Tria::dimension;
    static constexpr unsigned int spacedim = Tria::space_dimension;

    static constexpr bool is_distributed =
      std::is_same_v<std::remove_cv_t<Tria>,
                     parallel::distributed::Triangulation<dim, spacedim>>;

    static constexpr bool isConstrainedPnt =
      std::is_same_v<CstType, ConstrainedPnt>;
    static constexpr bool isConstrainedFunc =
      std::is_same_v<CstType, ConstrainedFunc>;

    static_assert(
      isConstrainedPnt || isConstrainedFunc,
      "CstHelper::__verifyAllFound only supports ConstrainedPnt or ConstrainedFunc.");

    for (unsigned int ith = 0; ith < cstPnts.size(); ++ith)
      {
        const CstType &cstPoint = cstPnts[ith];

        if constexpr (isConstrainedPnt)
          {
            for (unsigned int k = 0; k < cstPoint.csts.size(); ++k)
              {
                const unsigned int local_found =
                  cstPoint.csts[k].hasValidDoF() ? 1 : 0;

                unsigned int global_found = local_found;

                if constexpr (is_distributed)
                  {
                    global_found =
                      Utilities::MPI::sum(local_found, tria.get_communicator());
                  }

                AssertThrow(
                  global_found == 1,
                  ExcMessage(
                    std::to_string(ith) + "-th ConstrainedPnt, " +
                    std::to_string(k) +
                    "-th ConstraintEntry was not uniquely found by owning DoF rank. Found count = " +
                    std::to_string(global_found) + "."));
              }
          }
        else if constexpr (isConstrainedFunc)
          {
            const unsigned int local_found = cstPoint.nCsts > 0 ? 1 : 0;

            unsigned int global_found = local_found;

            if constexpr (is_distributed)
              {
                global_found =
                  Utilities::MPI::sum(local_found, tria.get_communicator());
              }

            AssertThrow(
              global_found > 0,
              ExcMessage(
                std::to_string(ith) +
                "-th ConstrainedFunc did not find any valid constrained DoF."));
          }
      }
  }


  template <typename Tria, typename CstType>
  void
  CstHelper ::__verifyNoRepeat(const Tria                 &tria,
                               const std::vector<CstType> &cstPnts)
  {
    using namespace ::dealii;
    using global_dof_index = ::dealii::types::global_dof_index;

    static constexpr unsigned int dim      = Tria::dimension;
    static constexpr unsigned int spacedim = Tria::space_dimension;

    static constexpr bool is_distributed =
      std::is_same_v<std::remove_cv_t<Tria>,
                     parallel::distributed::Triangulation<dim, spacedim>>;

    static constexpr bool isConstrainedPnt =
      std::is_same_v<CstType, ConstrainedPnt>;
    static constexpr bool isConstrainedFunc =
      std::is_same_v<CstType, ConstrainedFunc>;

    static_assert(
      isConstrainedPnt || isConstrainedFunc,
      "CstHelper::__verifyNoRepeat only supports ConstrainedPnt or ConstrainedFunc.");

    std::set<global_dof_index> visited_dofs;

    bool has_local_repetition = false;

    global_dof_index repeated_dof = numbers::invalid_dof_index;

    for (unsigned int ith = 0; ith < cstPnts.size(); ++ith)
      {
        const CstType &cstPoint = cstPnts[ith];

        const auto &entries = cstPoint.cstEntrys();

        for (unsigned int k = 0; k < entries.size(); ++k)
          {
            const ConstraintEntry &cstEntry = entries[k];

            if (!cstEntry.hasValidDoF())
              continue;

            const global_dof_index dof = cstEntry.getGlobalDoF();

            if (!visited_dofs.insert(dof).second)
              {
                has_local_repetition = true;
                repeated_dof         = dof;
                break;
              }
          }

        if (has_local_repetition)
          break;
      }

    const unsigned int local_repetition = has_local_repetition ? 1 : 0;

    unsigned int global_repetition = local_repetition;

    if constexpr (is_distributed)
      {
        global_repetition =
          Utilities::MPI::sum(local_repetition, tria.get_communicator());
      }

    AssertThrow(
      global_repetition == 0,
      ExcMessage("Repeated constrained DoF detected in CstHelper::addPntCsts." +
                 (has_local_repetition ? (" Repeated local DoF = " +
                                          std::to_string(repeated_dof) + ".") :
                                         std::string(""))));
  }



  template <typename Tria>
  std::string
  ConstrainedPnt ::debugString(const Tria &tria, const std::string &name) const
  {
    using namespace ::dealii;

    static constexpr unsigned int dim      = Tria::dimension;
    static constexpr unsigned int spacedim = Tria::space_dimension;

    static constexpr bool is_distributed =
      std::is_same_v<std::remove_cv_t<Tria>,
                     parallel::distributed::Triangulation<dim, spacedim>>;

    std::ostringstream local_out;

    unsigned int rank    = 0;
    unsigned int n_ranks = 1;

    if constexpr (is_distributed)
      {
        rank    = Utilities::MPI::this_mpi_process(tria.get_communicator());
        n_ranks = Utilities::MPI::n_mpi_processes(tria.get_communicator());
      }

    local_out << std::setprecision(16);
    local_out << "==============================\n";
    local_out << name << " debug info\n";
    local_out << "rank: " << rank << " / " << n_ranks << "\n";

    std::array<double, 3> point{{0.0, 0.0, 0.0}};
    bool                  has_point = false;

    if (pntCoorinates)
      {
        point     = *pntCoorinates;
        has_point = true;

        local_out << "prescribed point: (" << point[0] << ", " << point[1]
                  << ", " << point[2] << ")\n";
      }
    else
      {
        local_out << "prescribed point: <null>\n";
      }

    local_out << "number of prescribed entries: " << csts.size() << "\n";

    for (unsigned int i = 0; i < csts.size(); ++i)
      {
        const ConstraintEntry &entry = csts[i];

        local_out << "  entry[" << i << "]\n";

        if (has_point)
          {
            local_out << "    point: (" << point[0] << ", " << point[1] << ", "
                      << point[2] << ")\n";
          }
        else
          {
            local_out << "    point: <null>\n";
          }

        local_out << "    local dof at point: " << entry.dofAtPnt << "\n";

        if (entry.hasValidDoF())
          {
            local_out << "    global dof: " << entry.getGlobalDoF() << "\n";
          }
        else
          {
            local_out << "    global dof: <invalid / not owned on this rank>\n";
          }

        local_out << "    value: " << entry.getValue() << "\n";
      }

    local_out << "==============================\n";

    const std::string local_string = local_out.str();

    if constexpr (is_distributed)
      {
        const std::vector<std::string> all_strings =
          Utilities::MPI::all_gather(tria.get_communicator(), local_string);

        std::ostringstream global_out;

        global_out << "========== MPI gathered " << name
                   << " debug info ==========\n";

        for (unsigned int r = 0; r < all_strings.size(); ++r)
          {
            global_out << "\n----- rank " << r << " -----\n";
            global_out << all_strings[r];
          }

        global_out << "========== end MPI gathered " << name
                   << " debug info ==========\n";

        return global_out.str();
      }
    else
      {
        return local_string;
      }
  }


  template <typename Tria>
  std::string
  ConstrainedFunc ::debugString(const Tria &tria, const std::string &name) const
  {
    using namespace ::dealii;

    static constexpr unsigned int dim      = Tria::dimension;
    static constexpr unsigned int spacedim = Tria::space_dimension;

    static constexpr bool is_distributed =
      std::is_same_v<std::remove_cv_t<Tria>,
                     parallel::distributed::Triangulation<dim, spacedim>>;

    std::ostringstream local_out;

    unsigned int rank    = 0;
    unsigned int n_ranks = 1;

    if constexpr (is_distributed)
      {
        rank    = Utilities::MPI::this_mpi_process(tria.get_communicator());
        n_ranks = Utilities::MPI::n_mpi_processes(tria.get_communicator());
      }

    local_out << std::setprecision(16);
    local_out << "==============================\n";
    local_out << name << " debug info\n";
    local_out << "rank: " << rank << " / " << n_ranks << "\n";

    local_out << "number of matched points on this rank: " << nPnts << "\n";

    local_out << "number of constrained entries on this rank: " << nCsts
              << "\n";

    local_out << "number of stored pointsFound: " << pointsFound.size() << "\n";

    local_out << "number of stored cstsFound: " << cstsFound.size() << "\n";

    local_out << "number of stored cstsFoundPoints: " << pointsFound.size()
              << "\n";

    AssertThrow(
      cstsFound.size() == pointsFound.size(),
      ExcMessage(
        "ConstrainedFunc::debugString(): cstsFound and cstsFoundPoints have inconsistent sizes."));

    if (!pointsFound.empty())
      {
        local_out << "\nmatched points:\n";

        for (unsigned int i = 0; i < pointsFound.size(); ++i)
          {
            local_out << "  point[" << i << "]: (" << pointsFound[i][0] << ", "
                      << pointsFound[i][1] << ", " << pointsFound[i][2]
                      << ")\n";
          }
      }

    local_out << "\nconstrained entries:\n";

    for (unsigned int i = 0; i < cstsFound.size(); ++i)
      {
        const ConstraintEntry &entry = cstsFound[i];
        const auto            &point = pointsFound[i];

        local_out << "  entry[" << i << "]\n";

        local_out << "    point: (" << point[0] << ", " << point[1] << ", "
                  << point[2] << ")\n";

        local_out << "    local dof at point: " << entry.dofAtPnt << "\n";

        if (entry.hasValidDoF())
          {
            local_out << "    global dof: " << entry.getGlobalDoF() << "\n";
          }
        else
          {
            local_out << "    global dof: <invalid>\n";
          }

        local_out << "    value: " << entry.getValue() << "\n";
      }

    local_out << "==============================\n";

    const std::string local_string = local_out.str();

    if constexpr (is_distributed)
      {
        const std::vector<std::string> all_strings =
          Utilities::MPI::all_gather(tria.get_communicator(), local_string);

        std::ostringstream global_out;

        global_out << "========== MPI gathered " << name
                   << " debug info ==========\n";

        for (unsigned int r = 0; r < all_strings.size(); ++r)
          {
            global_out << "\n----- rank " << r << " -----\n";
            global_out << all_strings[r];
          }

        global_out << "========== end MPI gathered " << name
                   << " debug info ==========\n";

        return global_out.str();
      }
    else
      {
        return local_string;
      }
  }



} // namespace bcs


#endif /* CstHelper_h */
