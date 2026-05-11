//
//  CstEntry.h
//  main
//
//

#ifndef CstEntry_h
#define CstEntry_h

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

#include "../Traits.h"

namespace bcs
{

  template <typename Tria>
  struct CstEntry
  {
  public:
    using LocalDoF = unsigned int;

  private:
    const LocalDoF dofAtPnt;
    double         value;


  public:
    explicit CstEntry(const LocalDoF dofAtPnt, const double value = 0.0);

    CstEntry(const CstEntry &entry);
    CstEntry(CstEntry &&entry);

    LocalDoF
    getDofAtPnt() const;


    double
    getValue() const;


    void
    setValue(const double value);
  };



  template <typename Tria>
  struct CstEntryResult : public CstEntry<Tria>
  {
  public:
    using GlobalDoF = ::dealii::types::global_dof_index;
    using LocalDoF  = typename CstEntry<Tria>::LocalDoF;

  private:
    GlobalDoF dof;


  public:
    explicit CstEntryResult(const LocalDoF dofAtPnt, const double value = 0.0);

    CstEntryResult(const CstEntryResult &entry);
    CstEntryResult(CstEntryResult &&entry);

    explicit CstEntryResult(const CstEntry<Tria> &entry);
    explicit CstEntryResult(CstEntry<Tria> &&entry);


    bool
    hasValidDoF() const;

    GlobalDoF
    getGlobalDoF() const;
    void
    setGlobalDof(const GlobalDoF new_dof);

    void
    resetDoF();
  };



} // namespace bcs

#endif /* CstEntry_h */
