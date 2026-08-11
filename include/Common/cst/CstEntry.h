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


  /**
   The class is used as a standard input argument to describe and record the
   boundary conditions on points, including ith-dof *at the point*, and the
   prescribed value for this dof. This class is *node-independent* and only
   categorizes the boundary condition by both **ith dofs and values**.

   For example:

    A system has 5 primary known at nodes: (u_x, u_y, u_z, T, d).
    The ith dofs at node:                  [0,   1,   2,   3, 4].

    2 nodes [120.0, 105.5, -32.5] and [-25.0, 15.0, 12.5] have 3 constraints:
        - fixed displacement along y,
        - constantly constrained displacement 1.5 long z
        - constantly constrained temperature = 20.0 for T
   The CstEntry<Tria> is node-independent, and these three constraints can be
   described as:

   ```cpp
    // u_y = 0
    const CstEntry<Tria>  u_y(1); // or const CstEntry<Tria>  fixeX(1, 0.0);
    // u_z = 1.5
    const CstEntry<Tria>  u_z(2, 1.5);
    // T = 20.0
    const CstEntry<Tria>  T(3, 20.0);
   ```


   For more complex boundary conditions, such as a constrained value is
   *function-dependent*, the given value in the constructor is not important in
   the declaration and will be updated by a specific lambda expression passed
   into the constructor of CstPnt<Tria> or CstFunc<Tria>.
  */
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



  /**
   This class is an *internal* interface for the resolved constraint as an
   output.
   */
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
