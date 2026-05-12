//
//  CstFunc.h
//  main
//
//

#ifndef CstFunc_h
#define CstFunc_h

#include "CstSelectorBase.h"

namespace bcs
{

  /**
   This class is an interface to select the nodes satisfied with a function, or
   the constrained values are function-based. This class is derived from
   CstSelectorBase<Tria>.


   For example:

   A system has 5 primary known at nodes: (u_x, u_y, u_z, T, d).
   The ith dofs at node:                  [0,   1,   2,   3, 4].

   The nodes on a top surface (z = h_t) are under displacement-controlled
   loading along z direction, governed by a function u_z = x^2 + y^2, and the
   nodes on the bottom surface (z = h_0) are fully fixed.

   These three constraints can be described as:

   ```cpp
   // u_x = 0
   const CstEntry<Tria>  u_x(0); // or const CstEntry<Tria>  fixeX(0, 0.0);
   // u_y = 0
   const CstEntry<Tria>  u_y(1); // or const CstEntry<Tria>  fixeY(1, 0.0);
   // u_z = 0
   const CstEntry<Tria>  u_z(2); // or const CstEntry<Tria> fixeZ(2, 0.0);
                                 // the default value is 0, it's not important
                                 // for top surface because function-base values
                                 // will added by lambda expression.
                                 // Thus, this CstEntry<Tria> can be reused.
   ```

   Define lambda expressions to select the nodes on the top and bottom:

   ```cpp
   // Assume the top surface is at z = 100, and the bottom surface is at z = 0;
   const double h = 100;



   const auto nodesBtm = [h](const double x, const double y, const double z) ->
   bool
   {
      return std::fabs(z) < 1e-9;
   };

   const auto nodesTop = [h](const double x, const double y, const double z) ->
   bool
   {
   return std::fabs(z - h) < 1e-9;
   };
   ```

   Define a lambda expression to setup the constrained values, following a
   function u_z = x^2 + y^2.

   ```cpp
   const auto cstValuesFunc = (const double x,
                               const double y,
                               const double z,
                               std::vector<double> & values)
   {
      values[2] = x*x + y*y; // put the calculated value to cache
   }
   ```

   The constrainted point can be built up by:

   ```cpp
   // fully fixed bottom surface
   CstFunc<Tria> btmCst(nodesBtm,           // functions to select nodes
                        {{u_x, u_y, u_z}}); // the constrained values are
   included

   // function-based constraints on top surface
   CstFunc<Tria> topCst(nodesTop,         // functions to select nodes
                        u_z,              // constrainted direction
                        cstValuesFunc);   // constrained values
   ```

   */
  template <typename Tria>
  class CstFunc : public CstSelectorBase<Tria>
  {
  public:
    static const int dim      = Tria::dimension;
    static const int spacedim = Tria::space_dimension;

    using SelFunc =
      std::function<bool(const double, const double, const double)>;


    using ValuesAtPntFunc = std::function<
      void(const double, const double, const double, std::vector<double> &)>;

  private:
    std::shared_ptr<SelFunc>         selectorFunc{};
    std::shared_ptr<ValuesAtPntFunc> valuesFunc{};


    virtual void
    assignValues(const double         x,
                 const double         y,
                 const double         z,
                 std::vector<double> &values) override;

  public:
    virtual ~CstFunc() = default;

    CstFunc(const SelFunc                     &selectorFunc,
            const std::vector<CstEntry<Tria>> &csts);

    CstFunc(const SelFunc                     &selectorFunc,
            const std::vector<CstEntry<Tria>> &csts,
            const ValuesAtPntFunc             &valuesFunc);


    CstFunc(const SelFunc &selectorFunc, const CstEntry<Tria> &csts);

    CstFunc(const SelFunc         &selectorFunc,
            const CstEntry<Tria>  &csts,
            const ValuesAtPntFunc &valuesFunc);

    CstFunc(const CstFunc &cstFunc);



    virtual bool
    isSelectedPnt(const ::dealii::Point<spacedim> &point) override;
    virtual std::unique_ptr<CstSelectorBase<Tria>>
    clone() const override;
  };

} // namespace bcs

#endif /* CstFunc_h */
