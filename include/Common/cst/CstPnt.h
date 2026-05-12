//
//  CstPnt.h
//  main
//
//

#ifndef CstPnt_h
#define CstPnt_h

#include "CstSelectorBase.h"

namespace bcs
{

  /**
   This class is an interface to describe 2 kinds of boundary conditions:
   1. known coordinates for a node and known constrained values.
   2. known coordinates for a node and function-based values.
   This class is derived from CstSelectorBase<Tria>.


   For example:

   A system has 5 primary known at nodes: (u_x, u_y, u_z, T, d).
   The ith dofs at node:                  [0,   1,   2,   3, 4].

   2 nodes, [120.0, 105.5, -32.5] and [-25.0, 15.0, 12.5], have 3 constraints:
        - fixed displacement along y,
        - constantly constrained displacement 1.5 long z
        - constantly constrained value = 20.0 for T
   These three constraints can be described as:

   ```cpp
   // u_y = 0
   const CstEntry<Tria>  u_y(1); // or const CstEntry<Tria>  fixeX(1, 0.0);
   // u_z = 1.5
   const CstEntry<Tria>  u_z(2, 1.5);
   // T = 20.0
   const CstEntry<Tria>  T(3, 20.0);
   ```

   Once the CstEntry<Tria> have been defined, the boundary conditions can be
   added to CstPnt<Tria> to create a point constraint for the CstMaker<Tria>.

   ```cpp
   CstPnt<Tria> cstPnt1({{120.0, 105.5, -32.5}},
                        {{u_y, y_z, T}});
   CstPnt<Tria> cstPnt2({{-25.0, 15.0, 12.5}},
                        {{u_y, y_z, T}});
   ```

   Also, if there is a third node, [0.0, 5.5, -2.5], constrained:
        - fixed displacement along y

   The input should be defined as:

   ```cpp
   CstPnt<Tria> cstPnt3({{0.0, 5.5, -2.5}}, u_y); // reuse defined
   CstEntry<Tria>

   ```

   The CstEntry<Tria> can be reused to define different CstPnt<Tria>. It's
   worth being emphasized that the CstSelectorBase<Tria>, the base class of
   CstPnt<Tria> will copy CstEntry<Tria> and make its own CstEntry<Tria> as the
   templated input for the future operations. Once the CstEntry<Tria> is given
   to CstPnt<Tria>, the object held by CstPnt<Tria> and the one passed are
   totally different.

   Assume a scenario, the constrained value will be changed from fixed along y
   to move 100 along y after a specific timestep. The following operations are
   useless for the objective:

   ```cpp
   CstEntry<Tria>  u_y(1);
   u_y.setValue(10.0); // this operation valid to cstPnt3.
   CstPnt<Tria> cstPnt3({{0.0, 5.5, -2.5}}, u_y);

   u_y.setValue(100.0); // this operation is useless to cstPnt3.
   ```

   If the constrained value will be changed sometimes. A lambda expression can
   be passed during constructing CstPnt<Tria>, which is defined as:

   std::function<void( const double,
                       const double,
                       const double,
                       std::vector<double> &)>;

   The first three variables represent the coordinates (x, y, z) of a node. No
   matter in the one-, two- or three-dimensional problems, these three arguments
   will be also be passed to unify the interface. Only the valid coordinates is
   real coordinates. and the others are zero. (1D: x is valid; 2D: x and y are
   valid; 3D: x, y, and z are valid, the mesh should be built correspondingly.)
   The last varibable is the results of the constrained values to be passed to
   the cache in the CstMaker<Tria>. This std::vector<double> has n slots whose
   size equals to the number of dofs at this point. Only the values in the
   constrained slots (described by CstEntry<Tria>) will be extracted and applied
   as the constraints. The other numbers passed into this vector will be
   ignored. Furthermore, this vector has already been initialized, and its size
   is consistant to the number of dof on a node. If the size of the std::vector
   is changed in the lambda expression, an expection will be thrown.

   For example, displacement on a point [0, 10, 15] is fixed along x direction
   from time step 0 to 7. After time step 7, the constrained horizontal
   displacement will be changed to 1.5.

   ```cpp
   unsigned int timestep; // assume this variable is used to define timestep

   // other operations .......

   // only x direction is constrained and the constained value is not important.
   const CstEntry<Tria>  u_x(0);
   // the lambda expression needs to capture timestep to obtain its value.
   // It works for other type of objects.
   const auto u_x_func[timestep](const double x,
                                 const double y,
                                 const double z,
                                 std::vector<double> &values)
   {
        if(timestep <= 7)
        {
            values[0] = 0;
            values[1] = 1; // invalid result: this value will be ignored,
                           // because only u_x is defined in CstEntry<Tria>
                           // and passed to CstPnt<Tria>.
        } else {
            values[0] = 1.5; // the value for horizontal displacement will be
                             // changed to 1.5.
        }
   };

   CstPnt<Tria> cstPnt3({{0, 10, 15}}, // coordinates
                        u_x,           // constrainted dof
                        u_x_func);     // expression for constrained value
   ```
   */
  template <typename Tria>
  class CstPnt : public CstSelectorBase<Tria>
  {
  public:
    static const int dim      = Tria::dimension;
    static const int spacedim = Tria::space_dimension;

    using ValuesAtPntFunc = std::function<
      void(const double, const double, const double, std::vector<double> &)>;

  private:
    std::array<double, 3>            pntCoorinates{};
    std::shared_ptr<ValuesAtPntFunc> valuesFunc{};

  public:
    const double tol;

    virtual ~CstPnt() = default;

    CstPnt(const std::array<double, 3>       &point,
           const std::vector<CstEntry<Tria>> &csts,
           const double                       tol = 1e-9);

    CstPnt(const std::array<double, 3> &point,
           const CstEntry<Tria>        &csts,
           const double                 tol = 1e-9);



    CstPnt(const std::array<double, 3>       &point,
           const std::vector<CstEntry<Tria>> &csts,
           const ValuesAtPntFunc             &valuesFunc,
           const double                       tol = 1e-9);

    CstPnt(const std::array<double, 3> &point,
           const CstEntry<Tria>        &csts,
           const ValuesAtPntFunc       &valuesFunc,
           const double                 tol = 1e-9);


    CstPnt(const CstPnt &cstPnt);


    virtual std::size_t
    expectedNumberOfCstPoints() const override;
    virtual std::size_t
    expectedNumberOfCstEntries() const override;

    virtual void
    assignValues(const double         x,
                 const double         y,
                 const double         z,
                 std::vector<double> &values) override;

    virtual bool
    isSelectedPnt(const ::dealii::Point<spacedim> &point) override;

    virtual std::unique_ptr<CstSelectorBase<Tria>>
    clone() const override;
  };
} // namespace bcs

#endif /* CstPnt_h */
