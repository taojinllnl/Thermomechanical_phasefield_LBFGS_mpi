//
//  InverseMatrix.h
//  main
//
//

#ifndef InverseMatrix_h
#define InverseMatrix_h


#include <deal.II/lac/solver_cg.h>



template <typename MatrixType, typename PreconditionerType>
class InverseMatrix 
: public dealii::Subscriptor
{
private:
    const dealii::SmartPointer<const MatrixType> matrix;
    const PreconditionerType&            preconditioner;
    
public:
    virtual ~InverseMatrix() = default;
    
    InverseMatrix(const MatrixType&         m,
                  const PreconditionerType& preconditioner);
    
    
    template <typename VectorType>
    void vmult(dealii::SolverControl&    solver_control,
               VectorType&               dst,
               const VectorType&         src) const;
    

};


template <typename MatrixType, typename PreconditionerType>
InverseMatrix<MatrixType, PreconditionerType>
::InverseMatrix(const MatrixType         &m,
                const PreconditionerType &preconditioner)
: matrix(&m)
, preconditioner(preconditioner)
{}



template <typename MatrixType, typename PreconditionerType>
template <typename VectorType>
void InverseMatrix<MatrixType, PreconditionerType>
::vmult(dealii::SolverControl&      solver_control,
        VectorType&                 dst,
        const VectorType&           src) const
{
    using namespace dealii;
    SolverCG<VectorType> cg(solver_control);
    
    dst = 0.;
    
    try
    {
        cg.solve(*matrix, dst, src, preconditioner);
    }
    catch (std::exception &e)
    {
        Assert(false, ExcMessage(e.what()));
    }
}

#endif /* InverseMatrix_h */
