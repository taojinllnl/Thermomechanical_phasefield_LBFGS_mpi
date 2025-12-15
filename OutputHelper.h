//
//  OutputHelper.hpp
//  main
//
//

#ifndef OutputHelper_hpp
#define OutputHelper_hpp

#include "Traits.h"
#include "BlockVectorWrapper.h"
#include "MPIInfo.h"

#include <deal.II/base/quadrature_point_data.h>



#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


namespace PhaseField_monolithic {

template <typename LATraits, typename Tria, typename PointHistory>
class OutputHelper
{
public:
    static constexpr int  dim       = Tria::dimension;
    static constexpr bool is_mpi    =
        !std::is_same_v<typename LATraits::TMTag, ::la::TagSerial>;

    using BVector  = ::la::BlockVectorWrapper<LATraits>;
    
    using CellDataStorage = dealii::CellDataStorage<typename dealii::Triangulation<dim>::cell_iterator, PointHistory>;
    
    using DataComponentInterpretationList = std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>;
private:
    const MPIInfo&                      __mpiInfo;
                
    const Tria&                         __tria;
    
    const dealii::DoFHandler<dim>&      __dof_handler;
    
    const dealii::QGauss<dim>&          __qf_cell;
    
    const std::vector<std::string>      __solution_name;
    
    const DataComponentInterpretationList __data_component_interpretation;
    
private:
    static std::vector<std::string> __makeSolutionName();
    
    static DataComponentInterpretationList __makeDataComponentInterpretation();
    
    
public:
    OutputHelper(const MPIInfo&                   mpiInfo,
                 const Tria&                      tria,
                 const dealii::DoFHandler<dim>&   dof_handler,
                 const dealii::QGauss<dim>&       qf_cell);
    
    
    void output(const unsigned int ithTimeStep,
                const unsigned int polyDegree,
                const std::string& dir,
                const BVector&     solution,
                const CellDataStorage& qPntHistory) const;
};



template <typename LATraits, typename Tria, typename PointHistory>
std::vector<std::string>
OutputHelper<LATraits, Tria, PointHistory>
::__makeSolutionName()
{
    std::vector<std::string> solution_name(dim, "displacement");
    solution_name.emplace_back("phasefield");
    solution_name.emplace_back("temperature");
    return solution_name;
}

template <typename LATraits, typename Tria, typename PointHistory>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
OutputHelper<LATraits, Tria, PointHistory>
::__makeDataComponentInterpretation()
{
    using namespace dealii;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    return data_component_interpretation;
}

template <typename LATraits, typename Tria, typename PointHistory>
OutputHelper<LATraits, Tria, PointHistory>
::OutputHelper(const MPIInfo&                   mpiInfo,
               const Tria&                      tria,
               const dealii::DoFHandler<dim>&   dof_handler,
               const dealii::QGauss<dim>&       qf_cell)
: __mpiInfo(mpiInfo)
, __tria(tria)
, __dof_handler(dof_handler)
, __qf_cell(qf_cell)
, __solution_name(OutputHelper<LATraits, Tria, PointHistory>::__makeSolutionName())
, __data_component_interpretation(OutputHelper<LATraits, Tria, PointHistory>::__makeDataComponentInterpretation())
{}



template <typename LATraits, typename Tria, typename PointHistory>
void OutputHelper<LATraits, Tria, PointHistory>
::output(const unsigned int ithTimeStep,
         const unsigned int polyDegree,
         const std::string& dir,
         const BVector&     solution,
         const CellDataStorage& qPntHistory) const
{
    using namespace dealii;
    
    const std::string filename = "Solution-" + std::to_string(dim) + "d-";
    
    DataOut<dim> data_out;
    data_out.attach_dof_handler(__dof_handler);
    
    
    Vector<double> cell_material_id(__tria.n_active_cells());
    // output material ID for each cell
    for (const auto &cell : __dof_handler.active_cell_iterators())
    {
        if constexpr (is_mpi)
        {
            if(!cell->is_locally_owned())
            {
                continue;
            }
        }
        cell_material_id(cell->active_cell_index()) = cell->material_id();
    }
    data_out.add_data_vector(cell_material_id, "materialID");
    
    
    if constexpr (!is_mpi) {
        data_out.add_data_vector(solution.base(),
                                 __solution_name,
                                 DataOut<dim>::type_dof_data,
                                 __data_component_interpretation);
        
        
        
        //L2 projection
        DoFHandler<dim> dof_handler_L2(__tria);
        FE_Q<dim>     fe_L2(polyDegree); //FE_Q element is continuous
        dof_handler_L2.distribute_dofs(fe_L2);
        AffineConstraints<double> constraints;
        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler_L2, constraints);
        constraints.close();
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation_L2(1,
                                         DataComponentInterpretation::component_is_scalar);
        
        //stress L2 projection
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = i; j < dim; ++j)
            {
                Vector<double> stress_field_L2;
                stress_field_L2.reinit(dof_handler_L2.n_dofs());
                
                MappingQ<dim> mapping(polyDegree + 1);
                VectorTools::project(mapping,
                                     dof_handler_L2,
                                     constraints,
                                     __qf_cell,
                                     [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
                                          const unsigned int q) -> double
                                     {
                    return qPntHistory.get_data(cell)[q]->get_cauchy_stress()[i][j];
                },
                                     stress_field_L2);
                
                std::string stress_name = "Cauchy_stress_" + std::to_string(i+1) + std::to_string(j+1)
                + "_L2";
                
                data_out.add_data_vector(dof_handler_L2,
                                         stress_field_L2,
                                         stress_name,
                                         data_component_interpretation_L2);
            }
        
        // Heat flux L2 projection
        Vector<double> heat_flux_field_L2_x;
        Vector<double> heat_flux_field_L2_y;
        Vector<double> heat_flux_field_L2_z;
        
        for (unsigned int i = 0; i < dim; ++i)
        {
            Vector<double> heat_flux_field_L2;
            heat_flux_field_L2.reinit(dof_handler_L2.n_dofs());
            
            MappingQ<dim> mapping(polyDegree + 1);
            VectorTools::project(mapping,
                                 dof_handler_L2,
                                 constraints,
                                 __qf_cell,
                                 [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
                                      const unsigned int q) -> double
                                 {
                return qPntHistory.get_data(cell)[q]->get_heat_flux()[i];
            },
                                 heat_flux_field_L2);
            
            std::string heat_flux_name = "Heat_flux_" + std::to_string(i+1) + "_L2";
            
            data_out.add_data_vector(dof_handler_L2,
                                     heat_flux_field_L2,
                                     heat_flux_name,
                                     data_component_interpretation_L2);
            if (i == 0)
                heat_flux_field_L2_x = heat_flux_field_L2;
            else if (i == 1)
                heat_flux_field_L2_y = heat_flux_field_L2;
            else if (i == 2)
                heat_flux_field_L2_z = heat_flux_field_L2;
            else
                AssertThrow(false,
                            ExcMessage("Heat flux output is wrong!"));
        }
        
        // For 2D problems, let the flux in the third direction is zero
        if (dim == 2)
        {
            Vector<double> heat_flux_field_L2;
            heat_flux_field_L2.reinit(dof_handler_L2.n_dofs());
            heat_flux_field_L2 = 0;
            std::string heat_flux_name = "Heat_flux_" + std::to_string(3) + "_L2";
            data_out.add_data_vector(dof_handler_L2,
                                     heat_flux_field_L2,
                                     heat_flux_name,
                                     data_component_interpretation_L2);
            heat_flux_field_L2_z = 0;
        }
        
        DoFHandler<dim> dof_handler_L2_flux(__tria);
        FESystem<dim>   fe_flux_L2(FE_Q<dim>(polyDegree), dim);
        dof_handler_L2_flux.distribute_dofs(fe_flux_L2);
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation_flux_L2(dim,
                                              DataComponentInterpretation::component_is_part_of_vector);
        
        Vector<double> heat_flux_field_L2;
        heat_flux_field_L2.reinit(dof_handler_L2_flux.n_dofs());
        
        if (dim == 2)
        {
            for (unsigned int i = 0; i < heat_flux_field_L2_x.size(); ++i)
            {
                heat_flux_field_L2(0+i*dim) = heat_flux_field_L2_x(i);
                heat_flux_field_L2(1+i*dim) = heat_flux_field_L2_y(i);
            }
        }
        
        if (dim == 3)
        {
            for (unsigned int i = 0; i < heat_flux_field_L2_x.size(); ++i)
            {
                heat_flux_field_L2(0+i*dim) = heat_flux_field_L2_x(i);
                heat_flux_field_L2(1+i*dim) = heat_flux_field_L2_y(i);
                heat_flux_field_L2(2+i*dim) = heat_flux_field_L2_z(i);
            }
        }
        
        std::vector<std::string> solution_name_flux(dim, "Heat_flux_vector");
        data_out.add_data_vector(dof_handler_L2_flux,
                                 heat_flux_field_L2,
                                 solution_name_flux,
                                 data_component_interpretation_flux_L2);
        
        data_out.build_patches(polyDegree);
        
        std::ofstream output(dir + filename +
                             Utilities::int_to_string(ithTimeStep, 4) + ".vtu");
        
        data_out.write_vtu(output);
    } else {
        // solution
        data_out.add_data_vector(solution.relevance(),
                                 __solution_name,
                                 DataOut<dim>::type_dof_data,
                                 __data_component_interpretation);
        
        
        // partitioning
        Vector<float> subdomain(__tria.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain(i) = __tria.locally_owned_subdomain();
        data_out.add_data_vector(subdomain, "Partitioning");
        
        
        
        
        data_out.build_patches(polyDegree);
        
//        const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(
//              dir, filename, cycle, mpi_communicator, 4 /*n_digits*/, 0 /*n_groups*/);
//
//        m_logfile << "\t\tVTU file: " << pvtu_filename << std::endl;
        
        data_out.write_vtu_with_pvtu_record(dir,
                                            filename, 
                                            ithTimeStep,
                                            *__mpiInfo.mpiCommPtr(),
                                            4 /*n_digits*/,
                                            0 /*n_groups*/);
    }

}


}
#endif /* OutputHelper_hpp */



