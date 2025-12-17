/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Tao Jin
 *         University of Ottawa, Ottawa, Ontario, Canada
 *         April. 2025
 *
 * How to cite:
 *         TBD
 */

/* A fully monolithic scheme based on the L-BFGS method to solve the phase-field
 * thermomechanically coupled crack problem:
 * 1. The phase-field formulation itself is based on "A phase field model for rate-independent
 *    crack propagation - Robust algorithmic implementation based on operator splits"
 *    by Christian Miehe , Martina Hofacker, Fabian Welschinger.
 * 2. The thermal conductivity tensor is isotropic and degraded by the phase-field.
 * 3. The thermal equation is transient and considers the temperature
 *    changing with time (T_dot). The backward Euler time integrator is used.
 * 4. The mechanical problem is quasi-static and does not consider the inertial effort
 *    (no acceleration term).
 * 5. This code implements a monolithic approach. The phase-field irreversibility
 *    is enforced through the history field Phi_0^+.
 * 6. Using TBB for stiffness assembly and Gauss point calculation.
 * 7. Using adaptive mesh refinement.
 * 8. The gradient-based line search method is used.
 * 9. The displacement field, phase-field, and the temperature field are solved
 *    simultaneously during each iteration.
 *10. The limited-memory BFGS method is used. See the reference:
 *    Jin T, Li Z, Chen K. A novel phase-field monolithic scheme for brittle crack
 *    propagation based on the limited-memory BFGS method with adaptive mesh refinement.
 *    Int J Numer Methods Eng. 2024;e7572. doi: 10.1002/nme.7572.
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/mapping_q_eulerian.h>


#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>


#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal.II/physics/elasticity/standard_tensors.h>

#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/base/work_stream.h>

#include <deal.II/numerics/solution_transfer.h>



#include <vector>
#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>




#include "SpectrumDecomposition.h"
#include "Utilities.h"
#include "FileSystem.h"

#include "MPIInfo.h"
#include "Logger.h"
#include "TimerOutputWrapper.h"


#include "BlockVectorWrapper.h"
#include "BlockSparseMatrixWrapper.h"
#include "BlockDesc.h"

#include "LASolver.h"

#include "OutputHelper.h"


//template <int dim, int spacedim = dim>
//using RTria = ::dealii::Triangulation<dim, spacedim>;
//
//template <int dim, int spacedim = dim>
//using DTria = ::dealii::parallel::distributed::Triangulation<dim, spacedim>;



namespace PhaseField_monolithic
{
  using namespace dealii;

  // body force
  template <int dim>
  void right_hand_side(const std::vector<Point<dim>> &points,
		       std::vector<Tensor<1, dim>> &  values,
		       const double fx,
		       const double fy,
		       const double fz)
  {
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));
    Assert(dim >= 2, ExcNotImplemented());

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
	if (dim == 2)
	  {
	    values[point_n][0] = fx;
	    values[point_n][1] = fy;
	  }
	else
	  {
	    values[point_n][0] = fx;
	    values[point_n][1] = fy;
	    values[point_n][2] = fz;
	  }
      }
  }

  // heat supply
  template <int dim>
  void heat_supply(const std::vector<Point<dim>> &points,
		   std::vector<double> &  values,
		   const double heat_supply)
  {
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));
    Assert(dim >= 2, ExcNotImplemented());

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
	values[point_n] = heat_supply;
      }
  }

  double degradation_function(const double d)
  {
    return (1.0 - d) * (1.0 - d);
  }

  double degradation_function_derivative(const double d)
  {
    return 2.0 * (d - 1.0);
  }

  double degradation_function_2nd_order_derivative(const double d)
  {
    (void) d;
    return 2.0;
  }

  namespace Parameters
  {
    struct Scenario
    {
      unsigned int m_dim;
      unsigned int m_scenario;
      std::string m_logfile_name;
      bool m_output_iteration_history;
      bool m_coupling_on_heat_eq;
      bool m_degrade_conductivity;
      std::string m_type_nonlinear_solver;
      std::string m_type_linear_solver;
      double m_cg_u_tol;
      double m_cg_d_tol;
      double m_cg_t_tol;
      std::string m_refinement_strategy;
      unsigned int m_LBFGS_m;
      unsigned int m_global_refine_times;
      unsigned int m_local_prerefine_times;
      unsigned int m_max_adaptive_refine_times;
      int m_max_allowed_refinement_level;
      double m_phasefield_refine_threshold;
      double m_allowed_max_h_l_ratio;
      unsigned int m_total_material_regions;
      std::string m_material_file_name;
      int m_reaction_force_face_id;
        
      std::string m_mpi_type;
      
      std::string m_config_dir;
      std::string m_output_dir;
        
      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm);
    };

    void Scenario::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        prm.declare_entry("dimension",
                            "2",
                            Patterns::Integer(2),
                            "dimension of the problem");
          
        prm.declare_entry("Scenario number",
                          "1",
                          Patterns::Integer(0),
                          "Geometry, loading and boundary conditions scenario");

        prm.declare_entry("Log file name",
			  "Output.log",
                          Patterns::FileName(Patterns::FileName::input),
			  "Name of the file for log");

        prm.declare_entry("Output iteration history",
			  "yes",
                          Patterns::Selection("yes|no"),
			  "Shall we write iteration history to the log file?");

        prm.declare_entry("Coupling on heat equation",
			  "no",
                          Patterns::Selection("yes|no"),
			  "Does the heat equation contain the coupling term?");

        prm.declare_entry("Degrade thermal conductivity",
			  "yes",
                          Patterns::Selection("yes|no"),
			  "Degrade thermal conductivity or not?");

        prm.declare_entry("Nonlinear solver type",
                          "LBFGS",
                          Patterns::Selection("LBFGS"),
                          "Type of solver used to solve the nonlinear system");

        prm.declare_entry("Linear solver type",
                          "Direct",
                          Patterns::Selection("Direct|CG"),
                          "Type of solver used to solve the linear system B0");

        prm.declare_entry("CG u tolerance",
			  "1.0e-9",
			  Patterns::Double(0.0),
			  "If CG is selected as linear solver, the tolerance of CG for inverse K_uu");

        prm.declare_entry("CG d tolerance",
			  "1.0e-9",
			  Patterns::Double(0.0),
			  "If CG is selected as linear solver, the tolerance of CG for inverse K_dd");

        prm.declare_entry("CG T tolerance",
			  "1.0e-9",
			  Patterns::Double(0.0),
			  "If CG is selected as linear solver, the tolerance of CG for inverse K_TT");

        prm.declare_entry("Mesh refinement strategy",
                          "adaptive-refine",
                          Patterns::Selection("pre-refine|adaptive-refine"),
                          "Mesh refinement strategy: pre-refine or adaptive-refine");

        prm.declare_entry("LBFGS m",
                          "40",
                          Patterns::Integer(0),
                          "Number of vectors used for LBFGS");

        prm.declare_entry("Global refinement times",
                          "0",
                          Patterns::Integer(0),
                          "Global refinement times (across the entire domain)");

        prm.declare_entry("Local prerefinement times",
                          "0",
                          Patterns::Integer(0),
                          "Local pre-refinement times (assume crack path is known a priori), "
                          "only refine along the crack path.");

        prm.declare_entry("Max adaptive refinement times",
                          "100",
                          Patterns::Integer(0),
                          "Maximum number of adaptive refinement times allowed in each step");

        prm.declare_entry("Max allowed refinement level",
                          "100",
                          Patterns::Integer(0),
                          "Maximum allowed cell refinement level");

        prm.declare_entry("Phasefield refine threshold",
			  "0.8",
			  Patterns::Double(),
			  "Phasefield-based refinement threshold value");

        prm.declare_entry("Allowed max hl ratio",
			  "0.25",
			  Patterns::Double(),
			  "Allowed maximum ratio between mesh size h and length scale l");

        prm.declare_entry("Material regions",
                          "1",
                          Patterns::Integer(0),
                          "Number of material regions");

        prm.declare_entry("Material data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Material data file");

        prm.declare_entry("Reaction force face ID",
                          "1",
                          Patterns::Integer(),
                          "Face id where reaction forces should be calculated "
                          "(negative integer means not to calculate reaction force)");
        
          
        prm.declare_entry("mpi type",
                          "PETSc",
                            Patterns::Selection("PETSc|Trilinos|Serial"),
                            "underlying mpi type");
            
          
        prm.declare_entry("Config dir",
                            "./",
                            Patterns::FileName(Patterns::FileName::input),
                            "Configuration directory");
          
        prm.declare_entry("Output dir",
                           "./",
                              Patterns::FileName(Patterns::FileName::input),
                              "Output directory");
          
      }
      prm.leave_subsection();
    }

    void Scenario::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Scenario");
      {
        m_dim  = prm.get_integer("dimension");
        m_scenario = prm.get_integer("Scenario number");
        m_logfile_name = prm.get("Log file name");
        m_output_iteration_history = prm.get_bool("Output iteration history");
        m_coupling_on_heat_eq = prm.get_bool("Coupling on heat equation");
        m_degrade_conductivity = prm.get_bool("Degrade thermal conductivity");
        m_type_nonlinear_solver = prm.get("Nonlinear solver type");
        m_type_linear_solver = prm.get("Linear solver type");
        m_cg_u_tol = prm.get_double("CG u tolerance");
        m_cg_d_tol = prm.get_double("CG d tolerance");
        m_cg_t_tol = prm.get_double("CG T tolerance");
        m_refinement_strategy = prm.get("Mesh refinement strategy");
        m_LBFGS_m = prm.get_integer("LBFGS m");
        m_global_refine_times = prm.get_integer("Global refinement times");
        m_local_prerefine_times = prm.get_integer("Local prerefinement times");
        m_max_adaptive_refine_times = prm.get_integer("Max adaptive refinement times");
        m_max_allowed_refinement_level = prm.get_integer("Max allowed refinement level");
        m_phasefield_refine_threshold = prm.get_double("Phasefield refine threshold");
        m_allowed_max_h_l_ratio = prm.get_double("Allowed max hl ratio");
        m_total_material_regions = prm.get_integer("Material regions");
        m_material_file_name = prm.get("Material data file");
        m_reaction_force_face_id = prm.get_integer("Reaction force face ID");
          
        m_mpi_type = prm.get("mpi type");
          
        m_config_dir = prm.get("Config dir");
        m_output_dir = prm.get("Output dir");
      }
      prm.leave_subsection();
    }

    struct FESystem
    {
      unsigned int m_poly_degree;
      unsigned int m_quad_order;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };


    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree",
                          "1",
                          Patterns::Integer(0),
                          "Phase field polynomial order");

        prm.declare_entry("Quadrature order",
                          "2",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        m_poly_degree = prm.get_integer("Polynomial degree");
        m_quad_order  = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

    // body force (N/m^3)
    struct BodyForce
    {
      double m_x_component;
      double m_y_component;
      double m_z_component;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void BodyForce::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Body force");
      {
        prm.declare_entry("Body force x component",
			  "0.0",
			  Patterns::Double(),
			  "Body force x-component (N/m^3)");

        prm.declare_entry("Body force y component",
			  "0.0",
			  Patterns::Double(),
			  "Body force y-component (N/m^3)");

        prm.declare_entry("Body force z component",
			  "0.0",
			  Patterns::Double(),
			  "Body force z-component (N/m^3)");
      }
      prm.leave_subsection();
    }

    void BodyForce::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Body force");
      {
        m_x_component = prm.get_double("Body force x component");
        m_y_component = prm.get_double("Body force y component");
        m_z_component = prm.get_double("Body force z component");
      }
      prm.leave_subsection();
    }


    // heat supply (Watt/m^3)
    struct HeatSupply
    {
      double m_heat_supply;
      double m_ref_temperature;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void HeatSupply::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Heat supply");
      {
        prm.declare_entry("Heat supply",
			  "0.0",
			  Patterns::Double(),
			  "Heat supply (Watt/m^3)");

        prm.declare_entry("Reference temperature",
			  "300.0",
			  Patterns::Double(),
			  "Reference temperature (K)");
      }
      prm.leave_subsection();
    }

    void HeatSupply::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Heat supply");
      {
        m_heat_supply = prm.get_double("Heat supply");
        m_ref_temperature = prm.get_double("Reference temperature");
      }
      prm.leave_subsection();
    }

    struct NonlinearSolver
    {
      unsigned int m_max_iterations_LBFGS;
      bool m_relative_residual;

      double       m_tol_u_residual;
      double       m_tol_d_residual;
      double       m_tol_t_residual;

      double       m_tol_u_incr;
      double       m_tol_d_incr;
      double       m_tol_t_incr;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations LBFGS",
                          "20",
                          Patterns::Integer(0),
                          "Number of LBFGS iterations allowed");

        prm.declare_entry("Relative residual",
			  "yes",
                          Patterns::Selection("yes|no"),
			  "Shall we use relative residual for convergence?");

        prm.declare_entry("Tolerance displacement residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Displacement residual tolerance");

        prm.declare_entry("Tolerance phasefield residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Phasefield residual tolerance");

        prm.declare_entry("Tolerance temperature residual",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Temperature residual tolerance");

        prm.declare_entry("Tolerance displacement increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Displacement increment tolerance");

        prm.declare_entry("Tolerance phasefield increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Phasefield increment tolerance");

        prm.declare_entry("Tolerance temperature increment",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Temperature increment tolerance");
      }
      prm.leave_subsection();
    }

    void NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        m_max_iterations_LBFGS = prm.get_integer("Max iterations LBFGS");
        m_relative_residual = prm.get_bool("Relative residual");

        m_tol_u_residual           = prm.get_double("Tolerance displacement residual");
        m_tol_d_residual           = prm.get_double("Tolerance phasefield residual");
        m_tol_t_residual           = prm.get_double("Tolerance temperature residual");

        m_tol_u_incr               = prm.get_double("Tolerance displacement increment");
        m_tol_d_incr               = prm.get_double("Tolerance phasefield increment");
        m_tol_t_incr               = prm.get_double("Tolerance temperature increment");
      }
      prm.leave_subsection();
    }

    struct TimeInfo
    {
      double m_end_time;
      std::string m_time_file_name;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void TimeInfo::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");

        prm.declare_entry("Time data file",
                          "1",
                          Patterns::FileName(Patterns::FileName::input),
                          "Time data file");
      }
      prm.leave_subsection();
    }

    void TimeInfo::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        m_end_time = prm.get_double("End time");
        m_time_file_name = prm.get("Time data file");
      }
      prm.leave_subsection();
    }

    struct AllParameters : public Scenario,
	                   public FESystem,
	                   public BodyForce,
			   public HeatSupply,
			   public NonlinearSolver,
			   public TimeInfo
    {
      AllParameters(const std::string &input_file);

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
        
        std::string subDir;
        std::string histDir;
        std::string oriDir;
        std::string resultsDir;
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      Scenario::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      BodyForce::declare_parameters(prm);
      HeatSupply::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      TimeInfo::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      Scenario::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      BodyForce::parse_parameters(prm);
      HeatSupply::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      TimeInfo::parse_parameters(prm);
    }
  } // namespace Parameters

  class Time
  {
  public:
    Time(const double time_end)
      : m_timestep(0)
      , m_time_current(0.0)
      , m_time_end(time_end)
      , m_delta_t(0.0)
      , m_magnitude(1.0)
    {}

    virtual ~Time() = default;

    double current() const
    {
      return m_time_current;
    }
    double end() const
    {
      return m_time_end;
    }
    double get_delta_t() const
    {
      return m_delta_t;
    }
    double get_magnitude() const
    {
      return m_magnitude;
    }
    unsigned int get_timestep() const
    {
      return m_timestep;
    }
    void increment(std::vector<std::array<double, 4>> time_table)
    {
      double t_1, t_delta, t_magnitude;
      for (auto & time_group : time_table)
        {
	  t_1 = time_group[1];
	  t_delta = time_group[2];
	  t_magnitude = time_group[3];

	  if (m_time_current < t_1 - 1.0e-6*t_delta)
	    {
	      m_delta_t = t_delta;
	      m_magnitude = t_magnitude;
	      break;
	    }
        }

      m_time_current += m_delta_t;
      ++m_timestep;
    }

  private:
    unsigned int m_timestep;
    double       m_time_current;
    const double m_time_end;
    double m_delta_t;
    double m_magnitude;
  };

  template <int dim>
  class LinearIsotropicElasticityAdditiveSplit
  {
  public:
    LinearIsotropicElasticityAdditiveSplit(const double lame_lambda,
			                   const double lame_mu,
				           const double residual_k,
					   const double length_scale,
					   const double viscosity,
					   const double gc_0,
					   const double heat_capacity,
					   const double thermal_conductivity_0,
					   const double thermal_expansion_coeff,
					   const double reference_temperature,
					   const double max_temperature,
					   const double b_1,
					   const double b_2)
      : m_lame_lambda(lame_lambda)
      , m_lame_mu(lame_mu)
      , m_residual_k(residual_k)
      , m_length_scale(length_scale)
      , m_eta(viscosity)
      , m_gc_0(gc_0)
      , m_heat_capacity(heat_capacity)
      , m_kappa_0(thermal_conductivity_0)
      , m_alpha(thermal_expansion_coeff)
      , m_ref_t(reference_temperature)
      , m_max_t(max_temperature)
      , m_b_1(b_1)
      , m_b_2(b_2)
      , m_phase_field_value(0.0)
      , m_grad_phasefield(Tensor<1, dim>())
      , m_strain(SymmetricTensor<2, dim>())
      , m_stress(SymmetricTensor<2, dim>())
      , m_stress_positive(SymmetricTensor<2, dim>())
      , m_mechanical_C(SymmetricTensor<4, dim>())
      , m_strain_energy_positive(0.0)
      , m_strain_energy_negative(0.0)
      , m_strain_energy_total(0.0)
      , m_crack_energy_dissipation(0.0)
      , m_gc_t(0.0)
      , m_kappa_d(0.0)
      , m_temperature(0.0)
      , m_grad_temperature(Tensor<1, dim>())
      , m_heat_flux(Tensor<1, dim>())
    {
      Assert(  ( lame_lambda / (2*(lame_lambda + lame_mu)) <= 0.5)
	     & ( lame_lambda / (2*(lame_lambda + lame_mu)) >=-1.0),
	     ExcInternalError() );
    }

    const SymmetricTensor<4, dim> & get_mechanical_C() const
    {
      return m_mechanical_C;
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress() const
    {
      return m_stress;
    }

    const SymmetricTensor<2, dim> & get_strain() const
    {
      return m_strain;
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress_positive() const
    {
      return m_stress_positive;
    }

    double get_positive_strain_energy() const
    {
      return m_strain_energy_positive;
    }

    double get_negative_strain_energy() const
    {
      return m_strain_energy_negative;
    }

    double get_total_strain_energy() const
    {
      return m_strain_energy_total;
    }

    double get_crack_energy_dissipation() const
    {
      return m_crack_energy_dissipation;
    }

    double get_phase_field_value() const
    {
      return m_phase_field_value;
    }

    double get_thermal_expansion_coeff() const
    {
      return m_alpha;
    }

    double get_lame_lambda() const
    {
      return m_lame_lambda;
    }

    double get_lame_mu() const
    {
      return m_lame_mu;
    }

    const Tensor<1, dim> get_phase_field_gradient() const
    {
      return m_grad_phasefield;
    }

    double get_temperature_value() const
    {
      return m_temperature;
    }

    double get_ref_temperature() const
    {
      return m_ref_t;
    }

    const Tensor<1, dim> get_temperature_gradient() const
    {
      return m_grad_temperature;
    }

    const Tensor<1, dim> get_heat_flux() const
    {
      return m_heat_flux;
    }

    // temperature-dependent critical energy release rate
    double get_critical_energy_release_rate_temperature() const
    {
      return m_gc_t;
    }

    double get_thermal_conductivity_degraded() const
    {
      return m_kappa_d;
    }

    void update_material_data(const SymmetricTensor<2, dim> & strain,
			      const double phase_field_value,
			      const Tensor<1, dim> & grad_phasefield,
			      const double phase_field_value_previous_step,
			      const double delta_time,
			      const double temperature,
			      const Tensor<1, dim> & grad_temperature,
			      const bool degrade_conductivity_or_not)
    {
      // Total strain grad^{(s)}u
      m_strain = strain;
      m_phase_field_value = phase_field_value;
      m_grad_phasefield = grad_phasefield;
      m_temperature = temperature;
      m_grad_temperature = grad_temperature;

      // Thermal strain
      SymmetricTensor<2, dim> strain_t;
      strain_t = m_alpha * (temperature - m_ref_t)
	                 * Physics::Elasticity::StandardTensors<dim>::I;

      // Effective strain
      SymmetricTensor<2, dim> strain_e;
      strain_e = m_strain - strain_t;

      // temperature-dependent gc
      double term_1 = (temperature - m_ref_t) / m_max_t;
      double coeff = 1.0 - m_b_1 * term_1
	                 + m_b_2 * term_1 * term_1;
      m_gc_t = coeff * m_gc_0;

      Vector<double>              eigenvalues(dim);
      std::vector<Tensor<1, dim>> eigenvectors(dim);
      usr_spectrum_decomposition::spectrum_decomposition<dim>(strain_e,
    							      eigenvalues,
    							      eigenvectors);

      SymmetricTensor<2, dim> strain_positive, strain_negative;
      strain_positive = usr_spectrum_decomposition::positive_tensor(eigenvalues, eigenvectors);
      strain_negative = usr_spectrum_decomposition::negative_tensor(eigenvalues, eigenvectors);

      SymmetricTensor<4, dim> projector_positive, projector_negative;
      usr_spectrum_decomposition::positive_negative_projectors(eigenvalues,
    							       eigenvectors,
							       projector_positive,
							       projector_negative);

      SymmetricTensor<2, dim> stress_positive, stress_negative;
      const double degradation = degradation_function(m_phase_field_value);
      const double I_1 = trace(strain_e);
      stress_positive = m_lame_lambda * usr_spectrum_decomposition::positive_ramp_function(I_1)
                                      * Physics::Elasticity::StandardTensors<dim>::I
                      + 2 * m_lame_mu * strain_positive;
      stress_negative = m_lame_lambda * usr_spectrum_decomposition::negative_ramp_function(I_1)
                                      * Physics::Elasticity::StandardTensors<dim>::I
      		      + 2 * m_lame_mu * strain_negative;

      m_stress = degradation * stress_positive + stress_negative;
      m_stress_positive = stress_positive;

      SymmetricTensor<4, dim> C_positive, C_negative;
      C_positive = m_lame_lambda * usr_spectrum_decomposition::heaviside_function(I_1)
                                 * Physics::Elasticity::StandardTensors<dim>::IxI
		 + 2 * m_lame_mu * projector_positive;
      C_negative = m_lame_lambda * usr_spectrum_decomposition::heaviside_function(-I_1)
                                 * Physics::Elasticity::StandardTensors<dim>::IxI
      		 + 2 * m_lame_mu * projector_negative;
      m_mechanical_C = degradation * C_positive + C_negative;

      m_strain_energy_positive = 0.5 * m_lame_lambda * usr_spectrum_decomposition::positive_ramp_function(I_1)
                                                     * usr_spectrum_decomposition::positive_ramp_function(I_1)
                               + m_lame_mu * strain_positive * strain_positive;

      m_strain_energy_negative = 0.5 * m_lame_lambda * usr_spectrum_decomposition::negative_ramp_function(I_1)
                                                     * usr_spectrum_decomposition::negative_ramp_function(I_1)
                               + m_lame_mu * strain_negative * strain_negative;

      m_strain_energy_total = degradation * m_strain_energy_positive + m_strain_energy_negative;

      // The critical energy release rate m_gc should be temperature-dependent.
      m_crack_energy_dissipation = m_gc_t * (  0.5 / m_length_scale * m_phase_field_value * m_phase_field_value
	                                   + 0.5 * m_length_scale * m_grad_phasefield * m_grad_phasefield)
	                                   // the term due to viscosity regularization
	                                   + (m_phase_field_value - phase_field_value_previous_step)
					   * (m_phase_field_value - phase_field_value_previous_step)
				           * 0.5 * m_eta / delta_time;

      // degraded thermal conductivity
      if (degrade_conductivity_or_not)
	m_kappa_d = (degradation + m_residual_k) * m_kappa_0;
      else
	m_kappa_d = 1.0 * m_kappa_0;

      // heat flux
      m_heat_flux = - m_kappa_d * grad_temperature;

      //(void)delta_time;
      //(void)phase_field_value_previous_step;
    }

  private:
    const double m_lame_lambda;
    const double m_lame_mu;
    const double m_residual_k;
    const double m_length_scale;
    const double m_eta;
    const double m_gc_0;
    const double m_heat_capacity;
    const double m_kappa_0;
    const double m_alpha;
    const double m_ref_t;
    const double m_max_t;
    const double m_b_1;
    const double m_b_2;
    double m_phase_field_value;
    Tensor<1, dim> m_grad_phasefield;
    SymmetricTensor<2, dim> m_strain;
    SymmetricTensor<2, dim> m_stress;
    SymmetricTensor<2, dim> m_stress_positive;
    SymmetricTensor<4, dim> m_mechanical_C;
    double m_strain_energy_positive;
    double m_strain_energy_negative;
    double m_strain_energy_total;
    double m_crack_energy_dissipation;
    // temperature-dependent critical energy release rate
    double m_gc_t;
    // degraded thermal conductivity
    double m_kappa_d;
    double m_temperature;
    Tensor<1, dim> m_grad_temperature;
    Tensor<1, dim> m_heat_flux;
  };


  template <int dim>
  class PointHistory
  {
  public:
    PointHistory()
      : m_length_scale(0.0)
      , m_viscosity(0.0)
      , m_history_max_positive_strain_energy(0.0)
      , m_heat_capacity(0.0)
      , m_coupling_on_heat_eq(false)
    {}

    virtual ~PointHistory() = default;

    void setup_lqp(const double lame_lambda,
		   const double lame_mu,
		   const double length_scale,
		   const double gc_0,
		   const double viscosity,
		   const double residual_k,
		   const double heat_capacity,
		   const double thermal_conductivity_0,
		   const double thermal_expansion_coeff,
		   const double reference_temperature,
		   const double max_temperature,
		   const double b_1,
		   const double b_2,
		   const bool   coupling_on_heat_eq)
    {
      m_material =
              std::make_shared<LinearIsotropicElasticityAdditiveSplit<dim>>(lame_lambda,
        	                                                            lame_mu,
								            residual_k,
									    length_scale,
									    viscosity,
									    gc_0,
									    heat_capacity,
									    thermal_conductivity_0,
									    thermal_expansion_coeff,
									    reference_temperature,
									    max_temperature,
									    b_1,
									    b_2);
      m_history_max_positive_strain_energy = 0.0;
      m_length_scale = length_scale;
      m_viscosity = viscosity;
      m_heat_capacity = heat_capacity;
      m_coupling_on_heat_eq = coupling_on_heat_eq;

      update_field_values(SymmetricTensor<2, dim>(), 0.0, Tensor<1, dim>(),
			  0.0, 1.0, reference_temperature, Tensor<1, dim>(), true);
    }

    void update_field_values(const SymmetricTensor<2, dim> & strain,
		             const double phase_field_value,
			     const Tensor<1, dim> & grad_phasefield,
			     const double phase_field_value_previous_step,
			     const double delta_time,
			     const double temperature,
			     const Tensor<1, dim> & grad_temperature,
			     const bool degrade_conductivity_or_not)
    {
      m_material->update_material_data(strain, phase_field_value, grad_phasefield,
				       phase_field_value_previous_step, delta_time,
				       temperature, grad_temperature,
				       degrade_conductivity_or_not);
    }

    void update_history_variable()
    {
      double current_positive_strain_energy = m_material->get_positive_strain_energy();
      m_history_max_positive_strain_energy = std::fmax(m_history_max_positive_strain_energy,
					               current_positive_strain_energy);
    }

    // This is the function used to assign the history variable after remeshing
    void assign_history_variable(double history_variable_value)
    {
      m_history_max_positive_strain_energy = history_variable_value;
    }

    double get_current_positive_strain_energy() const
    {
      return m_material->get_positive_strain_energy();
    }

    const SymmetricTensor<4, dim> & get_mechanical_C() const
    {
      return m_material->get_mechanical_C();
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress() const
    {
      return m_material->get_cauchy_stress();
    }

    const SymmetricTensor<2, dim> & get_strain() const
    {
      return m_material->get_strain();
    }

    const SymmetricTensor<2, dim> & get_cauchy_stress_positive() const
    {
      return m_material->get_cauchy_stress_positive();
    }

    double get_total_strain_energy() const
    {
      return m_material->get_total_strain_energy();
    }

    double get_crack_energy_dissipation() const
    {
      return m_material->get_crack_energy_dissipation();
    }

    double get_phase_field_value() const
    {
      return m_material->get_phase_field_value();
    }

    double get_temperature_value() const
    {
      return m_material->get_temperature_value();
    }

    double get_ref_temperature() const
    {
      return m_material->get_ref_temperature();
    }

    const Tensor<1, dim> get_phase_field_gradient() const
    {
      return m_material->get_phase_field_gradient();
    }

    const Tensor<1, dim> get_temperature_gradient() const
    {
      return m_material->get_temperature_gradient();
    }

    const Tensor<1, dim> get_heat_flux() const
    {
      return m_material->get_heat_flux();
    }

    // return the temperature-dependent gc
    double get_critical_energy_release_rate() const
    {
      return m_material->get_critical_energy_release_rate_temperature();
    }

    double get_thermal_conductivity() const
    {
      return m_material->get_thermal_conductivity_degraded();
    }

    double get_history_max_positive_strain_energy() const
    {
      return m_history_max_positive_strain_energy;
    }

    double get_length_scale() const
    {
      return m_length_scale;
    }

    double get_viscosity() const
    {
      return m_viscosity;
    }

    double get_heat_capacity() const
    {
      return m_heat_capacity;
    }

    bool get_heat_coupling_flag() const
    {
      return m_coupling_on_heat_eq;
    }

    double get_thermal_expansion_coeff() const
    {
      return m_material->get_thermal_expansion_coeff();
    }

    double get_lame_lambda() const
    {
      return m_material->get_lame_lambda();
    }

    double get_lame_mu() const
    {
      return m_material->get_lame_mu();
    }

  private:
    std::shared_ptr<LinearIsotropicElasticityAdditiveSplit<dim>> m_material;
    double m_length_scale;
    double m_viscosity;
    double m_history_max_positive_strain_energy;
    double m_heat_capacity;
    bool   m_coupling_on_heat_eq;
  };


 

  template <typename LATraits, typename Tria>
  class PhaseFieldMonolithicSolve
  {
  public:
      constexpr static int dim = Tria::dimension;
      using BSMatrix = ::la::BlockSparseMatrixWrapper<LATraits>;
      using BVector  = ::la::BlockVectorWrapper<LATraits>;
      
      using CellDataStorage = CellDataStorage<typename Tria::cell_iterator,
      PointHistory<dim>>;
      
      static constexpr bool is_mpi =
          !std::is_same_v<typename LATraits::TMTag, ::la::TagSerial>;
//            using BSMatrix = la::BlockSparseMatrixWrapper<la::Traits<la::TagSerial>>;
//            using BVector  = la::BlockVectorWrapper<la::Traits<la::TagSerial>>;
      
//      using BSMatrix = BlockSparseMatrix<double>;
//      using BVector  = BlockVector<double>;
      
      PhaseFieldMonolithicSolve(const Parameters::AllParameters& parameters,
                                const MPIInfo& mpiInfo,
                                Tria& triangulation);

    virtual ~PhaseFieldMonolithicSolve() = default;
    void run();

  private:
    struct PerTaskData_ASM;
    struct ScratchData_ASM;

    struct PerTaskData_ASM_RHS_BFGS;
    struct ScratchData_ASM_RHS_BFGS;

    struct PerTaskData_UQPH;
    struct ScratchData_UQPH;

    const Parameters::AllParameters& m_parameters;
    Tria& m_triangulation;

    CellDataStorage m_quadrature_point_history;

    Time                m_time;
      
      
    const MPIInfo& m_mpiInfo;
      
//    Logger  m_logfile;
      
    std::ofstream      __ofstream;
    ConditionalOStream m_logfile;
      
//    mutable TimerOutput m_timer;
      mutable TimerOutputWrapper<LATraits> m_timer;
      
      BlockDesc                     m_blocks_desc;

    DoFHandler<dim>                  m_dof_handler;
    FESystem<dim>                    m_fe;
    const unsigned int               m_dofs_per_cell;
    const FEValuesExtractors::Vector m_u_fe;
    const FEValuesExtractors::Scalar m_d_fe;
    const FEValuesExtractors::Scalar m_t_fe;

    static const unsigned int m_n_blocks          = 3;
    static const unsigned int m_n_components      = dim + 1 + 1;
    static const unsigned int m_first_u_component = 0;
    static const unsigned int m_d_component       = dim;
    static const unsigned int m_t_component       = dim + 1;

    enum
    {
      m_u_dof = 0,
      m_d_dof = 1,
      m_t_dof = 2
    };

    std::vector<types::global_dof_index> m_dofs_per_block;

    const QGauss<dim>     m_qf_cell;
    const QGauss<dim - 1> m_qf_face;
    const unsigned int    m_n_q_points;

    double m_vol_reference;

    AffineConstraints<double> m_constraints;
    BlockSparsityPattern      m_sparsity_pattern;

//      BlockSparseMatrix<double> m_tangent_matrix;
//      BlockVector<double>       m_system_rhs;
//      BlockVector<double>       m_solution;

    BSMatrix                  m_tangent_matrix;
    BVector                   m_system_rhs;
    BVector                   m_solution;
      
    SparseDirectUMFPACK       m_A_direct;


    std::map<unsigned int, std::vector<double>> m_material_data;

    std::vector<std::pair<double, std::vector<double>>> m_history_reaction_force;
    std::vector<std::pair<double, std::array<double, 3>>> m_history_energy;

    LASolver<LATraits>     m_solver;
      
      
    OutputHelper<LATraits, Tria, PointHistory<dim>>   m_output;

    struct Errors
    {
      Errors()
        : m_norm(1.0)
        , m_u(1.0)
        , m_d(1.0)
        , m_t(1.0)
      {}

      void reset()
      {
        m_norm = 1.0;
        m_u    = 1.0;
        m_d    = 1.0;
        m_t    = 1.0;
      }

      void normalize(const Errors &rhs)
      {
        if (rhs.m_norm != 0.0)
          m_norm /= rhs.m_norm;
        if (rhs.m_u != 0.0)
          m_u /= rhs.m_u;
        if (rhs.m_d != 0.0)
          m_d /= rhs.m_d;
        if (rhs.m_t != 0.0)
          m_t /= rhs.m_t;
      }

      double m_norm, m_u, m_d, m_t;
    };

    Errors m_error_residual, m_error_residual_0, m_error_residual_norm, m_error_update,
      m_error_update_0, m_error_update_norm;

    void get_error_residual(Errors &error_residual);
    void get_error_update(const BVector &soln_update,
                          Errors & error_update);

    void make_grid();
    void make_grid_case_1();
    void make_grid_case_2();
    void make_grid_case_3();
    void make_grid_case_4();
    void make_grid_case_5();
    void make_grid_case_6();
    void make_grid_case_7();
    void make_grid_case_8();
    void make_grid_case_9();
    void make_grid_case_10();
    void make_grid_case_11();

    void setup_system();

    void setup_temperature_initial_conditions();

    void determine_component_extractors();

    void make_constraints(const unsigned int it_nr);

    void assemble_system_B0();

    void assemble_system_B0_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM &                                     scratch,
      PerTaskData_ASM &                                     data) const;

    void assemble_system_rhs_LBFGS_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM_RHS_BFGS &                           scratch,
      PerTaskData_ASM_RHS_BFGS &                           data) const;

    void assemble_system_rhs_LBFGS_parallel(const BVector & solution_old,
    				           BVector & system_rhs);

    void solve_nonlinear_timestep_LBFGS(BVector &solution_delta,
					BVector & LBFGS_update_refine);

    double line_search_stepsize_strong_wolfe(const double phi_0,
				             const double phi_0_prime,
				             const BVector & BFGS_p_vector,
				             const BVector & solution_delta);

    double line_search_stepsize_gradient_based(const BVector & BFGS_p_vector,
					       const BVector & solution_delta);

    double line_search_zoom_strong_wolfe(double phi_low, double phi_low_prime, double alpha_low,
					 double phi_high, double phi_high_prime, double alpha_high,
					 double phi_0, double phi_0_prime, const BVector & BFGS_p_vector,
					 double c1, double c2, unsigned int max_iter,
					 const BVector & solution_delta);

    double line_search_stepsize_residual_projection(const double f0,
				                    const BVector & BFGS_p_vector,
				                    const BVector & solution_delta);

    double binary_search(double a, double b,
			 double fa, double fb,
			 const double threshold,
			 const BVector & BFGS_p_vector,
			 const BVector & solution_delta);

    double line_search_interpolation_cubic(const double alpha_0, const double phi_0, const double phi_0_prime,
					   const double alpha_1, const double phi_1, const double phi_1_prime);

    std::pair<double, double> calculate_phi_and_phi_prime(const double alpha,
							  const BVector & BFGS_p_vector,
							  const BVector & solution_delta);

    double calculate_phi_prime(const double alpha,
  	    		       const BVector & BFGS_p_vector,
  			       const BVector & solution_delta);

    void LBFGS_B0(BVector & LBFGS_r_vector,
		  const BVector & LBFGS_q_vector);

    void update_history_field_step();

    void output_results() const;

    void setup_qph();

    void update_qph_incremental(const BVector &solution_delta,
				const BVector &solution_old,
				const bool is_print);

    void update_qph_incremental_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_UQPH &                                    scratch,
      PerTaskData_UQPH &                                    data);

    void copy_local_to_global_UQPH(const PerTaskData_UQPH & /*data*/)
    {}

    BVector
    get_total_solution(const BVector &solution_delta) const;

    // Should not make this function const
    void read_material_data(const std::string &data_file,
			    const unsigned int total_material_regions);

    void read_time_data(const std::string &data_file,
    		        std::vector<std::array<double, 4>> & time_table);

    void print_conv_header_LBFGS();

    void print_parameter_information();

    void calculate_reaction_force(unsigned int face_ID);

    void write_history_data();

    double calculate_energy_functional() const;

    std::pair<double, double> calculate_total_strain_energy_and_crack_energy_dissipation() const;

    bool local_refine_and_solution_transfer(BVector & solution_delta,
					    BVector & LBFGS_update_refine);
  }; // class PhaseFieldMonolithicSolve

namespace type{

template <typename LATraits, typename Tria>
using BSMatrix = typename PhaseFieldMonolithicSolve<LATraits, Tria>::BSMatrix;

template <typename LATraits, typename Tria>
using BVector  = typename PhaseFieldMonolithicSolve<LATraits, Tria>::BVector;

}

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::get_error_residual(Errors &error_residual)
  {
    BVector error_res(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
      error_res.initalize();
      
    for (unsigned int i = 0; i < m_dof_handler.n_dofs(); ++i)
      if (!m_constraints.is_constrained(i))
        error_res(i) = m_system_rhs(i);

    error_residual.m_norm = error_res.l2_norm();
    error_residual.m_u    = error_res.block(m_u_dof).l2_norm();
    error_residual.m_d    = error_res.block(m_d_dof).l2_norm();
    error_residual.m_t    = error_res.block(m_t_dof).l2_norm();
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::get_error_update(const BVector &soln_update,
                                                        Errors & error_update)
  {
    BVector error_ud(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
      error_ud.initalize();
      
    for (unsigned int i = 0; i < m_dof_handler.n_dofs(); ++i)
      if (!m_constraints.is_constrained(i))
        error_ud(i) = soln_update(i);

    error_update.m_norm = error_ud.l2_norm();
    error_update.m_u    = error_ud.block(m_u_dof).l2_norm();
    error_update.m_d    = error_ud.block(m_d_dof).l2_norm();
    error_update.m_t    = error_ud.block(m_t_dof).l2_norm();
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::read_material_data(const std::string &data_file,
				                     const unsigned int total_material_regions)
  {
    std::ifstream myfile (data_file);

    double lame_lambda, lame_mu, length_scale, gc_0, viscosity, residual_k;
    double heat_capacity, thermal_conductivity_0, thermal_expansion_coeff;
    double reference_temperature, max_temperature, b_1, b_2;
    int material_region;
    double poisson_ratio;
    if (myfile.is_open())
      {
        m_logfile << "Reading material data file ..." << std::endl;

        while ( myfile >> material_region
                       >> lame_lambda
		       >> lame_mu
		       >> length_scale
		       >> gc_0
		       >> viscosity
		       >> residual_k
		       >> heat_capacity
		       >> thermal_conductivity_0
		       >> thermal_expansion_coeff
		       >> reference_temperature
		       >> max_temperature
		       >> b_1
		       >> b_2)
          {
            m_material_data[material_region] = {lame_lambda,
        	                                lame_mu,
						length_scale,
						gc_0,
						viscosity,
                                                residual_k,
                                                heat_capacity,
                                                thermal_conductivity_0,
                                                thermal_expansion_coeff,
                                                reference_temperature,
                                                max_temperature,
                                                b_1,
                                                b_2};
            poisson_ratio = lame_lambda / (2*(lame_lambda + lame_mu));
            Assert( (poisson_ratio <= 0.5)&(poisson_ratio >=-1.0) , ExcInternalError());

            if (reference_temperature != m_parameters.m_ref_temperature)
              Assert(false, ExcMessage("Reference temperature inconsistent "
        	  "in the parameters.prm file and materialDataFile"));

            m_logfile << "\tRegion " << material_region << " : " << std::endl;
            m_logfile << "\t\tLame lambda = " << lame_lambda << std::endl;
            m_logfile << "\t\tLame mu = "  << lame_mu << std::endl;
            m_logfile << "\t\tPoisson ratio = "  << poisson_ratio << std::endl;
            m_logfile << "\t\tPhase field length scale (l) = " << length_scale << std::endl;
            m_logfile << "\t\tCritical energy release rate (gc_0) = "  << gc_0 << std::endl;
            m_logfile << "\t\tViscosity for regularization (eta) = "  << viscosity << std::endl;
            m_logfile << "\t\tResidual_k (k) = "  << residual_k << std::endl;
            m_logfile << "\t\tHeat capacity (c, density * specific capacity) = "  << heat_capacity << std::endl;
            m_logfile << "\t\tThermal conductivity (kappa_0) = "  << thermal_conductivity_0 << std::endl;
            m_logfile << "\t\tThermal expansion coeff (alpha) = "  << thermal_expansion_coeff << std::endl;
            m_logfile << "\t\tReference temperature (T_0) = "  << reference_temperature << std::endl;
            m_logfile << "\t\tMax temperature (T_max) = "  << max_temperature << std::endl;
            m_logfile << "\t\tb1 (temperature dependent coeff) = "  << b_1 << std::endl;
            m_logfile << "\t\tb2 (temperature dependent coeff) = "  << b_2 << std::endl;
          }

        if (m_material_data.size() != total_material_regions)
          {
            m_logfile << "Material data file has " << m_material_data.size() << " rows. However, "
        	      << "the mesh has " << total_material_regions << " material regions."
		      << std::endl;
            Assert(m_material_data.size() == total_material_regions,
                       ExcDimensionMismatch(m_material_data.size(), total_material_regions));
          }
        myfile.close();
      }
    else
      {
	m_logfile << "Material data file : " << data_file << " not exist!" << std::endl;
	Assert(false, ExcMessage("Failed to read material data file"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::read_time_data(const std::string &data_file,
				                      std::vector<std::array<double, 4>> & time_table)
  {
    std::ifstream myfile (data_file);

    double t_0, t_1, delta_t, t_magnitude;

    if (myfile.is_open())
      {
	m_logfile << "Reading time data file ..." << std::endl;

	while ( myfile >> t_0
		       >> t_1
		       >> delta_t
		       >> t_magnitude)
	  {
	    Assert( t_0 < t_1,
		    ExcMessage("For each time pair, "
			       "the start time should be smaller than the end time"));
	    time_table.push_back({{t_0, t_1, delta_t, t_magnitude}});
	  }

	Assert(std::fabs(t_1 - m_parameters.m_end_time) < 1.0e-9,
	       ExcMessage("End time in time table is inconsistent with input data in parameters.prm"));

	Assert(time_table.size() > 0,
	       ExcMessage("Time data file is empty."));
	myfile.close();
      }
    else
      {
        m_logfile << "Time data file : " << data_file << " not exist!" << std::endl;
        Assert(false, ExcMessage("Failed to read time data file"));
      }

    for (auto & time_group : time_table)
      {
	m_logfile << "\t\t"
	          << time_group[0] << ",\t"
	          << time_group[1] << ",\t"
		  << time_group[2] << ",\t"
		  << time_group[3] << std::endl;
      }
  }

template <typename LATraits, typename Tria>
void PhaseFieldMonolithicSolve<LATraits, Tria>::setup_qph()
{
    m_logfile << "\t\tSetting up quadrature point data ("
    << m_n_q_points
    << " points per cell)" << std::endl;
    
    m_quadrature_point_history.clear();
    for (auto const & cell : m_triangulation.active_cell_iterators())
    {
        // skip cells owned by other ranks in mpi mode
        if constexpr (is_mpi)
            if (!cell->is_locally_owned())
                continue;
        m_quadrature_point_history.initialize(cell, m_n_q_points);
    }
    
    unsigned int material_id;
    double lame_lambda = 0.0;
    double lame_mu = 0.0;
    double length_scale = 0.0;
    double gc_0 = 0.0;
    double viscosity = 0.0;
    double residual_k = 0.0;
    double heat_capacity = 0.0;
    double thermal_conductivity_0 = 0.0;
    double thermal_expansion_coeff = 0.0;
    double reference_temperature = 0.0;
    double max_temperature = 0.0;
    double b_1 = 0.0;
    double b_2 = 0.0;
    
    for (const auto &cell : m_triangulation.active_cell_iterators())
    {
        // skip cells owned by other ranks in mpi mode
        if constexpr (is_mpi)
            if (!cell->is_locally_owned())
                continue;
        
        material_id = cell->material_id();
        if (m_material_data.find(material_id) != m_material_data.end())
        {
            lame_lambda                = m_material_data[material_id][0];
            lame_mu                    = m_material_data[material_id][1];
            length_scale               = m_material_data[material_id][2];
            gc_0                       = m_material_data[material_id][3];
            viscosity                  = m_material_data[material_id][4];
            residual_k                 = m_material_data[material_id][5];
            heat_capacity              = m_material_data[material_id][6];
            thermal_conductivity_0     = m_material_data[material_id][7];
            thermal_expansion_coeff    = m_material_data[material_id][8];
            reference_temperature      = m_material_data[material_id][9];
            max_temperature            = m_material_data[material_id][10];
            b_1                        = m_material_data[material_id][11];
            b_2                        = m_material_data[material_id][12];
        }
        else
        {
            m_logfile << "Could not find material data for material id: " << material_id << std::endl;
            AssertThrow(false, ExcMessage("Could not find material data for material id."));
        }
        
        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
        m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());
        
        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
            lqph[q_point]->setup_lqp(lame_lambda, lame_mu, length_scale,
                                     gc_0, viscosity, residual_k,
                                     heat_capacity, thermal_conductivity_0,
                                     thermal_expansion_coeff, reference_temperature,
                                     max_temperature, b_1, b_2,
                                     m_parameters.m_coupling_on_heat_eq);
    }
}

  template <typename LATraits, typename Tria>
typename PhaseFieldMonolithicSolve<LATraits, Tria>::BVector
PhaseFieldMonolithicSolve<LATraits, Tria>::get_total_solution(
    const BVector &solution_delta) const
  {
    BVector solution_total(m_solution);
    solution_total += solution_delta;
    return solution_total;
  }

  template <typename LATraits, typename Tria>
  void
  PhaseFieldMonolithicSolve<LATraits, Tria>::update_qph_incremental(const BVector &solution_delta,
							 const BVector &solution_old,
							 const bool is_print)
  {
    m_timer.enter_subsection("Update QPH data");
    if (is_print && m_parameters.m_output_iteration_history)
      m_logfile << " UQPH " << std::flush;

    const BVector solution_total(get_total_solution(solution_delta));

    const UpdateFlags uf_UQPH(update_values | update_gradients);
    PerTaskData_UQPH  per_task_data_UQPH;
    ScratchData_UQPH  scratch_data_UQPH(m_fe,
					m_qf_cell,
					uf_UQPH,
					solution_total,
					solution_old,
					m_time.get_delta_t(),
					m_parameters.m_degrade_conductivity);

      if constexpr (!is_mpi){
          // non-mpi mode
          auto worker = [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
                               ScratchData_UQPH & scratch,
                               PerTaskData_UQPH & data)
          {
              this->update_qph_incremental_one_cell(cell, scratch, data);
          };
          
          auto copier = [this](const PerTaskData_UQPH &data)
          {
              this->copy_local_to_global_UQPH(data);
          };
          
          WorkStream::run(
                          m_dof_handler.begin_active(),
                          m_dof_handler.end(),
                          worker,
                          copier,
                          scratch_data_UQPH,
                          per_task_data_UQPH);
      } else {
          // mpi mode
          for (const auto &cell : m_dof_handler.active_cell_iterators())
              if (cell->is_locally_owned()) {
                  update_qph_incremental_one_cell(cell,
                                                  scratch_data_UQPH,
                                                  per_task_data_UQPH);
                  copy_local_to_global_UQPH(per_task_data_UQPH);
              }
      }

    m_timer.leave_subsection();
  }

  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::PerTaskData_UQPH
  {
    void reset()
    {}
  };

  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::ScratchData_UQPH
  {
    const BVector & m_solution_UQPH;

    std::vector<SymmetricTensor<2, dim>> m_solution_symm_grads_u_cell;
    std::vector<double>         m_solution_values_phasefield_cell;
    std::vector<Tensor<1, dim>> m_solution_grad_phasefield_cell;

    std::vector<double>         m_solution_values_temperature_cell;
    std::vector<Tensor<1, dim>> m_solution_grad_temperature_cell;

    FEValues<dim> m_fe_values;

    const BVector&       m_solution_previous_step;
    std::vector<double>              m_phasefield_previous_step_cell;

    const double                     m_delta_time;

    const bool m_degrade_conductivity_or_not;

    ScratchData_UQPH(const FiniteElement<dim> & fe_cell,
                     const QGauss<dim> &        qf_cell,
                     const UpdateFlags          uf_cell,
                     const BVector &solution_total,
		     const BVector &solution_old,
		     const double delta_time,
		     const bool degrade_conductivity_or_not)
      : m_solution_UQPH(solution_total)
      , m_solution_symm_grads_u_cell(qf_cell.size())
      , m_solution_values_phasefield_cell(qf_cell.size())
      , m_solution_grad_phasefield_cell(qf_cell.size())
      , m_solution_values_temperature_cell(qf_cell.size())
      , m_solution_grad_temperature_cell(qf_cell.size())
      , m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_solution_previous_step(solution_old)
      , m_phasefield_previous_step_cell(qf_cell.size())
      , m_delta_time(delta_time)
      , m_degrade_conductivity_or_not(degrade_conductivity_or_not)
    {}

    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      : m_solution_UQPH(rhs.m_solution_UQPH)
      , m_solution_symm_grads_u_cell(rhs.m_solution_symm_grads_u_cell)
      , m_solution_values_phasefield_cell(rhs.m_solution_values_phasefield_cell)
      , m_solution_grad_phasefield_cell(rhs.m_solution_grad_phasefield_cell)
      , m_solution_values_temperature_cell(rhs.m_solution_values_temperature_cell)
      , m_solution_grad_temperature_cell(rhs.m_solution_grad_temperature_cell)
      , m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_solution_previous_step(rhs.m_solution_previous_step)
      , m_phasefield_previous_step_cell(rhs.m_phasefield_previous_step_cell)
      , m_delta_time(rhs.m_delta_time)
      , m_degrade_conductivity_or_not(rhs.m_degrade_conductivity_or_not)
    {}

    void reset()
    {
      const unsigned int n_q_points = m_solution_symm_grads_u_cell.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          m_solution_symm_grads_u_cell[q]  = 0.0;
          m_solution_values_phasefield_cell[q] = 0.0;
          m_solution_grad_phasefield_cell[q] = 0.0;
          m_solution_values_temperature_cell[q] = 0.0;
          m_solution_grad_temperature_cell[q] = 0.0;
          m_phasefield_previous_step_cell[q] = 0.0;
        }
    }
  };

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::update_qph_incremental_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_UQPH & scratch,
    PerTaskData_UQPH & /*data*/)
  {
    scratch.reset();

    scratch.m_fe_values.reinit(cell);

    const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    scratch.m_fe_values[m_u_fe].get_function_symmetric_gradients(
      scratch.m_solution_UQPH, scratch.m_solution_symm_grads_u_cell);
    scratch.m_fe_values[m_d_fe].get_function_values(
      scratch.m_solution_UQPH, scratch.m_solution_values_phasefield_cell);
    scratch.m_fe_values[m_d_fe].get_function_gradients(
      scratch.m_solution_UQPH, scratch.m_solution_grad_phasefield_cell);
    scratch.m_fe_values[m_t_fe].get_function_values(
      scratch.m_solution_UQPH, scratch.m_solution_values_temperature_cell);
    scratch.m_fe_values[m_t_fe].get_function_gradients(
      scratch.m_solution_UQPH, scratch.m_solution_grad_temperature_cell);

    scratch.m_fe_values[m_d_fe].get_function_values(
      scratch.m_solution_previous_step, scratch.m_phasefield_previous_step_cell);

    for (const unsigned int q_point :
         scratch.m_fe_values.quadrature_point_indices())
      lqph[q_point]->update_field_values(scratch.m_solution_symm_grads_u_cell[q_point],
                                         scratch.m_solution_values_phasefield_cell[q_point],
					 scratch.m_solution_grad_phasefield_cell[q_point],
					 scratch.m_phasefield_previous_step_cell[q_point],
					 scratch.m_delta_time,
                                         scratch.m_solution_values_temperature_cell[q_point],
					 scratch.m_solution_grad_temperature_cell[q_point],
					 scratch.m_degrade_conductivity_or_not);
  }

  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::PerTaskData_ASM
  {
    FullMatrix<double>                   m_cell_matrix;
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM(const unsigned int dofs_per_cell)
      : m_cell_matrix(dofs_per_cell, dofs_per_cell)
      , m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_matrix = 0.0;
      m_cell_rhs    = 0.0;
    }
  };

  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::PerTaskData_ASM_RHS_BFGS
  {
    Vector<double>                       m_cell_rhs;
    std::vector<types::global_dof_index> m_local_dof_indices;

    PerTaskData_ASM_RHS_BFGS(const unsigned int dofs_per_cell)
      : m_cell_rhs(dofs_per_cell)
      , m_local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      m_cell_rhs    = 0.0;
    }
  };

  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::ScratchData_ASM
  {
    FEValues<dim>     m_fe_values;
    FEFaceValues<dim> m_fe_face_values;

    std::vector<std::vector<double>>                  m_Nx_phasefield;      // shape function values for phase-field
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx_phasefield; // gradient of shape function values for phase field

    std::vector<std::vector<double>>                  m_Nx_temperature;      // shape function values for temperature
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx_temperature; // gradient of shape function values for temperature

    std::vector<std::vector<Tensor<1, dim>>>          m_Nx_disp;       // shape function values for displacement
    std::vector<std::vector<Tensor<2, dim>>>          m_grad_Nx_disp;  // gradient of shape function values for displacement
    std::vector<std::vector<SymmetricTensor<2, dim>>> m_symm_grad_Nx_disp;  // symmetric gradient of shape function values for displacement

    ScratchData_ASM(const FiniteElement<dim> & fe_cell,
                    const QGauss<dim> &        qf_cell,
                    const UpdateFlags          uf_cell,
		    const QGauss<dim - 1> &    qf_face,
		    const UpdateFlags          uf_face)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_fe_face_values(fe_cell, qf_face, uf_face)
      , m_Nx_phasefield(qf_cell.size(),
	                std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_phasefield(qf_cell.size(),
		             std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_Nx_temperature(qf_cell.size(),
	                 std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_temperature(qf_cell.size(),
		              std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_Nx_disp(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_disp(qf_cell.size(),
                       std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx_disp(qf_cell.size(),
                            std::vector<SymmetricTensor<2, dim>>(fe_cell.n_dofs_per_cell()))
    {}

    ScratchData_ASM(const ScratchData_ASM &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_fe_face_values(rhs.m_fe_face_values.get_fe(),
	                 rhs.m_fe_face_values.get_quadrature(),
	                 rhs.m_fe_face_values.get_update_flags())
      , m_Nx_phasefield(rhs.m_Nx_phasefield)
      , m_grad_Nx_phasefield(rhs.m_grad_Nx_phasefield)
      , m_Nx_temperature(rhs.m_Nx_temperature)
      , m_grad_Nx_temperature(rhs.m_grad_Nx_temperature)
      , m_Nx_disp(rhs.m_Nx_disp)
      , m_grad_Nx_disp(rhs.m_grad_Nx_disp)
      , m_symm_grad_Nx_disp(rhs.m_symm_grad_Nx_disp)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx_phasefield.size();
      const unsigned int n_dofs_per_cell = m_Nx_phasefield[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx_phasefield[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_phasefield[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_Nx_temperature[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_temperature[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_Nx_disp[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_symm_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx_phasefield[q_point][k]           = 0.0;
              m_grad_Nx_phasefield[q_point][k]      = 0.0;
              m_Nx_temperature[q_point][k]          = 0.0;
              m_grad_Nx_temperature[q_point][k]     = 0.0;
              m_Nx_disp[q_point][k]                 = 0.0;
              m_grad_Nx_disp[q_point][k]            = 0.0;
              m_symm_grad_Nx_disp[q_point][k]       = 0.0;
            }
        }
    }
  };

  template <typename LATraits, typename Tria>
  struct PhaseFieldMonolithicSolve<LATraits, Tria>::ScratchData_ASM_RHS_BFGS
  {
    FEValues<dim>     m_fe_values;
    FEFaceValues<dim> m_fe_face_values;

    std::vector<std::vector<double>>                  m_Nx_phasefield;      // shape function values for phase-field
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx_phasefield; // gradient of shape function values for phase field

    std::vector<std::vector<double>>                  m_Nx_temperature;      // shape function values for temperature
    std::vector<std::vector<Tensor<1, dim>>>          m_grad_Nx_temperature; // gradient of shape function values for temperature

    std::vector<std::vector<Tensor<1, dim>>>          m_Nx_disp;       // shape function values for displacement
    std::vector<std::vector<Tensor<2, dim>>>          m_grad_Nx_disp;  // gradient of shape function values for displacement
    std::vector<std::vector<SymmetricTensor<2, dim>>> m_symm_grad_Nx_disp;  // symmetric gradient of shape function values for displacement

    const BVector&       m_solution_previous_step;
    std::vector<SymmetricTensor<2, dim>> m_strain_previous_step_cell;
    std::vector<double>              m_phasefield_previous_step_cell;
    std::vector<double>              m_temperature_previous_step_cell;

    ScratchData_ASM_RHS_BFGS(const FiniteElement<dim> & fe_cell,
                             const QGauss<dim> &        qf_cell,
                             const UpdateFlags          uf_cell,
		             const QGauss<dim - 1> &    qf_face,
		             const UpdateFlags          uf_face,
		             const BVector& solution_old)
      : m_fe_values(fe_cell, qf_cell, uf_cell)
      , m_fe_face_values(fe_cell, qf_face, uf_face)
      , m_Nx_phasefield(qf_cell.size(),
	                std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_phasefield(qf_cell.size(),
		             std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_Nx_temperature(qf_cell.size(),
	                 std::vector<double>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_temperature(qf_cell.size(),
		              std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_Nx_disp(qf_cell.size(),
		  std::vector<Tensor<1, dim>>(fe_cell.n_dofs_per_cell()))
      , m_grad_Nx_disp(qf_cell.size(),
                       std::vector<Tensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_symm_grad_Nx_disp(qf_cell.size(),
                            std::vector<SymmetricTensor<2, dim>>(fe_cell.n_dofs_per_cell()))
      , m_solution_previous_step(solution_old)
      , m_strain_previous_step_cell(qf_cell.size())
      , m_phasefield_previous_step_cell(qf_cell.size())
      , m_temperature_previous_step_cell(qf_cell.size())
    {}

    ScratchData_ASM_RHS_BFGS(const ScratchData_ASM_RHS_BFGS &rhs)
      : m_fe_values(rhs.m_fe_values.get_fe(),
                    rhs.m_fe_values.get_quadrature(),
                    rhs.m_fe_values.get_update_flags())
      , m_fe_face_values(rhs.m_fe_face_values.get_fe(),
	                 rhs.m_fe_face_values.get_quadrature(),
	                 rhs.m_fe_face_values.get_update_flags())
      , m_Nx_phasefield(rhs.m_Nx_phasefield)
      , m_grad_Nx_phasefield(rhs.m_grad_Nx_phasefield)
      , m_Nx_temperature(rhs.m_Nx_temperature)
      , m_grad_Nx_temperature(rhs.m_grad_Nx_temperature)
      , m_Nx_disp(rhs.m_Nx_disp)
      , m_grad_Nx_disp(rhs.m_grad_Nx_disp)
      , m_symm_grad_Nx_disp(rhs.m_symm_grad_Nx_disp)
      , m_solution_previous_step(rhs.m_solution_previous_step)
      , m_strain_previous_step_cell(rhs.m_strain_previous_step_cell)
      , m_phasefield_previous_step_cell(rhs.m_phasefield_previous_step_cell)
      , m_temperature_previous_step_cell(rhs.m_temperature_previous_step_cell)
    {}

    void reset()
    {
      const unsigned int n_q_points      = m_Nx_phasefield.size();
      const unsigned int n_dofs_per_cell = m_Nx_phasefield[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          Assert(m_Nx_phasefield[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_phasefield[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_Nx_temperature[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_temperature[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_Nx_disp[q_point].size() == n_dofs_per_cell,
		 ExcInternalError());

          Assert(m_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          Assert(m_symm_grad_Nx_disp[q_point].size() == n_dofs_per_cell,
                 ExcInternalError());

          m_strain_previous_step_cell[q_point] = 0.0;
          m_phasefield_previous_step_cell[q_point] = 0.0;
          m_temperature_previous_step_cell[q_point] = 0.0;

          for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
            {
              m_Nx_phasefield[q_point][k]           = 0.0;
              m_grad_Nx_phasefield[q_point][k]      = 0.0;
              m_Nx_temperature[q_point][k]          = 0.0;
              m_grad_Nx_temperature[q_point][k]     = 0.0;
              m_Nx_disp[q_point][k]                 = 0.0;
              m_grad_Nx_disp[q_point][k]            = 0.0;
              m_symm_grad_Nx_disp[q_point][k]       = 0.0;
            }
        }
    }
  };

  // constructor has no return type
  template <typename LATraits, typename Tria>
  PhaseFieldMonolithicSolve<LATraits, Tria>
::PhaseFieldMonolithicSolve(const Parameters::AllParameters& parameters,
                            const MPIInfo& mpiInfo,
                            Tria& triangulation)
    : m_parameters(parameters)
//    , m_triangulation(Triangulation<dim>::maximum_smoothing)
    , m_triangulation(triangulation)
    , m_time(m_parameters.m_end_time)
    , m_mpiInfo(mpiInfo)
//    , m_logfile(mpiInfo, parameters.m_output_dir, parameters.m_logfile_name, 0)
    , __ofstream(parameters.m_output_dir + parameters.m_logfile_name)
    , m_logfile(__ofstream, mpiInfo.rank() == 0)
//    , m_timer(*m_mpiInfo.mpiCommPtr(), m_logfile, TimerOutput::summary, TimerOutput::wall_times)
    , m_timer(m_logfile, m_mpiInfo, TimerOutput::summary, TimerOutput::wall_times)
    , m_blocks_desc(m_mpiInfo,
                    {
        {dim, "displacement"},
        {1,   "phase-field"},
        {1,   "temperature"}
        })
    , m_dof_handler(m_triangulation)
    , m_fe(FE_Q<dim>(m_parameters.m_poly_degree),
	   dim, // displacement
	   FE_Q<dim>(m_parameters.m_poly_degree),
	   1,   // phasefield
	   FE_Q<dim>(m_parameters.m_poly_degree),
	   1)   // temperature
    , m_dofs_per_cell(m_fe.n_dofs_per_cell())
    , m_u_fe(m_first_u_component)
    , m_d_fe(m_d_component)
    , m_t_fe(m_t_component)
    , m_dofs_per_block(m_n_blocks)
    , m_qf_cell(m_parameters.m_quad_order)
    , m_qf_face(m_parameters.m_quad_order)
    , m_n_q_points(m_qf_cell.size())
    , m_vol_reference(0.0)
    , m_tangent_matrix(m_mpiInfo, 
                       m_blocks_desc,
                       [](unsigned int, unsigned int){return DoFTools::always;})
    , m_system_rhs(m_mpiInfo, m_blocks_desc, /*relevance=*/ true)
    , m_solution(m_mpiInfo, m_blocks_desc, /*relevance=*/ true)
    , m_solver(m_parameters.m_type_linear_solver == "Direct"
               ? SolverType::Direct : SolverType::CG,
               m_parameters.m_cg_u_tol,
               m_parameters.m_cg_d_tol,
               m_parameters.m_cg_t_tol,
               m_blocks_desc)
    , m_output(m_mpiInfo,
               m_triangulation,
               m_dof_handler,
               m_qf_cell)
  {}

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid()
  {
    if (m_parameters.m_scenario == 1)
      make_grid_case_1();
    else if (m_parameters.m_scenario == 2)
      make_grid_case_2();
    else if (m_parameters.m_scenario == 3)
      make_grid_case_3();
    else if (m_parameters.m_scenario == 4)
      make_grid_case_4();
    else if (m_parameters.m_scenario == 5)
      make_grid_case_5();
    else if (m_parameters.m_scenario == 6)
      make_grid_case_6();
    else if (m_parameters.m_scenario == 7)
      make_grid_case_7();
    else if (m_parameters.m_scenario == 8)
      make_grid_case_8();
    else if (m_parameters.m_scenario == 9)
      make_grid_case_9();
    else if (m_parameters.m_scenario == 10)
      make_grid_case_10();
    else if (m_parameters.m_scenario == 11)
      make_grid_case_11();
    else
      Assert(false, ExcMessage("The scenario has not been implemented!"));

      if constexpr (std::is_same_v<Tria, DTria<2>> ||
                    std::is_same_v<Tria, DTria<3>>)
      {
          // TODO: flag for repartitioning
          if(true) {
              m_triangulation.repartition();
          }
      }
      
    m_logfile << "\t\tTriangulation:"
              << "\n\t\t\tNumber of active cells: "
              << m_triangulation.n_active_cells()
              << "\n\t\t\tNumber of used vertices: "
              << m_triangulation.n_used_vertices()
	      << std::endl;

    std::ofstream out(m_parameters.oriDir + "original_mesh.vtu");
    GridOut       grid_out;
    grid_out.write_vtu(m_triangulation, out);

    m_vol_reference = GridTools::volume(m_triangulation);
    m_logfile << "\t\tGrid:\n\t\t\tReference volume: " << m_vol_reference << std::endl;
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_1()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tSquare tension (unstructured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_tension_unstructured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] + 0.5 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.5 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else
	        face->set_boundary_id(2);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   std::fabs(cell->center()[1]) < 0.01
		    && cell->center()[0] > 0.495)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   std::fabs(cell->center()[1] - 0.0) < 0.05
		    && std::fabs(cell->center()[0] - 0.5) < 0.05)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }


  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_2()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSquare shear (unstructured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_shear_unstructured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] + 0.5 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.5 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (   (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9)
		       || (std::fabs(face->center()[0] - 1.0 ) < 1.0e-9))
	        face->set_boundary_id(2);
	      else
	        face->set_boundary_id(3);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (cell->center()[0] > 0.45)
		     && (cell->center()[1] < 0.05) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && cell->center()[1] < 0.0 && cell->center()[1] > -0.025)
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_3()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\tSquare tension (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_tension_structured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 1.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else
	        face->set_boundary_id(2);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (std::fabs(cell->center()[1] - 0.5) < 0.025)
		     && (cell->center()[0] > 0.475) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && std::fabs(cell->center()[1] - 0.5) < 0.025 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_4()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tSquare shear (structured)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    GridIn<dim> gridin;
    gridin.attach_triangulation(m_triangulation);
    std::ifstream f("square_shear_structured.msh");
    gridin.read_msh(f);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 1.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (   (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9)
		       || (std::fabs(face->center()[0] - 1.0 ) < 1.0e-9))
	        face->set_boundary_id(2);
	      else
	        face->set_boundary_id(3);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	unsigned int material_id;
	double length_scale;
	for (unsigned int i = 0; i < m_parameters.m_local_prerefine_times; i++)
	  {
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    (cell->center()[0] > 0.475)
		     && (cell->center()[1] < 0.525) )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      cell->set_refine_flag();
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (    std::fabs(cell->center()[0] - 0.5) < 0.025
		     && cell->center()[1] < 0.5 && cell->center()[1] > 0.475 )
		  {
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_5()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tQuenching test (quarter size)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    double const length = 25.0; //mm
    double const width  = 5.0;  //mm

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = 100;
    repetitions[1] = 20;

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
					      repetitions,
					      Point<dim>( 0.0,      0.0 ),
					      Point<dim>( length,   width ) );

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (std::fabs(face->center()[0] - 25.0 ) < 1.0e-9)
	        face->set_boundary_id(2);
	      else if (std::fabs(face->center()[1] - 5.0 ) < 1.0e-9)
	        face->set_boundary_id(3);
	      else
	        face->set_boundary_id(4);
	    }
	}

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	//m_triangulation.refine_global(m_parameters.m_global_refine_times);
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   (cell->center()[0] >  0.0 && cell->center()[0] <  3.0)
		    || (cell->center()[1] >  0.0 && cell->center()[1] <  3.0)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
			cell->set_refine_flag();
			initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   (cell->center()[0] >  0.0 && cell->center()[0] <  0.13)
		    || (cell->center()[1] >  0.0 && cell->center()[1] <  0.13)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_6()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tQuenching test (half size)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==2, ExcMessage("The dimension has to be 2D!"));

    double const length = 25.0; //mm
    double const width  = 10.0;  //mm

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = 125;
    repetitions[1] = 50;

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
					      repetitions,
					      Point<dim>( 0.0,      0.0 ),
					      Point<dim>( length,   width ) );

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (std::fabs(face->center()[0] - length ) < 1.0e-9)
	        face->set_boundary_id(2);
	      else if (std::fabs(face->center()[1] - width ) < 1.0e-9)
	        face->set_boundary_id(3);
	      else
	        face->set_boundary_id(4);
	    }
	}

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	m_triangulation.refine_global(m_parameters.m_global_refine_times);
	/*
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   (cell->center()[0] >  0.0 && cell->center()[0] <  3.0)
		    || (cell->center()[1] >  0.0 && cell->center()[1] <  3.0)
		    || (cell->center()[1] >  width - 3.0)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
			cell->set_refine_flag();
			initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
	  */
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   (cell->center()[0] >  0.0 && cell->center()[0] <  0.13)
		    || (cell->center()[1] >  0.0 && cell->center()[1] <  0.13)
		    || (cell->center()[1] >  width - 0.13)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::sqrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_7()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tQuenching test (3D, one layer)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==3, ExcMessage("The dimension has to be 3D!"));

    double const length = 25.0; //mm
    double const width  = 5.0;  //mm
    double const thickness = 0.25;  //mm

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = 100;
    repetitions[1] = 20;
    repetitions[2] = 1;

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
					      repetitions,
					      Point<dim>( 0.0,      0.0,    0.0),
					      Point<dim>( length,   width,  thickness ) );

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (std::fabs(face->center()[2] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(2);
	      else if (std::fabs(face->center()[0] - length ) < 1.0e-9)
	        face->set_boundary_id(3);
	      else if (std::fabs(face->center()[1] - width ) < 1.0e-9)
	        face->set_boundary_id(4);
	      else if (std::fabs(face->center()[2] - thickness ) < 1.0e-9)
	        face->set_boundary_id(5);
	      else
	        face->set_boundary_id(6);
	    }
	}

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	AssertThrow(false,
		    ExcMessage("3D problem cannot afford a pre-refined mesh!"));
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   (cell->center()[0] >  0.0 && cell->center()[0] <  0.13)
		    || (cell->center()[1] >  0.0 && cell->center()[1] <  0.13)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::cbrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_8()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tQuenching test (3D, quarter box)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==3, ExcMessage("The dimension has to be 3D!"));

    double const length = 5.0; //mm
    double const width  = 2.0;  //mm
    double const thickness = 1.0;  //mm

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = 20;
    repetitions[1] = 8;
    repetitions[2] = 4;

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
					      repetitions,
					      Point<dim>( 0.0,      0.0,    0.0),
					      Point<dim>( length,   width,  thickness ) );

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (std::fabs(face->center()[2] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(2);
	      else if (std::fabs(face->center()[0] - length ) < 1.0e-9)
	        face->set_boundary_id(3);
	      else if (std::fabs(face->center()[1] - width ) < 1.0e-9)
	        face->set_boundary_id(4);
	      else if (std::fabs(face->center()[2] - thickness ) < 1.0e-9)
	        face->set_boundary_id(5);
	      else
	        face->set_boundary_id(6);
	    }
	}

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	AssertThrow(false,
		    ExcMessage("3D problem cannot afford a pre-refined mesh!"));
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   (cell->center()[0] >  0.0 && cell->center()[0] <  0.13)
		    || (cell->center()[1] >  0.0 && cell->center()[1] <  0.13)
		    || (cell->center()[2] >  0.0 && cell->center()[2] <  0.13)
		    || (cell->center()[2] >  thickness - 0.13)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::cbrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_9()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tQuenching test (3D, quarter box, bottom shock)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==3, ExcMessage("The dimension has to be 3D!"));

    double const length = 5.0; //mm
    double const width  = 2.0;  //mm
    double const thickness = 1.0;  //mm

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = 20;
    repetitions[1] = 8;
    repetitions[2] = 4;

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
					      repetitions,
					      Point<dim>( 0.0,      0.0,    0.0),
					      Point<dim>( length,   width,  thickness ) );

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (std::fabs(face->center()[2] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(2);
	      else if (std::fabs(face->center()[0] - length ) < 1.0e-9)
	        face->set_boundary_id(3);
	      else if (std::fabs(face->center()[1] - width ) < 1.0e-9)
	        face->set_boundary_id(4);
	      else if (std::fabs(face->center()[2] - thickness ) < 1.0e-9)
	        face->set_boundary_id(5);
	      else
	        face->set_boundary_id(6);
	    }
	}

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	AssertThrow(false,
		    ExcMessage("3D problem cannot afford a pre-refined mesh!"));
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (   (cell->center()[0] >  0.0 && cell->center()[0] <  0.13)
		    || (cell->center()[1] >  0.0 && cell->center()[1] <  0.13)
		    || (cell->center()[2] >  0.0 && cell->center()[2] <  0.13)
		    || (cell->center()[2] >  thickness - 0.13)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::cbrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_10()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tQuenching test (3D, quarter box, side shock)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==3, ExcMessage("The dimension has to be 3D!"));

    double const length = 5.0; //mm
    double const width  = 2.0;  //mm
    double const thickness = 1.0;  //mm

    std::vector<unsigned int> repetitions(dim, 1);
    repetitions[0] = 20;
    repetitions[1] = 8;
    repetitions[2] = 4;

    GridGenerator::subdivided_hyper_rectangle(m_triangulation,
					      repetitions,
					      Point<dim>( 0.0,      0.0,    0.0),
					      Point<dim>( length,   width,  thickness ) );

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (std::fabs(face->center()[2] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(2);
	      else if (std::fabs(face->center()[0] - length ) < 1.0e-9)
	        face->set_boundary_id(3);
	      else if (std::fabs(face->center()[1] - width ) < 1.0e-9)
	        face->set_boundary_id(4);
	      else if (std::fabs(face->center()[2] - thickness ) < 1.0e-9)
	        face->set_boundary_id(5);
	      else
	        face->set_boundary_id(6);
	    }
	}

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	AssertThrow(false,
		    ExcMessage("3D problem cannot afford a pre-refined mesh!"));
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		if (  (cell->center()[1] >  0.0 && cell->center()[1] <  0.13)
		    )
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::cbrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_grid_case_11()
  {
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;
    m_logfile << "\t\t\t\tQuenching test (3D, quarter ball)" << std::endl;
    for (unsigned int i = 0; i < 80; ++i)
      m_logfile << "*";
    m_logfile << std::endl;

    AssertThrow(dim==3, ExcMessage("The dimension has to be 3D!"));

    const double radius = 5.0;
    GridGenerator::quarter_hyper_ball(m_triangulation, Point<dim>(), radius);

    for (const auto &cell : m_triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
	{
	  if (face->at_boundary() == true)
	    {
	      if (std::fabs(face->center()[0] - 0.0 ) < 1.0e-9 )
		face->set_boundary_id(0);
	      else if (std::fabs(face->center()[1] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(1);
	      else if (std::fabs(face->center()[2] - 0.0 ) < 1.0e-9)
	        face->set_boundary_id(2);
	      else
	        face->set_boundary_id(3);
	    }
	}

    m_triangulation.refine_global(m_parameters.m_global_refine_times);

    if (m_parameters.m_refinement_strategy == "pre-refine")
      {
	//m_triangulation.refine_global(m_parameters.m_global_refine_times);
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		double distance2center = std::sqrt( cell->center()[0]*cell->center()[0]
					          + cell->center()[1]*cell->center()[1]
						  + cell->center()[2]*cell->center()[2]
						  );

		if (distance2center > 2.0*radius/3)
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::cbrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
			cell->set_refine_flag();
			initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	unsigned int material_id;
	double length_scale;
	bool initiation_point_refine_unfinished = true;
	while (initiation_point_refine_unfinished)
	  {
	    initiation_point_refine_unfinished = false;
	    for (const auto &cell : m_triangulation.active_cell_iterators())
	      {
		double distance2center = std::sqrt( cell->center()[0]*cell->center()[0]
							          + cell->center()[1]*cell->center()[1]
								  + cell->center()[2]*cell->center()[2]
								  );
                double ratio = std::pow(2, m_parameters.m_global_refine_times+1);

		if (distance2center > radius * (ratio-1) / ratio)
		  {
		    // Because the mesh is not imported from gmsh, there is no
		    // material ID associated with each cell. We need to manually
		    // set this ID based on the materialDateFIle
		    material_id = cell->material_id();
		    length_scale = m_material_data[material_id][2];
		    if (  std::cbrt(cell->measure())
			> length_scale * m_parameters.m_allowed_max_h_l_ratio )
		      {
		        cell->set_refine_flag();
		        initiation_point_refine_unfinished = true;
		      }
		  }
	      }
	    m_triangulation.execute_coarsening_and_refinement();
	  }
      }
    else
      {
	AssertThrow(false,
	            ExcMessage("Selected mesh refinement strategy not implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::setup_system()
  {
    m_timer.enter_subsection("Setup system");

    std::vector<unsigned int> block_component(m_n_components,
                                              m_u_dof); // displacement
    block_component[m_d_component] = m_d_dof;           // phasefield
    block_component[m_t_component] = m_t_dof;           // temperature

    m_dof_handler.distribute_dofs(m_fe);
    DoFRenumbering::Cuthill_McKee(m_dof_handler);
    DoFRenumbering::component_wise(m_dof_handler, block_component);

    m_constraints.clear();
    DoFTools::make_hanging_node_constraints(m_dof_handler, m_constraints);
    m_constraints.close();

      // TODO: move m_dofs_per_block = DoFTools::count_dofs_per_fe_block(m_dof_handler, block_component); to m_blocks_desc.updateDoFsInfo(m_dof_handler);
    m_dofs_per_block =
      DoFTools::count_dofs_per_fe_block(m_dof_handler, block_component);
      m_blocks_desc.updateDoFsInfo(m_dof_handler, false);
    

    m_logfile << "\t\tTriangulation:"
              << "\n\t\t\t Number of active cells: "
              << m_triangulation.n_active_cells()
              << "\n\t\t\t Number of used vertices: "
              << m_triangulation.n_used_vertices()
              << "\n\t\t\t Number of active edges: "
              << m_triangulation.n_active_lines()
              << "\n\t\t\t Number of active faces: "
              << m_triangulation.n_active_faces()
              << "\n\t\t\t Number of degrees of freedom (total): "
	      << m_dof_handler.n_dofs()
	      << "\n\t\t\t Number of degrees of freedom (disp): "
	      << m_dofs_per_block[m_u_dof]
	      << "\n\t\t\t Number of degrees of freedom (phasefield): "
	      << m_dofs_per_block[m_d_dof]
	      << "\n\t\t\t Number of degrees of freedom (temperature): "
	      << m_dofs_per_block[m_t_dof]
              << std::endl;

//
//    m_tangent_matrix.clear();
//    {
//      BlockDynamicSparsityPattern dsp(m_dofs_per_block, m_dofs_per_block);
//
//      Table<2, DoFTools::Coupling> coupling(m_n_components, m_n_components);
//      for (unsigned int ii = 0; ii < m_n_components; ++ii)
//        for (unsigned int jj = 0; jj < m_n_components; ++jj)
//          coupling[ii][jj] = DoFTools::always;
//
//      DoFTools::make_sparsity_pattern(
//        m_dof_handler, coupling, dsp, m_constraints, false);
//      m_sparsity_pattern.copy_from(dsp);
//    }
//
//    m_tangent_matrix.reinit(m_sparsity_pattern);
      

//    m_system_rhs.reinit(m_dofs_per_block);
//    m_solution.reinit(m_dofs_per_block);

      
      m_tangent_matrix.init(m_dof_handler, m_constraints, false);
      
      m_system_rhs.initalize();
      m_solution.initalize();
      
    setup_qph();

    m_timer.leave_subsection();
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::setup_temperature_initial_conditions()
  {
    if (   m_parameters.m_scenario == 3
    	|| m_parameters.m_scenario == 4)
      {
	for(unsigned int i = 0; i < m_dofs_per_block[m_t_dof]; ++i)
	  {
	    m_solution.block(m_t_dof)(i) = m_parameters.m_ref_temperature;
	  }
      }
    else if (m_parameters.m_scenario == 5)
      {
	for(unsigned int i = 0; i < m_dofs_per_block[m_t_dof]; ++i)
	  {
	    m_solution.block(m_t_dof)(i) = m_parameters.m_ref_temperature;
	  }

	const double cool_down_temperature = 293.15; // Kelvin

	std::map<types::global_dof_index, Point<dim> > support_points_T;
	ComponentMask temperature_mask = m_fe.component_mask(m_t_fe);
	support_points_T = DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					                         m_dof_handler,
					                         temperature_mask);

	for (auto const & item : support_points_T)
	  {
	    if (   (std::fabs(item.second[0] -  0.0) < 1.0e-9)
		|| (std::fabs(item.second[1] -  0.0) < 1.0e-9))
	      {
		m_solution(item.first) = cool_down_temperature;
	      }
	  }
      }
    else if (m_parameters.m_scenario == 6)
      {
	for(unsigned int i = 0; i < m_dofs_per_block[m_t_dof]; ++i)
	  {
	    m_solution.block(m_t_dof)(i) = m_parameters.m_ref_temperature;
	  }

	const double cool_down_temperature = 293.15; // Kelvin

	std::map<types::global_dof_index, Point<dim> > support_points_T;
	ComponentMask temperature_mask = m_fe.component_mask(m_t_fe);
	support_points_T = DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					                         m_dof_handler,
					                         temperature_mask);

	for (auto const & item : support_points_T)
	  {
	    if (   (std::fabs(item.second[0] -  0.0) < 1.0e-9)
		|| (std::fabs(item.second[1] -  0.0) < 1.0e-9)
		|| (std::fabs(item.second[1] -  10.0) < 1.0e-9))
	      {
		m_solution(item.first) = cool_down_temperature;
	      }
	  }
      }
    else if (m_parameters.m_scenario == 7)
      {
	for(unsigned int i = 0; i < m_dofs_per_block[m_t_dof]; ++i)
	  {
	    m_solution.block(m_t_dof)(i) = m_parameters.m_ref_temperature;
	  }

	const double cool_down_temperature = 293.15; // Kelvin

	std::map<types::global_dof_index, Point<dim> > support_points_T;
	ComponentMask temperature_mask = m_fe.component_mask(m_t_fe);
	support_points_T = DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					                         m_dof_handler,
					                         temperature_mask);

	for (auto const & item : support_points_T)
	  {
	    if (   (std::fabs(item.second[0] -  0.0) < 1.0e-9)
		|| (std::fabs(item.second[1] -  0.0) < 1.0e-9)
		//|| (std::fabs(item.second[2] -  0.0) < 1.0e-9)
		//|| (std::fabs(item.second[2] -  0.25) < 1.0e-9)
		)
	      {
		m_solution(item.first) = cool_down_temperature;
	      }
	  }
      }
    else if (m_parameters.m_scenario == 8)
      {
	for(unsigned int i = 0; i < m_dofs_per_block[m_t_dof]; ++i)
	  {
	    m_solution.block(m_t_dof)(i) = m_parameters.m_ref_temperature;
	  }

	const double cool_down_temperature = 293.15; // Kelvin

	std::map<types::global_dof_index, Point<dim> > support_points_T;
	ComponentMask temperature_mask = m_fe.component_mask(m_t_fe);
	support_points_T = DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					                         m_dof_handler,
					                         temperature_mask);

	for (auto const & item : support_points_T)
	  {
	    if (   (std::fabs(item.second[0] -  0.0) < 1.0e-9)
		|| (std::fabs(item.second[1] -  0.0) < 1.0e-9)
		|| (std::fabs(item.second[2] -  0.0) < 1.0e-9)
		|| (std::fabs(item.second[2] -  1.0) < 1.0e-9)
		)
	      {
		m_solution(item.first) = cool_down_temperature;
	      }
	  }
      }
    else if (m_parameters.m_scenario == 9)
      {
	for(unsigned int i = 0; i < m_dofs_per_block[m_t_dof]; ++i)
	  {
	    m_solution.block(m_t_dof)(i) = m_parameters.m_ref_temperature;
	  }

	const double cool_down_temperature = 293.15; // Kelvin

	std::map<types::global_dof_index, Point<dim> > support_points_T;
	ComponentMask temperature_mask = m_fe.component_mask(m_t_fe);
	support_points_T = DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					                         m_dof_handler,
					                         temperature_mask);

	for (auto const & item : support_points_T)
	  {
	    if ( (std::fabs(item.second[2] -  0.0) < 1.0e-9)
		)
	      {
		m_solution(item.first) = cool_down_temperature;
	      }
	  }
      }
    else if (m_parameters.m_scenario == 10)
      {
	for(unsigned int i = 0; i < m_dofs_per_block[m_t_dof]; ++i)
	  {
	    m_solution.block(m_t_dof)(i) = m_parameters.m_ref_temperature;
	  }

	const double cool_down_temperature = 293.15; // Kelvin

	std::map<types::global_dof_index, Point<dim> > support_points_T;
	ComponentMask temperature_mask = m_fe.component_mask(m_t_fe);
	support_points_T = DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					                         m_dof_handler,
					                         temperature_mask);

	for (auto const & item : support_points_T)
	  {
	    if ( (std::fabs(item.second[1] -  0.0) < 1.0e-9)
		)
	      {
		m_solution(item.first) = cool_down_temperature;
	      }
	  }
      }
    else if (m_parameters.m_scenario == 11)
      {
	for(unsigned int i = 0; i < m_dofs_per_block[m_t_dof]; ++i)
	  {
	    m_solution.block(m_t_dof)(i) = m_parameters.m_ref_temperature;
	  }

	const double cool_down_temperature = 293.15; // Kelvin

	std::map<types::global_dof_index, Point<dim> > support_points_T;
	ComponentMask temperature_mask = m_fe.component_mask(m_t_fe);
	support_points_T = DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
					                         m_dof_handler,
					                         temperature_mask);

	// This radius has to be consistent with the radius value
	// used in make_grid_case_11()
	const double radius = 5.0;
	for (auto const & item : support_points_T)
	  {
	    double distance2center = std::sqrt( item.second[0]*item.second[0]
	    				      + item.second[1]*item.second[1]
	    				      + item.second[2]*item.second[2]
	    				      );

	    if (std::fabs(distance2center - radius) < 1.0e-6)
	      {
		m_solution(item.first) = cool_down_temperature;
	      }
	  }
      }
    else
      {
	Assert(false, ExcMessage("The scenario has not been implemented!"));
      }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::make_constraints(const unsigned int it_nr)
  {
    const bool apply_dirichlet_bc = (it_nr == 0);

    if (it_nr > 1)
      {
	if (m_parameters.m_output_iteration_history)
          m_logfile << " --- " << std::flush;
        return;
      }

    if (m_parameters.m_output_iteration_history)
      m_logfile << " CST " << std::flush;

    if (apply_dirichlet_bc)
      {
	m_constraints.clear();
	DoFTools::make_hanging_node_constraints(m_dof_handler,
						m_constraints);

	const FEValuesExtractors::Scalar x_displacement(0);
	const FEValuesExtractors::Scalar y_displacement(1);
	const FEValuesExtractors::Scalar z_displacement(dim-1);

	const FEValuesExtractors::Vector displacements(0);

	const FEValuesExtractors::Scalar temperature(dim+1);

	if (   m_parameters.m_scenario == 1
	    || m_parameters.m_scenario == 3)
	  {
	    // Remember, the essential B.C. is applied incrementally during each time step.
	    // If a constant temperature is needed through time, the B.C should be set as zero.
	    double delta_temperature = 0.0; // temperature change per load step

	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    // temperature B.C. at the bottom surface
	    VectorTools::interpolate_boundary_values(m_dof_handler,
	    				             boundary_id_bottom_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
	    					     m_constraints,
	    					     m_fe.component_mask(temperature));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_xy(m_fe.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] - 0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] - 0.0) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    m_constraints.add_line(node_xy[0]);
	    m_constraints.set_inhomogeneity(node_xy[0], 0.0);

	    m_constraints.add_line(node_xy[1]);
	    m_constraints.set_inhomogeneity(node_xy[1], 0.0);

	    const int boundary_id_top_surface = 1;
	    /*
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));
	    */
            const double time_inc = m_time.get_delta_t();
            double disp_magnitude = m_time.get_magnitude();
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       disp_magnitude*time_inc, m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    // temperature B.C. at the top surface
	    if (m_time.current() <= 0.25e-3)
	      delta_temperature = -time_inc * 1.0e5; //cool down
//	      delta_temperature =  time_inc * 1.0e5; //warn up
//	      delta_temperature = 0.0; //constant

	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));
	  }
	else if (   m_parameters.m_scenario == 2
	         || m_parameters.m_scenario == 4)
	  {
	    // Remember, the essential B.C. is applied incrementally during each time step.
	    // If a constant temperature is needed through time, the B.C should be set as zero.
	    double delta_temperature = 0.0; // temperature change per load step

	    // Dirichlet B,C. bottom surface
	    const int boundary_id_bottom_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_bottom_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(displacements));

	    // temperature B.C. at the bottom surface
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_bottom_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));

	    const int boundary_id_top_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    const double time_inc = m_time.get_delta_t();
	    double disp_magnitude = m_time.get_magnitude();
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       disp_magnitude*time_inc, m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));

	    const int boundary_id_side_surfaces = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_side_surfaces,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    // temperature B.C. at the top surface
	    if (m_time.current() <= 10.0001e-3)
	      delta_temperature = -time_inc * 2.0e4; //cool down
//	      delta_temperature =  time_inc * 2.0e4; //warm up

	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));
	  }
	else if (m_parameters.m_scenario == 5)
	  {
	    const int boundary_id_mid_surface_x = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_mid_surface_x,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));

	    const int boundary_id_mid_surface_y = 3;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_mid_surface_y,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    // Remember, the essential B.C. is applied incrementally during each time step.
	    // If a constant temperature is needed through time, the B.C should be set as zero.
	    double delta_temperature = 0.0; // temperature change per load step
	    const int boundary_id_left_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_left_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));

	    const int boundary_id_bottom_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_bottom_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));
	  }
	else if (m_parameters.m_scenario == 6)
	  {
	    const int boundary_id_mid_surface_x = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_mid_surface_x,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_xy(m_fe.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] - 25.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] -  5.0) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    m_constraints.add_line(node_xy[1]);
	    m_constraints.set_inhomogeneity(node_xy[1], 0.0);

	    // Remember, the essential B.C. is applied incrementally during each time step.
	    // If a constant temperature is needed through time, the B.C should be set as zero.
	    double delta_temperature = 0.0; // temperature change per load step
	    const int boundary_id_left_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_left_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));

	    const int boundary_id_bottom_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_bottom_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));

	    const int boundary_id_top_surface = 3;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));
	  }
	else if (m_parameters.m_scenario == 7)
	  {
	    const int boundary_id_mid_surface_x = 3;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_mid_surface_x,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));

	    const int boundary_id_mid_surface_y = 4;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_mid_surface_y,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_xy(m_fe.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] -  0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] -  0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[2] -0.125) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    m_constraints.add_line(node_xy[2]);
	    m_constraints.set_inhomogeneity(node_xy[2], 0.0);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] - 25.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] -  0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[2] -0.125) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    m_constraints.add_line(node_xy[2]);
	    m_constraints.set_inhomogeneity(node_xy[2], 0.0);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] -  0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] -  5.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[2] -0.125) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    m_constraints.add_line(node_xy[2]);
	    m_constraints.set_inhomogeneity(node_xy[2], 0.0);

	    // Remember, the essential B.C. is applied incrementally during each time step.
	    // If a constant temperature is needed through time, the B.C should be set as zero.
	    double delta_temperature = 0.0; // temperature change per load step
	    const int boundary_id_left_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_left_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));

	    const int boundary_id_front_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_front_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));
	  }
	else if (m_parameters.m_scenario == 8)
	  {
	    const int boundary_id_mid_surface_x = 3;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_mid_surface_x,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));

	    const int boundary_id_mid_surface_y = 4;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_mid_surface_y,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_xy(m_fe.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] -  0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] -  0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[2] -  0.5) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    m_constraints.add_line(node_xy[2]);
	    m_constraints.set_inhomogeneity(node_xy[2], 0.0);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] -  5.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] -  0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[2] -  0.5) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    m_constraints.add_line(node_xy[2]);
	    m_constraints.set_inhomogeneity(node_xy[2], 0.0);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] -  0.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] -  2.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[2] -  0.5) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    m_constraints.add_line(node_xy[2]);
	    m_constraints.set_inhomogeneity(node_xy[2], 0.0);

	    // Remember, the essential B.C. is applied incrementally during each time step.
	    // If a constant temperature is needed through time, the B.C should be set as zero.
	    double delta_temperature = 0.0; // temperature change per load step
	    const int boundary_id_left_surface = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_left_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));

	    const int boundary_id_front_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_front_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));

	    const int boundary_id_bottom_surface = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_bottom_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));

	    const int boundary_id_top_surface = 5;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_top_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));
	  }
	else if (m_parameters.m_scenario == 9)
	  {
	    const int boundary_id_mid_surface_x = 3;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_mid_surface_x,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));

	    const int boundary_id_mid_surface_y = 4;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_mid_surface_y,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_xy(m_fe.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] -  5.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] -  2.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[2] -  0.5) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    m_constraints.add_line(node_xy[2]);
	    m_constraints.set_inhomogeneity(node_xy[2], 0.0);

	    // Remember, the essential B.C. is applied incrementally during each time step.
	    // If a constant temperature is needed through time, the B.C should be set as zero.
	    double delta_temperature = 0.0; // temperature change per load step
	    const int boundary_id_bottom_surface = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_bottom_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));
	  }
	else if (m_parameters.m_scenario == 10)
	  {
	    const int boundary_id_mid_surface_x = 3;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_mid_surface_x,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));

	    const int boundary_id_mid_surface_y = 4;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_mid_surface_y,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    typename Triangulation<dim>::active_vertex_iterator vertex_itr;
	    vertex_itr = m_triangulation.begin_active_vertex();
	    std::vector<types::global_dof_index> node_xy(m_fe.dofs_per_vertex);

	    for (; vertex_itr != m_triangulation.end_vertex(); ++vertex_itr)
	      {
		if (   (std::fabs(vertex_itr->vertex()[0] -  5.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[1] -  2.0) < 1.0e-9)
		    && (std::fabs(vertex_itr->vertex()[2] -  1.0) < 1.0e-9) )
		  {
		    node_xy = usr_utilities::get_vertex_dofs(vertex_itr, m_dof_handler);
		  }
	      }
	    m_constraints.add_line(node_xy[2]);
	    m_constraints.set_inhomogeneity(node_xy[2], 0.0);

	    // Remember, the essential B.C. is applied incrementally during each time step.
	    // If a constant temperature is needed through time, the B.C should be set as zero.
	    double delta_temperature = 0.0; // temperature change per load step
	    const int boundary_id_front_surface = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_front_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));
	  }
	else if (m_parameters.m_scenario == 11)
	  {
	    const int boundary_id_surface_x = 0;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_surface_x,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(x_displacement));

	    const int boundary_id_surface_y = 1;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_surface_y,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(y_displacement));

	    const int boundary_id_surface_z = 2;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_surface_z,
						     Functions::ZeroFunction<dim>(m_n_components),
						     m_constraints,
						     m_fe.component_mask(z_displacement));

	    // Remember, the essential B.C. is applied incrementally during each time step.
	    // If a constant temperature is needed through time, the B.C should be set as zero.
	    double delta_temperature = 0.0; // temperature change per load step
	    const int boundary_id_sphere_surface = 3;
	    VectorTools::interpolate_boundary_values(m_dof_handler,
						     boundary_id_sphere_surface,
						     Functions::ConstantFunction<dim>(
						       delta_temperature, m_n_components),
						     m_constraints,
						     m_fe.component_mask(temperature));
	  }
	else
	  Assert(false, ExcMessage("The scenario has not been implemented!"));
      }
    else  // inhomogeneous constraints
      {
        if (m_constraints.has_inhomogeneities())
          {
            AffineConstraints<double> homogeneous_constraints(m_constraints);
            for (unsigned int dof = 0; dof != m_dof_handler.n_dofs(); ++dof)
              if (homogeneous_constraints.is_inhomogeneously_constrained(dof))
                homogeneous_constraints.set_inhomogeneity(dof, 0.0);
            m_constraints.clear();
            m_constraints.copy_from(homogeneous_constraints);
          }
      }
    m_constraints.close();
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::assemble_system_B0()
  {
    m_timer.enter_subsection("Assemble B0");

    m_tangent_matrix = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    PerTaskData_ASM per_task_data(m_fe.n_dofs_per_cell());
    ScratchData_ASM scratch_data(m_fe, m_qf_cell, uf_cell, m_qf_face, uf_face);

      if constexpr (!is_mpi){
          // non-mpi mode
          
          auto worker =
          [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
                 ScratchData_ASM & scratch,
                 PerTaskData_ASM & data)
          {
              this->assemble_system_B0_one_cell(cell, scratch, data);
          };
          
          auto copier = [this](const PerTaskData_ASM &data)
          {
              this->m_constraints.distribute_local_to_global(data.m_cell_matrix,
                                                             data.m_local_dof_indices,
                                                             m_tangent_matrix.base());
          };
          
          WorkStream::run(
                          m_dof_handler.active_cell_iterators(),
                          worker,
                          copier,
                          scratch_data,
                          per_task_data);
      } else {
          // mpi mode
          for (const auto &cell : m_dof_handler.active_cell_iterators())
              if (cell->is_locally_owned())
              {
                  assemble_system_B0_one_cell(cell, scratch_data, per_task_data);
                  
                  m_constraints.distribute_local_to_global(per_task_data.m_cell_matrix,
                                                           per_task_data.m_local_dof_indices,
                                                           m_tangent_matrix.base());
              }
          
          /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
          m_tangent_matrix.compress(VectorOperation::add);
          /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
          
      }

    m_timer.leave_subsection();
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::assemble_system_rhs_LBFGS_parallel(const BVector & solution_old,
								         BVector & system_rhs)
  {
    m_timer.enter_subsection("Assemble RHS");

    //m_logfile << " A_RHS " << std::flush;

    system_rhs = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
			      update_JxW_values);

    PerTaskData_ASM_RHS_BFGS per_task_data(m_fe.n_dofs_per_cell());
    ScratchData_ASM_RHS_BFGS scratch_data(m_fe, m_qf_cell, uf_cell, m_qf_face, uf_face, solution_old);

      if constexpr (!is_mpi){
          // non-mpi mode
          
    auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
	     ScratchData_ASM_RHS_BFGS & scratch,
	     PerTaskData_ASM_RHS_BFGS & data)
      {
        this->assemble_system_rhs_LBFGS_one_cell(cell, scratch, data);
      };

    auto copier = [this, &system_rhs](const PerTaskData_ASM_RHS_BFGS &data)
      {
        this->m_constraints.distribute_local_to_global(data.m_cell_rhs,
                                                       data.m_local_dof_indices,
						       system_rhs.base());
      };

    WorkStream::run(
      m_dof_handler.active_cell_iterators(),
      worker,
      copier,
      scratch_data,
      per_task_data);
          
          
      } else {
          // mpi mode
          
          for (const auto &cell : m_dof_handler.active_cell_iterators())
              if (cell->is_locally_owned())
              {
                  assemble_system_rhs_LBFGS_one_cell(cell, scratch_data, per_task_data);
                  
                  m_constraints.distribute_local_to_global(per_task_data.m_cell_rhs,
                                                           per_task_data.m_local_dof_indices,
                                                           system_rhs.base());
              }
          

          /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
          system_rhs.compress(VectorOperation::add);
          /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
      }

    m_timer.leave_subsection();
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::assemble_system_rhs_LBFGS_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM_RHS_BFGS & scratch,
      PerTaskData_ASM_RHS_BFGS & data) const
{
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);
    
    scratch.m_fe_values[m_u_fe].get_function_symmetric_gradients(
                                                                 scratch.m_solution_previous_step, scratch.m_strain_previous_step_cell);
    
    scratch.m_fe_values[m_d_fe].get_function_values(
                                                    scratch.m_solution_previous_step, scratch.m_phasefield_previous_step_cell);
    
    scratch.m_fe_values[m_t_fe].get_function_values(
                                                    scratch.m_solution_previous_step, scratch.m_temperature_previous_step_cell);
    
    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
    m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());
    
    const double time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);
    
    right_hand_side(scratch.m_fe_values.get_quadrature_points(),
                    rhs_values,
                    m_parameters.m_x_component*1.0,
                    m_parameters.m_y_component*1.0,
                    m_parameters.m_z_component*1.0);
    
    std::vector<double> heat_supply_values(m_n_q_points);
    
    heat_supply(scratch.m_fe_values.get_quadrature_points(),
                heat_supply_values,
                m_parameters.m_heat_supply*1.0);
    
    const double delta_time = m_time.get_delta_t();
    
    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
    {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
        {
            const unsigned int k_group = m_fe.system_to_base_index(k).first.first;
            
            if (k_group == m_u_dof)
            {
                scratch.m_Nx_disp[q_point][k] =
                scratch.m_fe_values[m_u_fe].value(k, q_point);
                scratch.m_grad_Nx_disp[q_point][k] =
                scratch.m_fe_values[m_u_fe].gradient(k, q_point);
                scratch.m_symm_grad_Nx_disp[q_point][k] =
                symmetrize(scratch.m_grad_Nx_disp[q_point][k]);
            }
            else if (k_group == m_d_dof)
            {
                scratch.m_Nx_phasefield[q_point][k] =
                scratch.m_fe_values[m_d_fe].value(k, q_point);
                scratch.m_grad_Nx_phasefield[q_point][k] =
                scratch.m_fe_values[m_d_fe].gradient(k, q_point);
            }
            else if (k_group == m_t_dof)
            {
                scratch.m_Nx_temperature[q_point][k] =
                scratch.m_fe_values[m_t_fe].value(k, q_point);
                scratch.m_grad_Nx_temperature[q_point][k] =
                scratch.m_fe_values[m_t_fe].gradient(k, q_point);
            }
            else
                Assert(k_group <= m_t_dof, ExcInternalError());
        }
    }
    
    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
    {
        const double length_scale            = lqph[q_point]->get_length_scale();
        // temperature-dependent critical energy release rate
        const double gc_t                    = lqph[q_point]->get_critical_energy_release_rate();
        const double eta                     = lqph[q_point]->get_viscosity();
        const double history_strain_energy   = lqph[q_point]->get_history_max_positive_strain_energy();
        const double current_positive_strain_energy = lqph[q_point]->get_current_positive_strain_energy();
        const double heat_capacity           = lqph[q_point]->get_heat_capacity();
        const double ref_t                   = lqph[q_point]->get_ref_temperature();
        const double thermal_expansion       = lqph[q_point]->get_thermal_expansion_coeff();
        const double lame_lambda             = lqph[q_point]->get_lame_lambda();
        const double lame_mu                 = lqph[q_point]->get_lame_mu();
        const bool   coupling_on_heat_eq     = lqph[q_point]->get_heat_coupling_flag();
        
        double coupling_tensor_coeff = thermal_expansion
        * (trace(Physics::Elasticity::StandardTensors<dim>::I)*lame_lambda + 2.0*lame_mu);
        
        if (!coupling_on_heat_eq)
            coupling_tensor_coeff = 0.0;
        
        const SymmetricTensor<2, dim> ut_coupling_tensor
        = coupling_tensor_coeff * Physics::Elasticity::StandardTensors<dim>::I;
        
        double history_value = history_strain_energy;
        if (current_positive_strain_energy > history_strain_energy)
            history_value = current_positive_strain_energy;
        
        const double phasefield_value        = lqph[q_point]->get_phase_field_value();
        const Tensor<1, dim> phasefield_grad = lqph[q_point]->get_phase_field_gradient();
        
        const double temperature_value        = lqph[q_point]->get_temperature_value();
        
        // current total strain
        const SymmetricTensor<2, dim> & current_strain = lqph[q_point]->get_strain();
        // previous total strain
        const SymmetricTensor<2, dim> & old_strain = scratch.m_strain_previous_step_cell[q_point];
        
        const std::vector<double>         &      N_phasefield = scratch.m_Nx_phasefield[q_point];
        const std::vector<Tensor<1, dim>> & grad_N_phasefield = scratch.m_grad_Nx_phasefield[q_point];
        const double                old_phasefield = scratch.m_phasefield_previous_step_cell[q_point];
        
        const std::vector<double>         &      N_temperature = scratch.m_Nx_temperature[q_point];
        const std::vector<Tensor<1, dim>> & grad_N_temperature = scratch.m_grad_Nx_temperature[q_point];
        const double                old_temperature = scratch.m_temperature_previous_step_cell[q_point];
        
        const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();
        const Tensor<1, dim> & heat_flux = lqph[q_point]->get_heat_flux();
        
        const std::vector<Tensor<1,dim>> & N_disp = scratch.m_Nx_disp[q_point];
        const std::vector<SymmetricTensor<2, dim>> & symm_grad_N_disp =
        scratch.m_symm_grad_Nx_disp[q_point];
        const double JxW = scratch.m_fe_values.JxW(q_point);
        
        SymmetricTensor<2, dim> symm_grad_Nx_i_x_C;
        
        for (const unsigned int i : scratch.m_fe_values.dof_indices())
        {
            const unsigned int i_group = m_fe.system_to_base_index(i).first.first;
            
            if (i_group == m_u_dof)
            {
                data.m_cell_rhs(i) += (symm_grad_N_disp[i] * cauchy_stress) * JxW;
                
                // contributions from the body force to right-hand side
                data.m_cell_rhs(i) -= N_disp[i] * rhs_values[q_point] * JxW;
            }
            else if (i_group == m_d_dof)
            {
                data.m_cell_rhs(i) += (    gc_t * length_scale * grad_N_phasefield[i] * phasefield_grad
                                       +  (   gc_t / length_scale * phasefield_value
                                           + eta / delta_time  * (phasefield_value - old_phasefield)
                                           + degradation_function_derivative(phasefield_value) * history_value )
                                       * N_phasefield[i]
                                       ) * JxW;
            }
            else if (i_group == m_t_dof)
            {
                data.m_cell_rhs(i) += (heat_capacity * N_temperature[i]
                                       * (temperature_value - old_temperature) / ref_t) * JxW;
                data.m_cell_rhs(i) -= (grad_N_temperature[i] * heat_flux * delta_time / ref_t) * JxW;
                data.m_cell_rhs(i) -= N_temperature[i] * heat_supply_values[q_point] * delta_time /ref_t * JxW;
                
                // the mechanical-thermal coupling term
                data.m_cell_rhs(i) += N_temperature[i]
                * ut_coupling_tensor
                * (current_strain - old_strain)
                * JxW;
                
            }
            else
                Assert(i_group <= m_t_dof, ExcInternalError());
        }  // i
    }  // q_point
    
    // if there is surface pressure, this surface pressure always applied to the
    // reference configuration
    const unsigned int face_pressure_id = 100;
    const double p0 = 0.0;
    
    for (const auto &face : cell->face_iterators()) {
        if (face->at_boundary() && face->boundary_id() == face_pressure_id)
        {
            scratch.m_fe_face_values.reinit(cell, face);
            
            for (const unsigned int f_q_point : scratch.m_fe_face_values.quadrature_point_indices())
            {
                const Tensor<1, dim> &N = scratch.m_fe_face_values.normal_vector(f_q_point);
                
                const double         pressure  = p0 * time_ramp;
                const Tensor<1, dim> traction  = pressure * N;
                
                for (const unsigned int i : scratch.m_fe_values.dof_indices())
                {
                    const unsigned int i_group = m_fe.system_to_base_index(i).first.first;
                    
                    if (i_group == m_u_dof)
                    {
                        const unsigned int component_i = m_fe.system_to_component_index(i).first;
                        const double Ni = scratch.m_fe_face_values.shape_value(i, f_q_point);
                        const double JxW = scratch.m_fe_face_values.JxW(f_q_point);
                        data.m_cell_rhs(i) -= (Ni * traction[component_i]) * JxW;
                    }
                }
            }
        }
    }
    
    // surface heat flux (Neumann BC)
    const unsigned int face_flux_id = 100;
    const double h0 = 0.0;
    
    for (const auto &face : cell->face_iterators()){
        if (face->at_boundary() && face->boundary_id() == face_flux_id)
        {
            scratch.m_fe_face_values.reinit(cell, face);
            
            for (const unsigned int f_q_point : scratch.m_fe_face_values.quadrature_point_indices())
            {
                const double         flux  = h0 * time_ramp;
                
                for (const unsigned int i : scratch.m_fe_values.dof_indices())
                {
                    const unsigned int i_group = m_fe.system_to_base_index(i).first.first;
                    
                    if (i_group == m_t_dof)
                    {
                        const double Ni = scratch.m_fe_face_values.shape_value(i, f_q_point);
                        const double JxW = scratch.m_fe_face_values.JxW(f_q_point);
                        data.m_cell_rhs(i) -= Ni * flux * JxW;
                    }
                }
            }
        }
    }
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::assemble_system_B0_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM & scratch,
      PerTaskData_ASM & data) const
  {
    data.reset();
    scratch.reset();
    scratch.m_fe_values.reinit(cell);
    cell->get_dof_indices(data.m_local_dof_indices);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      m_quadrature_point_history.get_data(cell);
    Assert(lqph.size() == m_n_q_points, ExcInternalError());

    const double delta_time = m_time.get_delta_t();

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
        for (const unsigned int k : scratch.m_fe_values.dof_indices())
          {
            const unsigned int k_group = m_fe.system_to_base_index(k).first.first;

            if (k_group == m_u_dof)
              {
                scratch.m_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].value(k, q_point);
                scratch.m_grad_Nx_disp[q_point][k] =
                  scratch.m_fe_values[m_u_fe].gradient(k, q_point);
                scratch.m_symm_grad_Nx_disp[q_point][k] =
                  symmetrize(scratch.m_grad_Nx_disp[q_point][k]);
              }
            else if (k_group == m_d_dof)
              {
		scratch.m_Nx_phasefield[q_point][k] =
		  scratch.m_fe_values[m_d_fe].value(k, q_point);
		scratch.m_grad_Nx_phasefield[q_point][k] =
		  scratch.m_fe_values[m_d_fe].gradient(k, q_point);
              }
            else if (k_group == m_t_dof)
              {
		scratch.m_Nx_temperature[q_point][k] =
		  scratch.m_fe_values[m_t_fe].value(k, q_point);
		scratch.m_grad_Nx_temperature[q_point][k] =
		  scratch.m_fe_values[m_t_fe].gradient(k, q_point);
              }
            else
              Assert(k_group <= m_t_dof, ExcInternalError());
          }
      }

    for (const unsigned int q_point : scratch.m_fe_values.quadrature_point_indices())
      {
	const double length_scale            = lqph[q_point]->get_length_scale();
	// temperature-dependent critical energy release rate
	const double gc_t                    = lqph[q_point]->get_critical_energy_release_rate();
	const double eta                     = lqph[q_point]->get_viscosity();
	const double history_strain_energy   = lqph[q_point]->get_history_max_positive_strain_energy();
	const double current_positive_strain_energy = lqph[q_point]->get_current_positive_strain_energy();
        // degraded thermal conductivity
	const double kappa_d                 = lqph[q_point]->get_thermal_conductivity();
	const double heat_capacity           = lqph[q_point]->get_heat_capacity();
        const double ref_t                   = lqph[q_point]->get_ref_temperature();

	double history_value = history_strain_energy;
	if (current_positive_strain_energy > history_strain_energy)
	  history_value = current_positive_strain_energy;

	const double phasefield_value = lqph[q_point]->get_phase_field_value();

        const std::vector<double>         &      N_phasefield = scratch.m_Nx_phasefield[q_point];
        const std::vector<Tensor<1, dim>> & grad_N_phasefield = scratch.m_grad_Nx_phasefield[q_point];

        const std::vector<double>         &      N_temperature = scratch.m_Nx_temperature[q_point];
        const std::vector<Tensor<1, dim>> & grad_N_temperature = scratch.m_grad_Nx_temperature[q_point];

        const SymmetricTensor<4, dim> & mechanical_C  = lqph[q_point]->get_mechanical_C();

        const std::vector<SymmetricTensor<2, dim>> & symm_grad_N_disp =
          scratch.m_symm_grad_Nx_disp[q_point];
        const double JxW = scratch.m_fe_values.JxW(q_point);

        SymmetricTensor<2, dim> symm_grad_Nx_i_x_C;

        for (const unsigned int i : scratch.m_fe_values.dof_indices())
          {
            const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

            if (i_group == m_u_dof)
              {
                symm_grad_Nx_i_x_C = symm_grad_N_disp[i] * mechanical_C;
              }

            for (const unsigned int j : scratch.m_fe_values.dof_indices())
              {
                const unsigned int j_group = m_fe.system_to_base_index(j).first.first;

                if ((i_group == j_group) && (i_group == m_u_dof))
                  {
                    data.m_cell_matrix(i, j) += symm_grad_Nx_i_x_C * symm_grad_N_disp[j] * JxW;
                  }
                else if ((i_group == j_group) && (i_group == m_d_dof))
                  {
                    data.m_cell_matrix(i, j) += (  (   gc_t/length_scale + eta/delta_time
                	                             + degradation_function_2nd_order_derivative(phasefield_value)
						     * history_value  )
                	                          * N_phasefield[i] * N_phasefield[j]
					          + gc_t * length_scale * grad_N_phasefield[i] * grad_N_phasefield[j]
					        ) * JxW;
                  }
                else if ((i_group == j_group) && (i_group == m_t_dof))
                  {
                    data.m_cell_matrix(i, j) += (  heat_capacity
                	                         * N_temperature[i] * N_temperature[j]
						 + kappa_d
						 * grad_N_temperature[i] * grad_N_temperature[j] * delta_time
					        ) / ref_t * JxW;
                  }
                else
                  Assert((i_group <= m_t_dof) && (j_group <= m_t_dof),
                         ExcInternalError());
              } // j
          }  // i
      }  // q_point
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::update_history_field_step()
  {
    m_logfile << "\t\tUpdate history variable" << std::endl;

      for (const auto &cell : m_triangulation.active_cell_iterators())
      {
          // skip cells owned by other ranks in mpi mode
          if constexpr (is_mpi)
              if (!cell->is_locally_owned())
                  continue;
          std::vector<std::shared_ptr< PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
          Assert(lqph.size() == m_n_q_points, ExcInternalError());
          
          for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
              lqph[q_point]->update_history_variable();
          }
      }
  }

  template <typename LATraits, typename Tria>
  double PhaseFieldMonolithicSolve<LATraits, Tria>::line_search_stepsize_gradient_based(const BVector & BFGS_p_vector,
				                                             const BVector & solution_delta)
  {
    BVector g_old(m_system_rhs);

    // BFGS_p_vector is the search direction
    BVector solution_delta_trial(solution_delta);
    // take a full step size 1.0
    solution_delta_trial.add(1.0, BFGS_p_vector);

    update_qph_incremental(solution_delta_trial, m_solution, false);

    BVector g_new(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
      g_new.initalize();
    assemble_system_rhs_LBFGS_parallel(m_solution, g_new);

    BVector y_old(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
      y_old.initalize();

    y_old.base() = g_new.base() - g_old.base();

    double alpha = 1.0;

    double alpha_old = 0.0;

    double delta_alpha_old = alpha - alpha_old;

    double delta_alpha_new;

    unsigned int ls_max = 10;

    for (unsigned int i = 1; i <= ls_max; ++i)
      {
	delta_alpha_new = -delta_alpha_old
	                * (g_new * BFGS_p_vector)/(y_old * BFGS_p_vector);
	alpha += delta_alpha_new;

	if (std::fabs(delta_alpha_new) < 1.0e-5)
	  break;

        if (i == ls_max)
          {
            alpha = 1.0;
            break;
          }

        g_old = g_new;

        // BFGS_p_vector is the search direction
        solution_delta_trial = solution_delta;
        solution_delta_trial.add(alpha, BFGS_p_vector);
        update_qph_incremental(solution_delta_trial, m_solution, false);
        assemble_system_rhs_LBFGS_parallel(m_solution, g_new);

        y_old.base() = g_new.base() - g_old.base();

        delta_alpha_old = delta_alpha_new;
      }

    if (alpha < 1.0e-3)
      alpha = 1.0;

    return alpha;
  }

  template <typename LATraits, typename Tria>
  double PhaseFieldMonolithicSolve<LATraits, Tria>::line_search_stepsize_strong_wolfe(const double phi_0,
				                                           const double phi_0_prime,
				                                           const BVector & BFGS_p_vector,
				                                           const BVector & solution_delta)
  {
    //AssertThrow(phi_0_prime < 0,
    //            ExcMessage("The derivative of phi at alpha = 0 should be negative!"));

    // Some line search parameters
    const double c1 = 0.0001;
    const double c2 = 0.9;
    const double alpha_max = 100.0;
    const unsigned int max_iter = 20;
    double alpha = 1.0;

    double phi_old = phi_0;
    double phi_prime_old = phi_0_prime;
    double alpha_old = 0.0;

    double phi, phi_prime;

    std::pair<double, double> current_phi_phi_prime;

    unsigned int i = 0;
    for (; i < max_iter; ++i)
      {
	current_phi_phi_prime = calculate_phi_and_phi_prime(alpha, BFGS_p_vector, solution_delta);
	phi = current_phi_phi_prime.first;
	phi_prime = current_phi_phi_prime.second;

	if (   ( phi > (phi_0 + c1 * alpha * phi_0_prime) )
	    || ( i > 0 && phi > phi_old ) )
	  {
	    return line_search_zoom_strong_wolfe(phi_old, phi_prime_old, alpha_old,
						 phi,     phi_prime,     alpha,
						 phi_0,   phi_0_prime,   BFGS_p_vector,
						 c1,      c2,            max_iter, solution_delta);
	  }

	if (std::fabs(phi_prime) <= c2 * std::fabs(phi_0_prime))
	  {
	    return alpha;
	  }

	if (phi_prime >= 0)
	  {
	    return line_search_zoom_strong_wolfe(phi,     phi_prime,     alpha,
						 phi_old, phi_prime_old, alpha_old,
						 phi_0,   phi_0_prime,   BFGS_p_vector,
						 c1,      c2,            max_iter, solution_delta);
	  }

	phi_old = phi;
	phi_prime_old = phi_prime;
	alpha_old = alpha;

	alpha = std::min(2.0*alpha, alpha_max);

	//AssertThrow(alpha < alpha_max,
	//	    ExcMessage("alpha is bigger than alpha_max, line search failed!"));
      }

    //AssertThrow(i < max_iter,
    //            ExcMessage("max number attempts arrived, line search failed!"));
    // Instead of terminating the program, we can just take a full step.
    if (i == max_iter)
      alpha = 1.0;

    return alpha;
  }

  template <typename LATraits, typename Tria>
  double PhaseFieldMonolithicSolve<LATraits, Tria>::
    line_search_zoom_strong_wolfe(double phi_low, double phi_low_prime, double alpha_low,
				  double phi_high, double phi_high_prime, double alpha_high,
				  double phi_0, double phi_0_prime, const BVector & BFGS_p_vector,
				  double c1, double c2, unsigned int max_iter, const BVector & solution_delta)
  {
    double alpha = 0;
    std::pair<double, double> current_phi_phi_prime;
    double phi, phi_prime;

    unsigned int i = 0;
    for (; i < max_iter; ++i)
      {
	// a simple bisection is faster than cubic interpolation
	alpha = 0.5 * (alpha_low + alpha_high);
	//alpha = line_search_interpolation_cubic(alpha_low, phi_low, phi_low_prime,
	//					alpha_high, phi_high, phi_high_prime);
	current_phi_phi_prime = calculate_phi_and_phi_prime(alpha, BFGS_p_vector, solution_delta);
	phi = current_phi_phi_prime.first;
	phi_prime = current_phi_phi_prime.second;

	if (   (phi > phi_0 + c1 * alpha * phi_0_prime)
	    || (phi > phi_low) )
	  {
	    alpha_high = alpha;
	    phi_high = phi;
	    phi_high_prime = phi_prime;
	  }
	else
	  {
	    if (std::fabs(phi_prime) <= c2 * std::fabs(phi_0_prime))
	      {
		//if (alpha < 1.0e-3)
		//  alpha = 1.0e-3;
		return alpha;
	      }

	    if (phi_prime * (alpha_high - alpha_low) >= 0.0)
	      {
		alpha_high = alpha_low;
		phi_high_prime = phi_low_prime;
		phi_high = phi_low;
	      }

	    alpha_low = alpha;
	    phi_low_prime = phi_prime;
	    phi_low = phi;
	  }
      }

    if (alpha < 1.0e-3)
      alpha = 1.0;

    // avoid unused variable warnings from compiler
    (void)phi_high;
    (void)phi_high_prime;
    return alpha;
  }

  template <typename LATraits, typename Tria>
  double PhaseFieldMonolithicSolve<LATraits, Tria>::line_search_stepsize_residual_projection(const double f0,
				                                                  const BVector & BFGS_p_vector,
				                                                  const BVector & solution_delta)
  {
    // We want to find an alpha such that |f(alpha)| <= eta * |f(0)|
    // This value is suggested by Abaqus line search process
    double eta = 1.0;
    const double threshold = std::fabs(eta * f0);

    unsigned int n_sample_points = 100;

    double alpha = 1.0;
    double f1 = calculate_phi_prime(alpha, BFGS_p_vector, solution_delta);

    double f_alpha = 0.0;
    double sample_point = 0.0;

    if (std::fabs(f1) <= threshold)
      {
	//m_logfile << "f0 = " << f0 << std::endl;
	//m_logfile << "f1 = " << f1 << std::endl;
        return 1.0;
      }

    // f0 and f1 have opposite signs
    if (f0 * f1 <= 0.0)
      {
	// there exists an alpha in [0, 1] such that p^T * r(alpha) = 0
	// we can do a bisection search
	alpha = binary_search(0.0, 1.0, f0, f1, threshold,
			      BFGS_p_vector, solution_delta);
	return alpha;
      }
    // f0 and f1 have the same sign
    else
      {
	for (unsigned int i = 1; i < n_sample_points; ++i)
	  {
	    sample_point = 1.0 - i * 1.0/n_sample_points;
	    f_alpha = calculate_phi_prime(sample_point, BFGS_p_vector, solution_delta);
	    if (std::fabs(f_alpha) <= threshold)
	      {
		//m_logfile << "f0 = " << f0 << std::endl;
		//m_logfile << "f1 = " << f_alpha << std::endl;
	        return sample_point;
	      }

	    if (f0 * f_alpha <= 0.0)
	      {
		// there exists an alpha such that p^T * r(alpha) = 0
		// we can do a bisection search
		alpha = binary_search(0.0, sample_point, f0, f_alpha, threshold,
				      BFGS_p_vector, solution_delta);
		return alpha;
	      }
	  }
      }

    // if the code reaches here, it means that the line search failed
    // to find an alpha such that |f(alpha)| <= eta * |f(0)|
    return 1.01;
  }

  template <typename LATraits, typename Tria>
  double PhaseFieldMonolithicSolve<LATraits, Tria>::binary_search(double a, double b,
						       double fa, double fb,
						       const double threshold,
						       const BVector & BFGS_p_vector,
						       const BVector & solution_delta)
  {
    double m = 0.5* (a + b);
    double fm = calculate_phi_prime(m, BFGS_p_vector, solution_delta);

    while (std::fabs(fm) > threshold )
      {
	if (fm * fa <= 0)
	  {
	    b = m;
	    fb = fm;
	  }
	else
	  {
	    a = m;
	    fa = fm;
	  }

	m = 0.5* (a + b);
	fm = calculate_phi_prime(m, BFGS_p_vector, solution_delta);
      }

    (void) fb;
    return m;
  }

  template <typename LATraits, typename Tria>
  double PhaseFieldMonolithicSolve<LATraits, Tria>::
    line_search_interpolation_cubic(const double alpha_0, const double phi_0, const double phi_0_prime,
  			            const double alpha_1, const double phi_1, const double phi_1_prime)
  {
    const double d1 = phi_0_prime + phi_1_prime - 3.0 * (phi_0 - phi_1) / (alpha_0 - alpha_1);

    const double temp = d1 * d1 - phi_0_prime * phi_1_prime;

    if (temp < 0.0)
      return 0.5 * (alpha_0 + alpha_1);

    int sign;
    if (alpha_1 > alpha_0)
      sign = 1;
    else
      sign = -1;

    const double d2 = sign * std::sqrt(temp);

    const double alpha = alpha_1 - (alpha_1 - alpha_0)
	               * (phi_1_prime + d2 - d1) / (phi_1_prime - phi_0_prime + 2*d2);

    if (    (alpha_1 > alpha_0)
	 && (alpha > alpha_1 || alpha < alpha_0))
      return 0.5 * (alpha_0 + alpha_1);

    if (    (alpha_0 > alpha_1)
	 && (alpha > alpha_0 || alpha < alpha_1))
      return 0.5 * (alpha_0 + alpha_1);

    return alpha;
  }

  template <typename LATraits, typename Tria>
  std::pair<double, double> PhaseFieldMonolithicSolve<LATraits, Tria>::
    calculate_phi_and_phi_prime(const double alpha,
				const BVector & BFGS_p_vector,
				const BVector & solution_delta)
  {
    // the first component is phi(alpha), the second component is phi_prime(alpha),
    std::pair<double, double> phi_values;

    BVector solution_delta_trial(solution_delta);
    solution_delta_trial.add(alpha, BFGS_p_vector);

    update_qph_incremental(solution_delta_trial, m_solution, false);

        BVector system_rhs(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
        system_rhs.initalize();
    assemble_system_rhs_LBFGS_parallel(m_solution, system_rhs);
    //m_constraints.condense(system_rhs);

    phi_values.first = calculate_energy_functional();
    phi_values.second = system_rhs * BFGS_p_vector;
    return phi_values;
  }

  template <typename LATraits, typename Tria>
  double PhaseFieldMonolithicSolve<LATraits, Tria>::
    calculate_phi_prime(const double alpha,
			const BVector & BFGS_p_vector,
			const BVector & solution_delta)
  {
    // phi_prime(alpha) =  p^T * r(alpha)
    double phi_prime;

    BVector solution_delta_trial(solution_delta);
    solution_delta_trial.add(alpha, BFGS_p_vector);

    update_qph_incremental(solution_delta_trial, m_solution, false);

//    BVector system_rhs(m_dofs_per_block);
        BVector system_rhs(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
        system_rhs.initalize();
    assemble_system_rhs_LBFGS_parallel(m_solution, system_rhs);
    //m_constraints.condense(system_rhs);

    //phi_prime = system_rhs * BFGS_p_vector;

//    BVector error_res(m_dofs_per_block);
        BVector error_res(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
        error_res.initalize();

    for (unsigned int i = 0; i < m_dof_handler.n_dofs(); ++i)
      if (!m_constraints.is_constrained(i))
        error_res(i) = system_rhs(i);

    phi_prime = error_res.l2_norm();

    return phi_prime;
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::LBFGS_B0(BVector & LBFGS_r_vector,
						const BVector & LBFGS_q_vector)
  {
      m_timer.enter_subsection("Solve B0");
      
      assemble_system_B0();
      
      m_solver.solve(LBFGS_r_vector, LBFGS_q_vector, m_tangent_matrix);
      
      m_timer.leave_subsection();
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::print_conv_header_LBFGS()
  {
    static const unsigned int l_width = 140;
    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;

    m_logfile << "                  SOLVER STEP (LBFGS)  "
              << " |  LS-alpha     Energy      Res_Norm    "
              << " Res_u      Res_d      Res_t    Inc_Norm   "
              << " Inc_u      Inc_d      Inc_t" << std::endl;

    m_logfile << '\t' << '\t';
    for (unsigned int i = 0; i < l_width; ++i)
      m_logfile << '_';
    m_logfile << std::endl;
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::
  solve_nonlinear_timestep_LBFGS(BVector & solution_delta,
				 BVector & LBFGS_update_refine)
  {
    BVector LBFGS_update(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
      LBFGS_update.initalize();
//    LBFGS_update = 0.0;

    m_error_residual.reset();
    m_error_residual_0.reset();
    m_error_residual_norm.reset();
    m_error_update.reset();
    m_error_update_0.reset();
    m_error_update_norm.reset();

    if (m_parameters.m_output_iteration_history)
      print_conv_header_LBFGS();

    unsigned int LBFGS_iteration = 0;

    BVector LBFGS_r_vector(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
      LBFGS_r_vector.initalize();
    BVector LBFGS_y_vector(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
      LBFGS_y_vector.initalize();
    BVector LBFGS_q_vector(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
      LBFGS_q_vector.initalize();
    BVector LBFGS_s_vector(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
      LBFGS_s_vector.initalize();
    std::list<std::pair< std::pair<BVector,
                                   BVector>,
                         double>> LBFGS_vector_list;

    const unsigned int LBFGS_m = m_parameters.m_LBFGS_m;
    std::list<double> LBFGS_alpha_list;

    double line_search_parameter = 0.0;
    double LBFGS_beta = 0.0;
    double rho = 0.0;

    for (; LBFGS_iteration < m_parameters.m_max_iterations_LBFGS; ++LBFGS_iteration)
      {
	if (m_parameters.m_output_iteration_history)
	  m_logfile << '\t' << '\t' << std::setw(2) << LBFGS_iteration << ' '
                    << std::flush;

        make_constraints(LBFGS_iteration);

        // At the first step, we simply distribute the inhomogeneous part of
        // the constraints
        if (LBFGS_iteration == 0)
          {
            // use the solution from the previous solve on the
            // refined mesh as initial guess
            LBFGS_update = LBFGS_update_refine;

            m_constraints.distribute(LBFGS_update.base());
            solution_delta += LBFGS_update;
            if (m_parameters.m_output_iteration_history)
              {
                m_logfile << " --- " << std::flush;
                m_logfile << " --- " << std::flush;
              }
            update_qph_incremental(solution_delta, m_solution, false);
            if (m_parameters.m_output_iteration_history)
              {
                m_logfile << " ---  |" << std::flush;
                m_logfile << std::endl;
              }
            continue;
          }
        else if (LBFGS_iteration == 1)
          {
	    // Calculate the residual vector r. NOTICE that in the context of
	    // BFGS, this r is the gradient of the energy functional (objective function),
	    // NOT the negative gradient of the energy functional
	    assemble_system_rhs_LBFGS_parallel(m_solution, m_system_rhs);

	    // We cannot simply zero out the dofs that are constrained, since we might
	    // have hanging node constraints. In this case, we need to modify the RHS
	    // as C^T * b, which C contains entries of 0.5 (x_3 = 0.5*x_1 + 0.5*x_2)
	    //for (unsigned int i = 0; i < m_dof_handler.n_dofs(); ++i)
	      //if (m_constraints.is_constrained(i))
		//m_system_rhs(i) = 0.0;

	    // if m_constraints has inhomogeneity, we cannot call m_constraints.condense(m_system_rhs),
	    // since the m_system_matrix needs to be provided to modify the RHS properly. However, this
	    // error will not be detected in the release mode and only will be detected on the debug mode
	    // if we use assemble_system_rhs_LBFGS_parallel, then condense() is not necessary
	    //m_constraints.condense(m_system_rhs);
          }
	if (m_parameters.m_output_iteration_history)
	  {
            m_logfile << " --- " << std::flush;
            m_logfile << " --- " << std::flush;
            m_logfile << " --- " << std::flush;
	  }

        get_error_residual(m_error_residual);
        if (LBFGS_iteration == 1)
          m_error_residual_0 = m_error_residual;

        m_error_residual_norm = m_error_residual;
        // For three-point bending problem and 3D problem, we use absolute residual
        // for convergence test
        if (m_parameters.m_relative_residual)
          m_error_residual_norm.normalize(m_error_residual_0);

        if (LBFGS_iteration > 1 && m_error_update_norm.m_u <= m_parameters.m_tol_u_incr
                                && m_error_residual_norm.m_u <= m_parameters.m_tol_u_residual
			        && m_error_update_norm.m_d <= m_parameters.m_tol_d_incr
			        && m_error_residual_norm.m_d <= m_parameters.m_tol_d_residual
			        && m_error_update_norm.m_t <= m_parameters.m_tol_t_incr
			        && m_error_residual_norm.m_t <= m_parameters.m_tol_t_residual
				)
          {
            if (m_parameters.m_output_iteration_history)
              {
		m_logfile << " | ";
		m_logfile << " CONVERGED! " << std::fixed << std::setprecision(3) << std::setw(7)
			  << std::scientific
		      << "    ----    "
		      << "  " << m_error_residual_norm.m_norm
		      << "  " << m_error_residual_norm.m_u
		      << "  " << m_error_residual_norm.m_d
		      << "  " << m_error_residual_norm.m_t
		      << "  " << m_error_update_norm.m_norm
		      << "  " << m_error_update_norm.m_u
		      << "  " << m_error_update_norm.m_d
		      << "  " << m_error_update_norm.m_t
		      << "  " << std::endl;

		m_logfile << '\t' << '\t';
		for (unsigned int i = 0; i < 140; ++i)
		  m_logfile << '_';
		m_logfile << std::endl;
              }

            m_logfile << "\t\tConvergence is reached after "
        	      << LBFGS_iteration << " L-BFGS iterations."<< std::endl;

            m_logfile << "\t\tResidual information of convergence:" << std::endl;

            if (m_parameters.m_relative_residual)
              {
		m_logfile << "\t\t\tRelative residual of disp. equation: "
			  << m_error_residual_norm.m_u << std::endl;

		m_logfile << "\t\t\tAbsolute residual of disp. equation: "
			  << m_error_residual_norm.m_u * m_error_residual_0.m_u << std::endl;

		m_logfile << "\t\t\tRelative residual of phasefield equation: "
			  << m_error_residual_norm.m_d << std::endl;

		m_logfile << "\t\t\tAbsolute residual of phasefield equation: "
			  << m_error_residual_norm.m_d * m_error_residual_0.m_d << std::endl;

		m_logfile << "\t\t\tRelative residual of temperature equation: "
			  << m_error_residual_norm.m_t << std::endl;

		m_logfile << "\t\t\tAbsolute residual of temperature equation: "
			  << m_error_residual_norm.m_t * m_error_residual_0.m_t << std::endl;

		m_logfile << "\t\t\tRelative increment of disp.: "
			  << m_error_update_norm.m_u << std::endl;

		m_logfile << "\t\t\tAbsolute increment of disp.: "
			  << m_error_update_norm.m_u * m_error_update_0.m_u << std::endl;

		m_logfile << "\t\t\tRelative increment of phasefield: "
			  << m_error_update_norm.m_d << std::endl;

		m_logfile << "\t\t\tAbsolute increment of phasefield: "
			  << m_error_update_norm.m_d * m_error_update_0.m_d << std::endl;

		m_logfile << "\t\t\tRelative increment of temperature: "
			  << m_error_update_norm.m_t << std::endl;

		m_logfile << "\t\t\tAbsolute increment of temperature: "
			  << m_error_update_norm.m_t * m_error_update_0.m_t << std::endl;
              }
            else
              {
		m_logfile << "\t\t\tAbsolute residual of disp. equation: "
			  << m_error_residual_norm.m_u << std::endl;

		m_logfile << "\t\t\tAbsolute residual of phasefield equation: "
			  << m_error_residual_norm.m_d << std::endl;

		m_logfile << "\t\t\tAbsolute residual of temperature equation: "
			  << m_error_residual_norm.m_t << std::endl;

		m_logfile << "\t\t\tAbsolute increment of disp.: "
			  << m_error_update_norm.m_u << std::endl;

		m_logfile << "\t\t\tAbsolute increment of phasefield: "
			  << m_error_update_norm.m_d << std::endl;

		m_logfile << "\t\t\tAbsolute increment of temperature: "
			  << m_error_update_norm.m_t << std::endl;
              }

            break;
          }

        // LBFGS algorithm
        LBFGS_q_vector = m_system_rhs;

        LBFGS_alpha_list.clear();
        for (auto itr = LBFGS_vector_list.begin(); itr != LBFGS_vector_list.end(); ++itr)
          {
            LBFGS_s_vector = (itr->first).first;
            LBFGS_y_vector = (itr->first).second;
            rho = itr->second;

            const double alpha = rho * (LBFGS_s_vector * LBFGS_q_vector);
            LBFGS_alpha_list.push_back(alpha);

            LBFGS_q_vector.add(-alpha, LBFGS_y_vector);
          }
/*
        double scale_gamma = 0.0;
        if (LBFGS_iteration == 1)
          {
            scale_gamma = 1.0;
          }
        else
          {
            LBFGS_s_vector = LBFGS_vector_list.front().first.first;
            LBFGS_y_vector = LBFGS_vector_list.front().first.second;
            scale_gamma = (LBFGS_s_vector * LBFGS_y_vector)/(LBFGS_y_vector * LBFGS_y_vector);
          }

        LBFGS_q_vector *= scale_gamma;
        LBFGS_r_vector = LBFGS_q_vector;
*/
        LBFGS_B0(LBFGS_r_vector,
		 LBFGS_q_vector);

        for (auto itr = LBFGS_vector_list.rbegin(); itr != LBFGS_vector_list.rend(); ++itr)
          {
            LBFGS_s_vector = (itr->first).first;
            LBFGS_y_vector = (itr->first).second;
            rho = itr->second;

            LBFGS_beta = rho * (LBFGS_y_vector * LBFGS_r_vector);

            const double alpha = LBFGS_alpha_list.back();
            LBFGS_alpha_list.pop_back();

            LBFGS_r_vector.add(alpha - LBFGS_beta, LBFGS_s_vector);
          }

        LBFGS_r_vector *= -1.0; // this is the p_vector (search direction)

        m_constraints.distribute(LBFGS_r_vector.base());

        // We need a line search algorithm to decide line_search_parameter

        line_search_parameter = line_search_stepsize_gradient_based(LBFGS_r_vector,
        							    solution_delta);
        // const double phi_0 = calculate_energy_functional();
        // const double phi_0_prime = m_system_rhs * LBFGS_r_vector;
/*
        BlockVector<double> error_res(m_dofs_per_block);

        for (unsigned int i = 0; i < m_dof_handler.n_dofs(); ++i)
          if (!m_constraints.is_constrained(i))
            error_res(i) = m_system_rhs(i);

        const double phi_0_prime = error_res.l2_norm();
*/
/*
        line_search_parameter = line_search_stepsize_strong_wolfe(phi_0,
						                  phi_0_prime,
								  LBFGS_r_vector,
						                  solution_delta);
*/

        // phi_0_prime is p^T * r (dot product between residual and search direction)
        // LBFGS_r_vector is the search direction
        //line_search_parameter = line_search_stepsize_residual_projection(phi_0_prime,
	//								 LBFGS_r_vector,
	//								 solution_delta);

        // line_search_parameter = 1.0;
        LBFGS_r_vector *= line_search_parameter;
        LBFGS_update = LBFGS_r_vector;

        get_error_update(LBFGS_update, m_error_update);
        if (LBFGS_iteration == 1)
          m_error_update_0 = m_error_update;

        m_error_update_norm = m_error_update;
        // For three-point bending problem and the sphere inclusion problem,
        // we use absolute residual for convergence test
        if (m_parameters.m_relative_residual)
          m_error_update_norm.normalize(m_error_update_0);

        solution_delta += LBFGS_update;
        update_qph_incremental(solution_delta, m_solution, false);

        LBFGS_y_vector = m_system_rhs;
        LBFGS_y_vector *= -1.0;
        assemble_system_rhs_LBFGS_parallel(m_solution, m_system_rhs);
        // if we use assemble_system_rhs_LBFGS_parallel, then condense() is not necessary
        //m_constraints.condense(m_system_rhs);
        LBFGS_y_vector += m_system_rhs;

        LBFGS_s_vector = LBFGS_update;
/*
        if (LBFGS_iteration > LBFGS_m)
          LBFGS_vector_list.pop_back();

        rho = 1.0 / (LBFGS_y_vector * LBFGS_s_vector);

        LBFGS_vector_list.push_front(std::make_pair(std::make_pair(LBFGS_s_vector,
								   LBFGS_y_vector),
						    rho));
*/
        const double g_norm = m_system_rhs.l2_norm();

        const double yxs = LBFGS_y_vector * LBFGS_s_vector;

        const double sxs = LBFGS_s_vector * LBFGS_s_vector;

        if (yxs/sxs >= 1.0e-6 * g_norm)
          {
	    if (LBFGS_iteration > LBFGS_m)
	      LBFGS_vector_list.pop_back();

	    rho = 1.0 / yxs;

	    LBFGS_vector_list.push_front(std::make_pair(std::make_pair(LBFGS_s_vector,
								       LBFGS_y_vector),
							rho));
          }

        if (m_parameters.m_output_iteration_history)
          {
	    const double energy_functional = calculate_energy_functional();

	    m_logfile << " | " << std::fixed << std::setprecision(3) << std::setw(1)
		      << std::scientific
		      << "" << line_search_parameter
		      << std::fixed << std::setprecision(6) << std::setw(1)
					<< std::scientific
		      << "  " << energy_functional
		      << std::fixed << std::setprecision(3) << std::setw(1)
					<< std::scientific
		      << "  " << m_error_residual_norm.m_norm
		      << "  " << m_error_residual_norm.m_u
		      << "  " << m_error_residual_norm.m_d
		      << "  " << m_error_residual_norm.m_t
		      << "  " << m_error_update_norm.m_norm
		      << "  " << m_error_update_norm.m_u
		      << "  " << m_error_update_norm.m_d
		      << "  " << m_error_update_norm.m_t
		      << "  " << std::endl;
          }
      }

    AssertThrow(LBFGS_iteration < m_parameters.m_max_iterations_LBFGS,
                ExcMessage("No convergence in L-BFGS nonlinear solver!"));
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::output_results() const
  {
    m_timer.enter_subsection("Output results");

      m_output.output(m_time.get_timestep(),
                      m_parameters.m_poly_degree,
                      m_parameters.resultsDir,
                      m_solution,
                      m_quadrature_point_history);
      
      // TODO: 
//      if constexpr (!is_mpi) {
//          
//          DataOut<dim> data_out;
//          
//          std::vector<DataComponentInterpretation::DataComponentInterpretation>
//          data_component_interpretation(
//                                        dim, DataComponentInterpretation::component_is_part_of_vector);
//          
//          data_component_interpretation.push_back(
//                                                  DataComponentInterpretation::component_is_scalar);
//          
//          data_component_interpretation.push_back(
//                                                  DataComponentInterpretation::component_is_scalar);
//          
//          std::vector<std::string> solution_name(dim, "displacement");
//          solution_name.emplace_back("phasefield");
//          solution_name.emplace_back("temperature");
//          
//          data_out.attach_dof_handler(m_dof_handler);
//          data_out.add_data_vector(m_solution.base(),
//                                   solution_name,
//                                   DataOut<dim>::type_dof_data,
//                                   data_component_interpretation);
//          
//          Vector<double> cell_material_id(m_triangulation.n_active_cells());
//          // output material ID for each cell
//          for (const auto &cell : m_triangulation.active_cell_iterators())
//          {
//              cell_material_id(cell->active_cell_index()) = cell->material_id();
//          }
//          data_out.add_data_vector(cell_material_id, "materialID");
//          
//          //L2 projection
//          DoFHandler<dim> dof_handler_L2(m_triangulation);
//          FE_Q<dim>     fe_L2(m_parameters.m_poly_degree); //FE_Q element is continuous
//          dof_handler_L2.distribute_dofs(fe_L2);
//          AffineConstraints<double> constraints;
//          constraints.clear();
//          DoFTools::make_hanging_node_constraints(dof_handler_L2, constraints);
//          constraints.close();
//          std::vector<DataComponentInterpretation::DataComponentInterpretation>
//          data_component_interpretation_L2(1,
//                                           DataComponentInterpretation::component_is_scalar);
//          
//          //stress L2 projection
//          for (unsigned int i = 0; i < dim; ++i)
//              for (unsigned int j = i; j < dim; ++j)
//              {
//                  Vector<double> stress_field_L2;
//                  stress_field_L2.reinit(dof_handler_L2.n_dofs());
//                  
//                  MappingQ<dim> mapping(m_parameters.m_poly_degree + 1);
//                  VectorTools::project(mapping,
//                                       dof_handler_L2,
//                                       constraints,
//                                       m_qf_cell,
//                                       [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
//                                            const unsigned int q) -> double
//                                       {
//                      return m_quadrature_point_history.get_data(cell)[q]->get_cauchy_stress()[i][j];
//                  },
//                                       stress_field_L2);
//                  
//                  std::string stress_name = "Cauchy_stress_" + std::to_string(i+1) + std::to_string(j+1)
//                  + "_L2";
//                  
//                  data_out.add_data_vector(dof_handler_L2,
//                                           stress_field_L2,
//                                           stress_name,
//                                           data_component_interpretation_L2);
//              }
//          
//          // Heat flux L2 projection
//          Vector<double> heat_flux_field_L2_x;
//          Vector<double> heat_flux_field_L2_y;
//          Vector<double> heat_flux_field_L2_z;
//          
//          for (unsigned int i = 0; i < dim; ++i)
//          {
//              Vector<double> heat_flux_field_L2;
//              heat_flux_field_L2.reinit(dof_handler_L2.n_dofs());
//              
//              MappingQ<dim> mapping(m_parameters.m_poly_degree + 1);
//              VectorTools::project(mapping,
//                                   dof_handler_L2,
//                                   constraints,
//                                   m_qf_cell,
//                                   [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
//                                        const unsigned int q) -> double
//                                   {
//                  return m_quadrature_point_history.get_data(cell)[q]->get_heat_flux()[i];
//              },
//                                   heat_flux_field_L2);
//              
//              std::string heat_flux_name = "Heat_flux_" + std::to_string(i+1) + "_L2";
//              
//              data_out.add_data_vector(dof_handler_L2,
//                                       heat_flux_field_L2,
//                                       heat_flux_name,
//                                       data_component_interpretation_L2);
//              if (i == 0)
//                  heat_flux_field_L2_x = heat_flux_field_L2;
//              else if (i == 1)
//                  heat_flux_field_L2_y = heat_flux_field_L2;
//              else if (i == 2)
//                  heat_flux_field_L2_z = heat_flux_field_L2;
//              else
//                  AssertThrow(false,
//                              ExcMessage("Heat flux output is wrong!"));
//          }
//          
//          // For 2D problems, let the flux in the third direction is zero
//          if (dim == 2)
//          {
//              Vector<double> heat_flux_field_L2;
//              heat_flux_field_L2.reinit(dof_handler_L2.n_dofs());
//              heat_flux_field_L2 = 0;
//              std::string heat_flux_name = "Heat_flux_" + std::to_string(3) + "_L2";
//              data_out.add_data_vector(dof_handler_L2,
//                                       heat_flux_field_L2,
//                                       heat_flux_name,
//                                       data_component_interpretation_L2);
//              heat_flux_field_L2_z = 0;
//          }
//          
//          DoFHandler<dim> dof_handler_L2_flux(m_triangulation);
//          FESystem<dim>   fe_flux_L2(FE_Q<dim>(m_parameters.m_poly_degree), dim);
//          dof_handler_L2_flux.distribute_dofs(fe_flux_L2);
//          std::vector<DataComponentInterpretation::DataComponentInterpretation>
//          data_component_interpretation_flux_L2(dim,
//                                                DataComponentInterpretation::component_is_part_of_vector);
//          
//          Vector<double> heat_flux_field_L2;
//          heat_flux_field_L2.reinit(dof_handler_L2_flux.n_dofs());
//          
//          if (dim == 2)
//          {
//              for (unsigned int i = 0; i < heat_flux_field_L2_x.size(); ++i)
//              {
//                  heat_flux_field_L2(0+i*dim) = heat_flux_field_L2_x(i);
//                  heat_flux_field_L2(1+i*dim) = heat_flux_field_L2_y(i);
//              }
//          }
//          
//          if (dim == 3)
//          {
//              for (unsigned int i = 0; i < heat_flux_field_L2_x.size(); ++i)
//              {
//                  heat_flux_field_L2(0+i*dim) = heat_flux_field_L2_x(i);
//                  heat_flux_field_L2(1+i*dim) = heat_flux_field_L2_y(i);
//                  heat_flux_field_L2(2+i*dim) = heat_flux_field_L2_z(i);
//              }
//          }
//          
//          std::vector<std::string> solution_name_flux(dim, "Heat_flux_vector");
//          data_out.add_data_vector(dof_handler_L2_flux,
//                                   heat_flux_field_L2,
//                                   solution_name_flux,
//                                   data_component_interpretation_flux_L2);
//          
//          data_out.build_patches(m_parameters.m_poly_degree);
//          
//          std::ofstream output(m_parameters.resultsDir + "Solution-" + std::to_string(dim) + "d-" +
//                               Utilities::int_to_string(m_time.get_timestep(),4) + ".vtu");
//          
//          data_out.write_vtu(output);
//      } else {
//          
//      }
      
      
    m_timer.leave_subsection();
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::calculate_reaction_force(unsigned int face_ID)
  {
    m_timer.enter_subsection("Calculate reaction force");

    BVector       system_rhs(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
//    system_rhs.reinit(m_dofs_per_block);
      system_rhs.initalize();

    Vector<double> cell_rhs(m_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(m_dofs_per_cell);

    const double time_ramp = (m_time.current() / m_time.end());
    std::vector<Tensor<1, dim>> rhs_values(m_n_q_points);
    const UpdateFlags uf_cell(update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    const UpdateFlags uf_face(update_values | update_normal_vectors |
                              update_JxW_values);

    FEValues<dim> fe_values(m_fe, m_qf_cell, uf_cell);
    FEFaceValues<dim> fe_face_values(m_fe, m_qf_face, uf_face);

    // shape function values for displacement field
    std::vector<std::vector<Tensor<1, dim>>>
      Nx(m_qf_cell.size(), std::vector<Tensor<1, dim>>(m_dofs_per_cell));
    std::vector<std::vector<Tensor<2, dim>>>
      grad_Nx(m_qf_cell.size(), std::vector<Tensor<2, dim>>(m_dofs_per_cell));
    std::vector<std::vector<SymmetricTensor<2, dim>>>
      symm_grad_Nx(m_qf_cell.size(), std::vector<SymmetricTensor<2, dim>>(m_dofs_per_cell));

      // TODO:
    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
          // skip cells owned by other ranks in mpi mode
          if constexpr (is_mpi)
              if (!cell->is_locally_owned())
                  continue;
          
	// if calculate_reaction_force() is defined as const, then
	// we also need to put a const in std::shared_ptr,
	// that is, std::shared_ptr<const PointHistory<dim>>
	const std::vector<std::shared_ptr< PointHistory<dim>>> lqph =
	  m_quadrature_point_history.get_data(cell);
	Assert(lqph.size() == m_n_q_points, ExcInternalError());
        cell_rhs = 0.0;
        fe_values.reinit(cell);
        right_hand_side(fe_values.get_quadrature_points(),
    		        rhs_values,
    		        m_parameters.m_x_component*1.0,
    		        m_parameters.m_y_component*1.0,
    		        m_parameters.m_z_component*1.0);

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            for (const unsigned int k : fe_values.dof_indices())
              {
                const unsigned int k_group = m_fe.system_to_base_index(k).first.first;

                if (k_group == m_u_dof)
                  {
    		    Nx[q_point][k] = fe_values[m_u_fe].value(k, q_point);
    		    grad_Nx[q_point][k] = fe_values[m_u_fe].gradient(k, q_point);
    		    symm_grad_Nx[q_point][k] = symmetrize(grad_Nx[q_point][k]);
                  }
              }
          }

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            const SymmetricTensor<2, dim> & cauchy_stress = lqph[q_point]->get_cauchy_stress();

            const std::vector<Tensor<1,dim>> & N = Nx[q_point];
            const std::vector<SymmetricTensor<2, dim>> & symm_grad_N = symm_grad_Nx[q_point];
            const double JxW = fe_values.JxW(q_point);

            for (const unsigned int i : fe_values.dof_indices())
              {
                const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

                if (i_group == m_u_dof)
                  {
                    cell_rhs(i) -= (symm_grad_N[i] * cauchy_stress) * JxW;
    		    // contributions from the body force to right-hand side
    		    cell_rhs(i) += N[i] * rhs_values[q_point] * JxW;
                  }
              }
          }

        // if there is surface pressure, this surface pressure always applied to the
        // reference configuration
        const unsigned int face_pressure_id = 100;
        const double p0 = 0.0;

        for (const auto &face : cell->face_iterators())
          {
	    if (face->at_boundary() && face->boundary_id() == face_pressure_id)
	      {
		fe_face_values.reinit(cell, face);

		for (const unsigned int f_q_point : fe_face_values.quadrature_point_indices())
		  {
		    const Tensor<1, dim> &N = fe_face_values.normal_vector(f_q_point);

		    const double         pressure  = p0 * time_ramp;
		    const Tensor<1, dim> traction  = pressure * N;

		    for (const unsigned int i : fe_values.dof_indices())
		      {
			const unsigned int i_group = m_fe.system_to_base_index(i).first.first;

			if (i_group == m_u_dof)
			  {
			    const unsigned int component_i = m_fe.system_to_component_index(i).first;
			    const double Ni = fe_face_values.shape_value(i, f_q_point);
			    const double JxW = fe_face_values.JxW(f_q_point);
			    cell_rhs(i) += (Ni * traction[component_i]) * JxW;
			  }
		      }
		  }
	      }
          }

        cell->get_dof_indices(local_dof_indices);
        for (const unsigned int i : fe_values.dof_indices())
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
      } // for (const auto &cell : m_dof_handler.active_cell_iterators())

    // The difference between the above assembled system_rhs and m_system_rhs
    // is that m_system_rhs is condensed by the m_constraints, which zero out
    // the rhs values associated with the constrained DOFs and modify the rhs
    // values associated with the unconstrained DOFs.

    std::vector< types::global_dof_index > mapping;
    std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert(face_ID);
    DoFTools::map_dof_to_boundary_indices(m_dof_handler,
					  boundary_ids,
					  mapping);

    std::vector<double> reaction_force(dim, 0.0);

    for (unsigned int i = 0; i < m_dofs_per_block[m_u_dof]; ++i)
      {
	if (mapping[i] != numbers::invalid_dof_index)
	  {
	    reaction_force[i % dim] += system_rhs.block(m_u_dof)(i);
	  }
      }

    for (unsigned int i = 0; i < dim; i++)
      m_logfile << "\t\tReaction force in direction " << i << " on boundary ID " << face_ID
                << " = "
		<< std::fixed << std::setprecision(3) << std::setw(1)
                << std::scientific
		<< reaction_force[i] << std::endl;

    std::pair<double, std::vector<double>> time_force;
    time_force.first = m_time.current();
    time_force.second = reaction_force;
    m_history_reaction_force.push_back(time_force);

    m_timer.leave_subsection();
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::write_history_data()
  {
    m_logfile << "\t\tWrite history data ... \n"<<std::endl;

    std::ofstream myfile_reaction_force (m_parameters.histDir + "Reaction_force.hist");
    if (myfile_reaction_force.is_open())
    {
      myfile_reaction_force << 0.0 << "\t";
      if (dim == 2)
	myfile_reaction_force << 0.0 << "\t"
	       << 0.0 << std::endl;
      if (dim == 3)
	myfile_reaction_force << 0.0 << "\t"
	       << 0.0 << "\t"
	       << 0.0 << std::endl;

      for (auto const & time_force : m_history_reaction_force)
	{
	  myfile_reaction_force << time_force.first << "\t";
	  if (dim == 2)
	    myfile_reaction_force << time_force.second[0] << "\t"
	           << time_force.second[1] << std::endl;
	  if (dim == 3)
	    myfile_reaction_force << time_force.second[0] << "\t"
	           << time_force.second[1] << "\t"
		   << time_force.second[2] << std::endl;
	}
      myfile_reaction_force.close();
    }
    else
      m_logfile << "Unable to open file";

    std::ofstream myfile_energy (m_parameters.histDir + "Energy.hist");
    if (myfile_energy.is_open())
    {
      myfile_energy << std::fixed << std::setprecision(10) << std::scientific
                    << 0.0 << "\t"
                    << 0.0 << "\t"
	            << 0.0 << "\t"
	            << 0.0 << std::endl;

      for (auto const & time_energy : m_history_energy)
	{
	  myfile_energy << std::fixed << std::setprecision(10) << std::scientific
	                << time_energy.first     << "\t"
                        << time_energy.second[0] << "\t"
	                << time_energy.second[1] << "\t"
		        << time_energy.second[2] << std::endl;
	}
      myfile_energy.close();
    }
    else
      m_logfile << "Unable to open file";
  }

  template <typename LATraits, typename Tria>
  double PhaseFieldMonolithicSolve<LATraits, Tria>::calculate_energy_functional() const
  {
    double energy_functional = 0.0;

    FEValues<dim> fe_values(m_fe, m_qf_cell, update_JxW_values);

    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
          // skip cells owned by other ranks in mpi mode
          if constexpr (is_mpi)
              if (!cell->is_locally_owned())
                  continue;
          
        fe_values.reinit(cell);

        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            const double JxW = fe_values.JxW(q_point);
            energy_functional += lqph[q_point]->get_total_strain_energy() * JxW;
            energy_functional += lqph[q_point]->get_crack_energy_dissipation() * JxW;
          }
      }

      // sync energy_functional with all ranks
      if constexpr (is_mpi)
          energy_functional = Utilities::MPI::sum(energy_functional,
                                                  *m_mpiInfo.mpiCommPtr());
    return energy_functional;
  }

  template <typename LATraits, typename Tria>
  std::pair<double, double>
    PhaseFieldMonolithicSolve<LATraits, Tria>::calculate_total_strain_energy_and_crack_energy_dissipation() const
  {
    double total_strain_energy = 0.0;
    double crack_energy_dissipation = 0.0;

    FEValues<dim> fe_values(m_fe, m_qf_cell, update_JxW_values);

    for (const auto &cell : m_dof_handler.active_cell_iterators())
      {
          
          // skip cells owned by other ranks in mpi mode
          if constexpr (is_mpi)
              if (!cell->is_locally_owned())
                  continue;
        fe_values.reinit(cell);

        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          m_quadrature_point_history.get_data(cell);
        Assert(lqph.size() == m_n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < m_n_q_points; ++q_point)
          {
            const double JxW = fe_values.JxW(q_point);
            total_strain_energy += lqph[q_point]->get_total_strain_energy() * JxW;
            crack_energy_dissipation += lqph[q_point]->get_crack_energy_dissipation() * JxW;
          }
      }

      // sync total_strain_energy and crack_energy_dissipation with all ranks
      if constexpr (is_mpi) {
          total_strain_energy = Utilities::MPI::sum(total_strain_energy,
                                                    *m_mpiInfo.mpiCommPtr());
          crack_energy_dissipation = Utilities::MPI::sum(crack_energy_dissipation,
                                                         *m_mpiInfo.mpiCommPtr());
      }
      
    return std::make_pair(total_strain_energy, crack_energy_dissipation);
  }


  template <typename LATraits, typename Tria>
  bool PhaseFieldMonolithicSolve<LATraits, Tria>::local_refine_and_solution_transfer(BVector & solution_delta,
									  BVector & LBFGS_update_refine)
  {
    // This is the solution at (n+1) obtained from the old (coarse) mesh
    BVector solution_next_step(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
      solution_next_step.initalize();
    solution_next_step.base() = m_solution.base() + solution_delta.base();
    bool mesh_is_same = true;
    bool cell_refine_flag = true;

    unsigned int material_id;
    double length_scale;
    double cell_length;
    while(cell_refine_flag)
      {
	cell_refine_flag = false;

          // TODO:
	std::vector<types::global_dof_index> local_dof_indices(m_fe.dofs_per_cell);
	for (const auto &cell : m_dof_handler.active_cell_iterators())
	  {
	    cell->get_dof_indices(local_dof_indices);

	    for (unsigned int i = 0; i< m_fe.dofs_per_cell; ++i)
	      {
		const unsigned int comp_i = m_fe.system_to_component_index(i).first;
		if (comp_i == m_d_component) //phasefield component
		  {
		    if (  solution_next_step(local_dof_indices[i])
			> m_parameters.m_phasefield_refine_threshold )
		      {
			material_id = cell->material_id();
	                length_scale = m_material_data[material_id][2];
	                if (dim == 2)
	                  cell_length = std::sqrt(cell->measure());
	                else
	                  cell_length = std::cbrt(cell->measure());
			if (  cell_length
			    > length_scale * m_parameters.m_allowed_max_h_l_ratio )
			  {
			    if (cell->level() < m_parameters.m_max_allowed_refinement_level)
			      {
			        cell->set_refine_flag();
			        break;
			      }
			  }
		      }
		  }
	      }
	  }

          // TODO: 
	for (const auto &cell : m_dof_handler.active_cell_iterators())
	  {
	    if (cell->refine_flag_set())
	      {
		cell_refine_flag = true;
		break;
	      }
	  }

	// if any cell is refined, we need to project the solution
	// to the newly refined mesh
	if (cell_refine_flag)
	  {
	    mesh_is_same = false;
          //          TODO: type confict
          /*
	    std::vector<BVector> old_solutions(2, m_solution);
	    old_solutions[0] = solution_next_step;

	    // history variable field L2 projection
	    DoFHandler<dim> dof_handler_L2(m_triangulation);
	    FE_DGQ<dim>     fe_L2(m_parameters.m_poly_degree); //Discontinuous Galerkin
	    dof_handler_L2.distribute_dofs(fe_L2);
	    AffineConstraints<double> constraints;
	    constraints.clear();
	    DoFTools::make_hanging_node_constraints(dof_handler_L2, constraints);
	    constraints.close();

	    Vector<double> old_history_variable_field_L2;
	    old_history_variable_field_L2.reinit(dof_handler_L2.n_dofs());

	    MappingQ<dim> mapping(m_parameters.m_poly_degree + 1);
	    VectorTools::project(mapping,
			     dof_handler_L2,
			     constraints,
			     m_qf_cell,
			     [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
				  const unsigned int q) -> double
			     {
			       return m_quadrature_point_history.get_data(cell)[q]->get_history_max_positive_strain_energy();
			     },
			     old_history_variable_field_L2);

	    m_triangulation.prepare_coarsening_and_refinement();
        
	    SolutionTransfer<dim, BVector> solution_transfer(m_dof_handler);
	    solution_transfer.prepare_for_coarsening_and_refinement(old_solutions);
	    SolutionTransfer<dim, Vector<double>> solution_transfer_history_variable(dof_handler_L2);
	    solution_transfer_history_variable.prepare_for_coarsening_and_refinement(old_history_variable_field_L2);
	    m_triangulation.execute_coarsening_and_refinement();

	    setup_system();

	    dof_handler_L2.distribute_dofs(fe_L2);
	    constraints.clear();
	    DoFTools::make_hanging_node_constraints(dof_handler_L2, constraints);
	    constraints.close();

        std::vector<BVector> tmp_solutions;
        tmp_solutions.reserve(2);
        tmp_solutions.emplace_back(m_mpiInfo, m_blocks_desc, true);
        tmp_solutions.emplace_back(m_mpiInfo, m_blocks_desc, true);
//	    tmp_solutions[0].reinit(m_dofs_per_block);
//	    tmp_solutions[1].reinit(m_dofs_per_block);
          tmp_solutions[0].initalize();
          tmp_solutions[1].initalize();

	    Vector<double> new_history_variable_field_L2;
	    new_history_variable_field_L2.reinit(dof_handler_L2.n_dofs());

#  if DEAL_II_VERSION_GTE(9, 7, 0)
	    solution_transfer.interpolate(tmp_solutions);
#  else
	    // If an older version of dealII is used, for example, 9.4.0, interpolate()
            // needs to use the following interface.
            solution_transfer.interpolate(old_solutions, tmp_solutions);
#  endif

#  if DEAL_II_VERSION_GTE(9, 7, 0)
            solution_transfer_history_variable.interpolate(new_history_variable_field_L2);
#  else
	    // If an older version of dealII is used, for example, 9.4.0, interpolate()
            // needs to use the following interface.
            solution_transfer_history_variable.interpolate(old_history_variable_field_L2, new_history_variable_field_L2);
#  endif

	    solution_next_step = tmp_solutions[0];
	    m_solution = tmp_solutions[1];

	    // make sure the projected solutions still satisfy
	    // hanging node constraints
	    m_constraints.distribute(solution_next_step);
	    m_constraints.distribute(m_solution);
	    constraints.distribute(new_history_variable_field_L2);

            // new_history_variable_field_L2 contains the history variable projected
            // onto the newly refined mesh
	    FEValues<dim> fe_values(fe_L2,
				    m_qf_cell,
				    update_values | update_gradients |
				    update_quadrature_points | update_JxW_values);

	    for (const auto &cell : dof_handler_L2.active_cell_iterators())
	      {
	        fe_values.reinit(cell);

	        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
	              m_quadrature_point_history.get_data(cell);

	        std::vector<double> history_variable_values_cell(m_n_q_points);

	        fe_values.get_function_values(
	            new_history_variable_field_L2, history_variable_values_cell);

	        for (unsigned int q_point : fe_values.quadrature_point_indices())
	          {
	            lqph[q_point]->assign_history_variable(history_variable_values_cell[q_point]);
	          }
	      }
           */
	  } // if (cell_refine_flag)
      } // while(cell_refine_flag)

    // calculate field variables for newly refined cells
    if (!mesh_is_same)
      {
	BVector temp_solution_delta(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
	BVector temp_previous_solution(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
//	temp_solution_delta = 0.0;
          temp_solution_delta.initalize();
//	temp_previous_solution = 0.0;
          temp_previous_solution.initalize();
	update_qph_incremental(temp_solution_delta, temp_previous_solution, false);
	update_history_field_step();

	// initial guess for the resolve on the refined mesh
	LBFGS_update_refine.base() = solution_next_step.base() - m_solution.base();
      }

    return mesh_is_same;
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::print_parameter_information()
  {
    m_logfile << "Scenario number = " << m_parameters.m_scenario << std::endl;
    m_logfile << "Log file = " << m_parameters.m_logfile_name << std::endl;
    m_logfile << "Write iteration history to log file? = " << std::boolalpha
	      << m_parameters.m_output_iteration_history << std::endl;

    m_logfile << "Does the heat equation contain the coupling term? = " << std::boolalpha
	      << m_parameters.m_coupling_on_heat_eq << std::endl;
    m_logfile << "Is the thermal conductivity degraded by phasefield? = " << std::boolalpha
    	      << m_parameters.m_degrade_conductivity << std::endl;

    m_logfile << "Nonlinear solver type = " << m_parameters.m_type_nonlinear_solver << std::endl;
    m_logfile << "Linear solver type = " << m_parameters.m_type_linear_solver << std::endl;

    if (m_parameters.m_type_linear_solver == "CG")
      {
	m_logfile << "\tCG tolerance for inverse K_uu = "
	          << m_parameters.m_cg_u_tol << std::endl;
	m_logfile << "\tCG tolerance for inverse K_dd = "
		  << m_parameters.m_cg_d_tol << std::endl;
	m_logfile << "\tCG tolerance for inverse K_TT = "
		  << m_parameters.m_cg_t_tol << std::endl;
      }

    m_logfile << "Mesh refinement strategy = " << m_parameters.m_refinement_strategy << std::endl;

    if (m_parameters.m_refinement_strategy == "adaptive-refine")
      {
	m_logfile << "\tMaximum adaptive refinement times allowed in each step = "
		  << m_parameters.m_max_adaptive_refine_times << std::endl;
	m_logfile << "\tMaximum allowed cell refinement level = "
		  << m_parameters.m_max_allowed_refinement_level << std::endl;
	m_logfile << "\tPhasefield-based refinement threshold value = "
		  << m_parameters.m_phasefield_refine_threshold << std::endl;
      }

    m_logfile << "L-BFGS_m = " << m_parameters.m_LBFGS_m << std::endl;
    m_logfile << "Global refinement times = " << m_parameters.m_global_refine_times << std::endl;
    m_logfile << "Local prerefinement times = " <<m_parameters. m_local_prerefine_times << std::endl;
    m_logfile << "Allowed maximum h/l ratio = " << m_parameters.m_allowed_max_h_l_ratio << std::endl;
    m_logfile << "total number of material types = " << m_parameters.m_total_material_regions << std::endl;
    m_logfile << "material data file name = " << m_parameters.m_material_file_name << std::endl;
    if (m_parameters.m_reaction_force_face_id >= 0)
      m_logfile << "Calculate reaction forces on Face ID = " << m_parameters.m_reaction_force_face_id << std::endl;
    else
      m_logfile << "No need to calculate reaction forces." << std::endl;

    if (m_parameters.m_relative_residual)
      m_logfile << "Relative residual for convergence." << std::endl;
    else
      m_logfile << "Absolute residual for convergence." << std::endl;

    m_logfile << "Body force = (" << m_parameters.m_x_component << ", "
                                  << m_parameters.m_y_component << ", "
	                          << m_parameters.m_z_component << ") (N/m^3)"
				  << std::endl;
    m_logfile << "Heat supply = " << m_parameters.m_heat_supply << " (Watt/m^3)"
	      << std::endl;
    m_logfile << "Reference temperature = " << m_parameters.m_ref_temperature << " (K)"
	      << std::endl;

    m_logfile << "End time = " << m_parameters.m_end_time << std::endl;
    m_logfile << "Time data file name = " << m_parameters.m_time_file_name << std::endl;
  }

  template <typename LATraits, typename Tria>
  void PhaseFieldMonolithicSolve<LATraits, Tria>::run()
  {
    print_parameter_information();

    read_material_data(m_parameters.m_config_dir + m_parameters.m_material_file_name,
                       m_parameters.m_total_material_regions);

    std::vector<std::array<double, 4>> time_table;

    read_time_data(m_parameters.m_config_dir + m_parameters.m_time_file_name,
                   time_table);

    make_grid();
    setup_system();

    // initial conditions for temperature field
    setup_temperature_initial_conditions();

    output_results();

    m_time.increment(time_table);

    while(m_time.current() < m_time.end() + m_time.get_delta_t()*1.0e-6)
      {
	m_logfile << std::endl
		  << "Timestep " << m_time.get_timestep() << " @ " << m_time.current()
		  << 's' << std::endl;

        bool mesh_is_same = false;

        // initial guess for the resolve on the refined mesh
	BVector LBFGS_update_refine(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
          LBFGS_update_refine.initalize();
//	LBFGS_update_refine = 0.0;

        // local adaptive mesh refinement loop
	unsigned int adp_refine_iteration = 0;
        for (; adp_refine_iteration < m_parameters.m_max_adaptive_refine_times + 1; ++adp_refine_iteration)
          {
	    if (m_parameters.m_refinement_strategy == "adaptive-refine")
	      m_logfile << "\tAdaptive refinement-"<< adp_refine_iteration << ": " << std::endl;

	    BVector solution_delta(m_mpiInfo, m_blocks_desc, /*relevance=*/true);
              solution_delta.initalize();
//	    solution_delta = 0.0;

	    if (m_parameters.m_type_nonlinear_solver == "LBFGS")
	      solve_nonlinear_timestep_LBFGS(solution_delta, LBFGS_update_refine);
	    else
	      AssertThrow(false, ExcMessage("Nonlinear solver type not implemented"));

	    if (m_parameters.m_refinement_strategy == "adaptive-refine")
	      {

		if (adp_refine_iteration == m_parameters.m_max_adaptive_refine_times)
		  {
		    m_solution += solution_delta;
		    break;
		  }

		mesh_is_same = local_refine_and_solution_transfer(solution_delta,
								  LBFGS_update_refine);

		if (mesh_is_same)
		  {
		    m_solution += solution_delta;
		    break;
		  }
	      }
	    else if (m_parameters.m_refinement_strategy == "pre-refine")
	      {
		m_solution += solution_delta;
	        break;
	      }
	    else
	      {
		AssertThrow(false,
		            ExcMessage("Selected mesh refinement strategy not implemented!"));
	      }
          } // for (; adp_refine_iteration < m_parameters.m_max_adaptive_refine_times; ++adp_refine_iteration)

        //AssertThrow(adp_refine_iteration < m_parameters.m_max_adaptive_refine_times,
        //            ExcMessage("Number of local adaptive mesh refinement exceeds allowed maximum times!"));

	update_history_field_step();
	// output vtk files every 10 steps if there are too
	// many time steps
	//if (m_time.get_timestep() % 10 == 0)
        output_results();

	double energy_functional_current = calculate_energy_functional();
	m_logfile << "\t\tEnergy functional (J) = " << std::fixed << std::setprecision(10) << std::scientific
	          << energy_functional_current << std::endl;

	std::pair<double, double> energy_pair = calculate_total_strain_energy_and_crack_energy_dissipation();
	m_logfile << "\t\tTotal strain energy (J) = " << std::fixed << std::setprecision(10) << std::scientific
		  << energy_pair.first << std::endl;
	m_logfile << "\t\tCrack energy dissipation (J) = " << std::fixed << std::setprecision(10) << std::scientific
		  << energy_pair.second << std::endl;

	std::pair<double, std::array<double, 3>> time_energy;
	time_energy.first = m_time.current();
	time_energy.second[0] = energy_pair.first;
	time_energy.second[1] = energy_pair.second;
	time_energy.second[2] = energy_pair.first + energy_pair.second;
	m_history_energy.push_back(time_energy);

	int face_ID = m_parameters.m_reaction_force_face_id;
	if (face_ID >= 0)
	  calculate_reaction_force(face_ID);

        write_history_data();

	m_time.increment(time_table);
      } // while(m_time.current() < m_time.end() + m_time.get_delta_t()*1.0e-6)
  }
} // namespace PhaseField_monolithic


void init_dirs(PhaseField_monolithic::Parameters::AllParameters &parameters)
{
    // verify if output dir is existed. if not, create
    ::FileSystem::dir(parameters.m_output_dir);
    
    // find out the potential subdir name for output
    parameters.subDir  = ::FileSystem::find_next_numeric_subdir(parameters.m_output_dir);
    
    // update output dir and sub-dirs
    parameters.m_output_dir = parameters.m_output_dir + parameters.subDir + "/";
    parameters.oriDir       = parameters.m_output_dir + "ori/";
    parameters.histDir      = parameters.m_output_dir + "hist/";
    parameters.resultsDir   = parameters.m_output_dir + "results/";
    
    // create sub-dirs
    ::FileSystem::dir(parameters.oriDir);
    ::FileSystem::dir(parameters.histDir);
    ::FileSystem::dir(parameters.resultsDir);
}





int main(int argc, char* argv[])
{

  using namespace ::dealii;
  using namespace PhaseField_monolithic;
    
    
  if (argc != 2)
    AssertThrow(false,
    		ExcMessage("The number of arguments provided to the program has to be 2!"));
  
    // read prm by input command
  Parameters::AllParameters parameters(argv[1]);

    // initialize MPI by prm settings
    MPIInfo mpiInfo(parameters.m_mpi_type == "PETSc" ||
                    parameters.m_mpi_type == "Trilinos",
                    argc, argv);

    // print MPI / non-MPI info at rank 0
    if(mpiInfo.rank() == 0)
        mpiInfo.summary(std::cout);
    
    // create dirctories
    if(mpiInfo.isMPI())
    {
        // only rank 0 creates dirs to avoid repeated creations
        std::vector<std::string> dirNames;
        if(mpiInfo.rank() == 0)
        {
            init_dirs(parameters);
            dirNames = {parameters.m_output_dir, parameters.subDir, parameters.oriDir, parameters.histDir, parameters.resultsDir};
        }
        
        // sync dir names
        dirNames = Utilities::MPI::broadcast(*mpiInfo.mpiCommPtr(), dirNames, 0);
        
        // update local variables
        parameters.m_output_dir = dirNames[0];
        parameters.subDir       = dirNames[1];
        parameters.oriDir       = dirNames[2];
        parameters.histDir      = dirNames[3];
        parameters.resultsDir   = dirNames[4];
        
    } else {
        init_dirs(parameters);
    }


    
    // dimension by prm setting
    const unsigned int dim = parameters.m_dim;
    if(parameters.m_mpi_type == "PETSc") {
#ifdef HAVE_PETSC
        // PETSc type mpi
        if (dim == 2 )
        {
            DTria<2> tria(*mpiInfo.mpiCommPtr(),
                          typename Triangulation<2>::MeshSmoothing(
                            Triangulation<2>::smoothing_on_refinement |
                            Triangulation<2>::smoothing_on_coarsening),
                          DTria<2>::no_automatic_repartitioning);
            
            PhaseFieldMonolithicSolve<la::Traits<la::TagPETSc>, DTria<2>> Phasefield2D(parameters, mpiInfo, tria);
            Phasefield2D.run();
        }
        else if (dim == 3)
        {
            DTria<3> tria(*mpiInfo.mpiCommPtr(),
                          typename Triangulation<3>::MeshSmoothing(
                            Triangulation<3>::smoothing_on_refinement |
                            Triangulation<3>::smoothing_on_coarsening),
                          DTria<3>::no_automatic_repartitioning);
            
            PhaseFieldMonolithicSolve<la::Traits<la::TagPETSc>, DTria<3>> Phasefield3D(parameters, mpiInfo, tria);
            Phasefield3D.run();
        }
        else
        {
            AssertThrow(false,
                        ExcMessage("Dimension has to be either 2 or 3"));
        }
#endif
    } else if(parameters.m_mpi_type == "Trilinos") {
#ifdef HAVE_TRILINOS
        // Trilinos type mpi
        if (dim == 2 )
        {
            DTria<2> tria(*mpiInfo.mpiCommPtr(),
                          typename Triangulation<2>::MeshSmoothing(
                            Triangulation<2>::smoothing_on_refinement |
                            Triangulation<2>::smoothing_on_coarsening),
                          DTria<2>::no_automatic_repartitioning);
            
            PhaseFieldMonolithicSolve<la::Traits<la::TagTrilinos>, DTria<2>> Phasefield2D(parameters, mpiInfo, tria);
            Phasefield2D.run();
        }
        else if (dim == 3)
        {
            DTria<3> tria(*mpiInfo.mpiCommPtr(),
                          typename Triangulation<3>::MeshSmoothing(
                            Triangulation<3>::smoothing_on_refinement |
                            Triangulation<3>::smoothing_on_coarsening),
                          DTria<3>::no_automatic_repartitioning);
            
            PhaseFieldMonolithicSolve<la::Traits<la::TagTrilinos>, DTria<3>> Phasefield3D(parameters, mpiInfo, tria);
            Phasefield3D.run();
        }
        else
        {
            AssertThrow(false,
                        ExcMessage("Dimension has to be either 2 or 3"));
        }
#endif
    } else {
        // Serial type
        if (dim == 2 )
        {
            RTria<2> tria(Triangulation<2>::maximum_smoothing);
            
            PhaseFieldMonolithicSolve<la::Traits<la::TagSerial>, RTria<2>> Phasefield2D(parameters, mpiInfo, tria);
            Phasefield2D.run();
        }
        else if (dim == 3)
        {
            RTria<3> tria(Triangulation<3>::maximum_smoothing);
            
            PhaseFieldMonolithicSolve<la::Traits<la::TagSerial>, RTria<3>> Phasefield3D(parameters, mpiInfo, tria);
            Phasefield3D.run();
        }
        else
        {
            AssertThrow(false,
                        ExcMessage("Dimension has to be either 2 or 3"));
        }
    }
    
    
  

  return 0;
}
