#include "drake/multibody/plant/compliant_contact_manager.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

#include "drake/common/eigen_types.h"
#include "drake/common/scope_exit.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/query_results/penetration_as_point_pair.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#include "drake/multibody/contact_solvers/sap/sap_contact_problem.h"
#include "drake/multibody/contact_solvers/sap/sap_coupler_constraint.h"
#include "drake/multibody/contact_solvers/sap/sap_friction_cone_constraint.h"
#include "drake/multibody/contact_solvers/sap/sap_limit_constraint.h"
#include "drake/multibody/contact_solvers/sap/sap_solver.h"
#include "drake/multibody/contact_solvers/sap/sap_solver_results.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/triangle_quadrature/gaussian_triangle_quadrature_rule.h"
#include "drake/systems/framework/context.h"
#include "drake/multibody/plant/stop_watch.h"

#include <iostream>
#define PRINT_VAR(a) std::cout << #a": " << a << std::endl;

using drake::geometry::GeometryId;
using drake::geometry::PenetrationAsPointPair;
using drake::math::RotationMatrix;
using drake::multibody::contact_solvers::internal::ExtractNormal;
using drake::multibody::contact_solvers::internal::ExtractTangent;
using drake::multibody::contact_solvers::internal::SapContactProblem;
using drake::multibody::contact_solvers::internal::SapSolver;
using drake::multibody::contact_solvers::internal::SapSolverParameters;
using drake::multibody::contact_solvers::internal::SapSolverResults;
using drake::multibody::contact_solvers::internal::SapFrictionConeConstraint;
using drake::multibody::contact_solvers::internal::SapCouplerConstraint;
using drake::multibody::contact_solvers::internal::SapLimitConstraint;
using drake::multibody::internal::MultibodyTreeTopology;
using drake::multibody::internal::StopWatch;
using drake::systems::Context;

namespace drake {
namespace multibody {
namespace internal {

template <typename T>
AccelerationsDueToExternalForcesCache<T>::AccelerationsDueToExternalForcesCache(
    const MultibodyTreeTopology& topology)
    : forces(topology.num_bodies(), topology.num_velocities()),
      abic(topology),
      Zb_Bo_W(topology.num_bodies()),
      aba_forces(topology),
      ac(topology) {}

template <typename T>
CompliantContactManager<T>::~CompliantContactManager() {}

template <typename T>
void CompliantContactManager<T>::AddCouplerConstraint(
    const Joint<T>& joint0, const Joint<T>& joint1, const T& gear_ratio,
    const T& stiffness, const T& dissipation_time_scale) {
  DRAKE_THROW_UNLESS(joint0.num_velocities() == 1);
  DRAKE_THROW_UNLESS(joint1.num_velocities() == 1);
  coupler_constraints_info_.push_back(
      CouplerConstraintInfo{joint0.velocity_start(), joint1.velocity_start(),
                            gear_ratio, stiffness, dissipation_time_scale});
}

template <typename T>
void CompliantContactManager<T>::DeclareCacheEntries() {
  // N.B. We use xd_ticket() instead of q_ticket() since discrete
  // multibody plant does not have q's, but rather discrete state.
  // Therefore if we make it dependent on q_ticket() the Jacobian only
  // gets evaluated once at the start of the simulation.

  // Cache discrete contact pairs.
  const auto& discrete_contact_pairs_cache_entry = this->DeclareCacheEntry(
      "Discrete contact pairs.",
      systems::ValueProducer(
          this, &CompliantContactManager<T>::CalcDiscreteContactPairs),
      {systems::System<T>::xd_ticket(),
       systems::System<T>::all_parameters_ticket()});
  cache_indexes_.discrete_contact_pairs =
      discrete_contact_pairs_cache_entry.cache_index();

  // Due to issue #12786, we cannot mark
  // CacheIndexes::non_contact_forces_accelerations dependent on the
  // MultibodyPlant's inputs, as it should. However if we remove this
  // dependency, we run the risk of having an undetected algebraic loop. We use
  // this cache entry to signal when the computation of non-contact forces is in
  // progress so that we can detect an algebraic loop.
  const auto& non_contact_forces_evaluation_in_progress =
      this->DeclareCacheEntry(
          "Evaluation of non-contact forces and accelerations is in progress.",
          // N.B. This flag is set to true only when the computation is in
          // progress. Therefore its default value is `false`.
          systems::ValueProducer(false, &systems::ValueProducer::NoopCalc),
          {systems::System<T>::nothing_ticket()});
  cache_indexes_.non_contact_forces_evaluation_in_progress =
      non_contact_forces_evaluation_in_progress.cache_index();

  // Accelerations due to non-contact forces.
  // We cache non-contact forces, ABA forces and accelerations into a
  // AccelerationsDueToExternalForcesCache.
  AccelerationsDueToExternalForcesCache<T> non_contact_forces_accelerations(
      this->internal_tree().get_topology());
  const auto& non_contact_forces_accelerations_cache_entry =
      this->DeclareCacheEntry(
          "Non-contact forces accelerations.",
          systems::ValueProducer(
              this, non_contact_forces_accelerations,
              &CompliantContactManager<
                  T>::CalcAccelerationsDueToNonContactForcesCache),
          // Due to issue #12786, we cannot properly mark this entry dependent
          // on inputs. CalcAccelerationsDueToNonContactForcesCache() uses
          // CacheIndexes::non_contact_forces_evaluation_in_progress to guard
          // against algebraic loops.
          {systems::System<T>::xd_ticket(),
           systems::System<T>::all_parameters_ticket()});
  cache_indexes_.non_contact_forces_accelerations =
      non_contact_forces_accelerations_cache_entry.cache_index();

  const auto& contact_problem_cache_entry = this->DeclareCacheEntry(
      "Contact Problem.",
      systems::ValueProducer(
          this, ContactProblemCache<T>(plant().time_step()),
          &CompliantContactManager<T>::CalcContactProblemCache),
      {plant().cache_entry_ticket(cache_indexes_.discrete_contact_pairs)});
  cache_indexes_.contact_problem = contact_problem_cache_entry.cache_index();
}

template <typename T>
std::vector<ContactPairKinematics<T>>
CompliantContactManager<T>::CalcContactKinematics(
    const systems::Context<T>& context) const {
  const std::vector<internal::DiscreteContactPair<T>>& contact_pairs =
      EvalDiscreteContactPairs(context);
  const int num_contacts = contact_pairs.size();

  // N.B. Start stop watch after getting discrete pairs to avoid counting them
  // twice.
  StopWatch stop_watch;

  std::vector<ContactPairKinematics<T>> contact_kinematics;
  contact_kinematics.reserve(num_contacts);

  // Quick no-op exit.
  if (num_contacts == 0) return contact_kinematics;

  // Scratch workspace variables.
  const int nv = plant().num_velocities();
  Matrix3X<T> Jv_WAc_W(3, nv);
  Matrix3X<T> Jv_WBc_W(3, nv);
  Matrix3X<T> Jv_AcBc_W(3, nv);

  const Frame<T>& frame_W = plant().world_frame();
  for (int icontact = 0; icontact < num_contacts; ++icontact) {
    const auto& point_pair = contact_pairs[icontact];

    const GeometryId geometryA_id = point_pair.id_A;
    const GeometryId geometryB_id = point_pair.id_B;

    BodyIndex bodyA_index = this->geometry_id_to_body_index().at(geometryA_id);
    const Body<T>& bodyA = plant().get_body(bodyA_index);
    BodyIndex bodyB_index = this->geometry_id_to_body_index().at(geometryB_id);
    const Body<T>& bodyB = plant().get_body(bodyB_index);

    // Contact normal from point A into B.
    const Vector3<T>& nhat_W = -point_pair.nhat_BA_W;
    const Vector3<T>& p_WC = point_pair.p_WC;

    // Since v_AcBc_W = v_WBc - v_WAc the relative velocity Jacobian will be:
    //   J_AcBc_W = Jv_WBc_W - Jv_WAc_W.
    // That is the relative velocity at C is v_AcBc_W = J_AcBc_W * v.
    this->internal_tree().CalcJacobianTranslationalVelocity(
        context, JacobianWrtVariable::kV, bodyA.body_frame(), frame_W, p_WC,
        frame_W, frame_W, &Jv_WAc_W);
    this->internal_tree().CalcJacobianTranslationalVelocity(
        context, JacobianWrtVariable::kV, bodyB.body_frame(), frame_W, p_WC,
        frame_W, frame_W, &Jv_WBc_W);
    Jv_AcBc_W = Jv_WBc_W - Jv_WAc_W;

    // Define a contact frame C at the contact point such that the z-axis Cz
    // equals nhat_W. The tangent vectors are arbitrary, with the only
    // requirement being that they form a valid right handed basis with nhat_W.
    math::RotationMatrix<T> R_WC =
        math::RotationMatrix<T>::MakeFromOneVector(nhat_W, 2);

    const TreeIndex& treeA_index =
        tree_topology().body_to_tree_index(bodyA_index);
    const TreeIndex& treeB_index =
        tree_topology().body_to_tree_index(bodyB_index);
    // Sanity check, at least one must be valid.
    DRAKE_DEMAND(treeA_index.is_valid() || treeB_index.is_valid());

    // We have at most two blocks per contact.
    std::vector<typename ContactPairKinematics<T>::JacobianTreeBlock>
        jacobian_blocks;
    jacobian_blocks.reserve(2);

    // Tree A contribution to contact Jacobian Jv_W_AcBc_C.
    if (treeA_index.is_valid()) {
      Matrix3X<T> J = R_WC.matrix().transpose() *
                      Jv_AcBc_W.middleCols(
                          tree_topology().tree_velocities_start(treeA_index),
                          tree_topology().num_tree_velocities(treeA_index));
      jacobian_blocks.emplace_back(treeA_index, std::move(J));
    }

    // Tree B contribution to contact Jacobian Jv_W_AcBc_C.
    // This contribution must be added only if B is different from A.
    if ((treeB_index.is_valid() && !treeA_index.is_valid()) ||
        (treeB_index.is_valid() && treeB_index != treeA_index)) {
      Matrix3X<T> J = R_WC.matrix().transpose() *
                      Jv_AcBc_W.middleCols(
                          tree_topology().tree_velocities_start(treeB_index),
                          tree_topology().num_tree_velocities(treeB_index));
      jacobian_blocks.emplace_back(treeB_index, std::move(J));
    }

    contact_kinematics.emplace_back(point_pair.phi0, std::move(jacobian_blocks),
                                    std::move(R_WC));
  }

  stats_.contact_kinematics_time += stop_watch.Elapsed();

  return contact_kinematics;
}

template <typename T>
T CompliantContactManager<T>::GetPointContactStiffness(
    geometry::GeometryId id,
    const geometry::SceneGraphInspector<T>& inspector) const {
  const geometry::ProximityProperties* prop =
      inspector.GetProximityProperties(id);
  DRAKE_DEMAND(prop != nullptr);
  // N.B. Here we rely on the resolution of #13289 and #5454 to get properties
  // with the proper scalar type T. This will not work on scalar converted
  // models until those issues are resolved.
  return prop->template GetPropertyOrDefault<T>(
      geometry::internal::kMaterialGroup, geometry::internal::kPointStiffness,
      this->default_contact_stiffness());
}

template <typename T>
T CompliantContactManager<T>::GetDissipationTimeConstant(
    geometry::GeometryId id,
    const geometry::SceneGraphInspector<T>& inspector) const {
  const geometry::ProximityProperties* prop =
      inspector.GetProximityProperties(id);
  DRAKE_DEMAND(prop != nullptr);
  // N.B. Here we rely on the resolution of #13289 and #5454 to get properties
  // with the proper scalar type T. This will not work on scalar converted
  // models until those issues are resolved.
  return prop->template GetPropertyOrDefault<T>(
      geometry::internal::kMaterialGroup, "dissipation_time_constant",
      plant().time_step());
}

template <typename T>
double CompliantContactManager<T>::GetCoulombFriction(
    geometry::GeometryId id,
    const geometry::SceneGraphInspector<T>& inspector) const {
  const geometry::ProximityProperties* prop =
      inspector.GetProximityProperties(id);
  DRAKE_DEMAND(prop != nullptr);
  DRAKE_THROW_UNLESS(prop->HasProperty(geometry::internal::kMaterialGroup,
                                       geometry::internal::kFriction));
  return prop
      ->GetProperty<CoulombFriction<double>>(geometry::internal::kMaterialGroup,
                                             geometry::internal::kFriction)
      .dynamic_friction();
}

template <typename T>
T CompliantContactManager<T>::CombineStiffnesses(const T& k1, const T& k2) {
  // Simple utility to detect 0 / 0. As it is used in this method, denom
  // can only be zero if num is also zero, so we'll simply return zero.
  auto safe_divide = [](const T& num, const T& denom) {
    return denom == 0.0 ? 0.0 : num / denom;
  };
  return safe_divide(k1 * k2, k1 + k2);
}

template <typename T>
T CompliantContactManager<T>::CombineDissipationTimeConstant(const T& tau1,
                                                             const T& tau2) {
  return tau1 + tau2;
}

template <typename T>
void CompliantContactManager<T>::CalcDiscreteContactPairs(
    const systems::Context<T>& context,
    std::vector<internal::DiscreteContactPair<T>>* contact_pairs) const {
  plant().ValidateContext(context);
  DRAKE_DEMAND(contact_pairs != nullptr);

  StopWatch stop_watch;

  contact_pairs->clear();
  if (plant().num_collision_geometries() == 0) return;

  const auto contact_model = plant().get_contact_model();

  // We first compute the number of contact pairs so that we can allocate all
  // memory at once.
  // N.B. num_point_pairs = 0 when:
  //   1. There are legitimately no point pairs or,
  //   2. the point pair model is not even in use.
  // We guard for case (2) since EvalPointPairPenetrations() cannot be called
  // when point contact is not used and would otherwise throw an exception.
  int num_point_pairs = 0;  // The number of point contact pairs.
  if (contact_model == ContactModel::kPoint ||
      contact_model == ContactModel::kHydroelasticWithFallback) {
    num_point_pairs = plant().EvalPointPairPenetrations(context).size();
  }

  int num_quadrature_pairs = 0;
  // N.B. For discrete hydro we use a first order quadrature rule. As such,
  // the per-face quadrature point is the face's centroid and the weight is 1.
  // This is compatible with a mesh that is triangle or polygon. If we attempted
  // higher order quadrature, polygons would have to be decomposed into smaller
  // n-gons which can receive an appropriate set of quadrature points.
  if (contact_model == ContactModel::kHydroelastic ||
      contact_model == ContactModel::kHydroelasticWithFallback) {
    const std::vector<geometry::ContactSurface<T>>& surfaces =
        this->EvalContactSurfaces(context);
    for (const auto& s : surfaces) {
      // One quadrature point per face.
      num_quadrature_pairs += s.num_faces();
    }
  }
  const int num_contact_pairs = num_point_pairs + num_quadrature_pairs;
  contact_pairs->reserve(num_contact_pairs);
  if (contact_model == ContactModel::kPoint ||
      contact_model == ContactModel::kHydroelasticWithFallback) {
    AppendDiscreteContactPairsForPointContact(context, contact_pairs);
  }
  if (contact_model == ContactModel::kHydroelastic ||
      contact_model == ContactModel::kHydroelasticWithFallback) {
    AppendDiscreteContactPairsForHydroelasticContact(context, contact_pairs);
  }

  stats_.discrete_pairs_time += stop_watch.Elapsed();
}

template <typename T>
void CompliantContactManager<T>::AppendDiscreteContactPairsForPointContact(
    const systems::Context<T>& context,
    std::vector<internal::DiscreteContactPair<T>>* result) const {
  std::vector<internal::DiscreteContactPair<T>>& contact_pairs = *result;

  const geometry::QueryObject<T>& query_object =
      this->plant()
          .get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
  const geometry::SceneGraphInspector<T>& inspector = query_object.inspector();

  // Simple utility to detect 0 / 0. As it is used in this method, denom
  // can only be zero if num is also zero, so we'll simply return zero.
  auto safe_divide = [](const T& num, const T& denom) {
    return denom == 0.0 ? T(0.0) : num / denom;
  };

  // Fill in the point contact pairs.
  const std::vector<PenetrationAsPointPair<T>>& point_pairs =
      plant().EvalPointPairPenetrations(context);
  for (const PenetrationAsPointPair<T>& pair : point_pairs) {
    const T kA = GetPointContactStiffness(pair.id_A, inspector);
    const T kB = GetPointContactStiffness(pair.id_B, inspector);
    const T k = CombineStiffnesses(kA, kB);
    const T tauA = GetDissipationTimeConstant(pair.id_A, inspector);
    const T tauB = GetDissipationTimeConstant(pair.id_B, inspector);
    const T tau = CombineDissipationTimeConstant(tauA, tauB);

    // Combine friction coefficients.
    const double muA = GetCoulombFriction(pair.id_A, inspector);
    const double muB = GetCoulombFriction(pair.id_B, inspector);
    const T mu = T(safe_divide(2.0 * muA * muB, muA + muB));

    // We compute the position of the point contact based on Hertz's theory
    // for contact between two elastic bodies.
    const T denom = kA + kB;
    const T wA = (denom == 0 ? 0.5 : kA / denom);
    const T wB = (denom == 0 ? 0.5 : kB / denom);
    const Vector3<T> p_WC = wA * pair.p_WCa + wB * pair.p_WCb;

    const T phi0 = -pair.depth;
    const T fn0 = NAN;  // not used.
    const T d = NAN;    // not used.
    contact_pairs.push_back(
        {pair.id_A, pair.id_B, p_WC, pair.nhat_BA_W, phi0, fn0, k, d, tau, mu});
  }
}

// Most of the calculation in this function should be the same as in
// MultibodyPlant<T>::CalcDiscreteContactPairs().
template <typename T>
void CompliantContactManager<T>::
    AppendDiscreteContactPairsForHydroelasticContact(
        const systems::Context<T>& context,
        std::vector<internal::DiscreteContactPair<T>>* result) const {
  std::vector<internal::DiscreteContactPair<T>>& contact_pairs = *result;

  // Simple utility to detect 0 / 0. As it is used in this method, denom
  // can only be zero if num is also zero, so we'll simply return zero.
  auto safe_divide = [](const T& num, const T& denom) {
    return denom == 0.0 ? 0.0 : num / denom;
  };

  // N.B. For discrete hydro we use a first order quadrature rule. As such,
  // the per-face quadrature point is the face's centroid and the weight is 1.
  // This is compatible with a mesh that is triangle or polygon. If we attempted
  // higher order quadrature, polygons would have to be decomposed into smaller
  // n-gons which can receive an appropriate set of quadrature points.

  const geometry::QueryObject<T>& query_object =
      this->plant()
          .get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
  const geometry::SceneGraphInspector<T>& inspector = query_object.inspector();
  const std::vector<geometry::ContactSurface<T>>& surfaces =
      this->EvalContactSurfaces(context);
  for (const auto& s : surfaces) {
    const bool M_is_compliant = s.HasGradE_M();
    const bool N_is_compliant = s.HasGradE_N();
    DRAKE_DEMAND(M_is_compliant || N_is_compliant);

    // Combine dissipation.
    const T tau_M = GetDissipationTimeConstant(s.id_M(), inspector);
    const T tau_N = GetDissipationTimeConstant(s.id_N(), inspector);
    const T tau = CombineDissipationTimeConstant(tau_M, tau_N);

    // Combine friction coefficients.
    const double muA = GetCoulombFriction(s.id_M(), inspector);
    const double muB = GetCoulombFriction(s.id_N(), inspector);
    const T mu = T(safe_divide(2.0 * muA * muB, muA + muB));

    for (int face = 0; face < s.num_faces(); ++face) {
      const T& Ae = s.area(face);  // Face element area.

      // We found out that the hydroelastic query might report
      // infinitesimally small triangles (consider for instance an initial
      // condition that perfectly places an object at zero distance from the
      // ground.) While the area of zero sized triangles is not a problem by
      // itself, the badly computed normal on these triangles leads to
      // problems when computing the contact Jacobians (since we need to
      // obtain an orthonormal basis based on that normal.)
      // We therefore ignore infinitesimally small triangles. The tolerance
      // below is somehow arbitrary and could possibly be tightened.
      if (Ae > 1.0e-14) {
        // From ContactSurface's documentation: The normal of each face is
        // guaranteed to point "out of" N and "into" M.
        const Vector3<T>& nhat_W = s.face_normal(face);

        // One dimensional pressure gradient (in Pa/m). Unlike [Masterjohn
        // et al. 2021], for convenience we define both pressure gradients
        // to be positive in the direction "into" the bodies. Therefore,
        // we use the minus sign for gN.
        // [Masterjohn et al., 2021] Discrete Approximation of Pressure
        // Field Contact Patches.
        const T gM = M_is_compliant
                         ? s.EvaluateGradE_M_W(face).dot(nhat_W)
                         : T(std::numeric_limits<double>::infinity());
        const T gN = N_is_compliant
                         ? -s.EvaluateGradE_N_W(face).dot(nhat_W)
                         : T(std::numeric_limits<double>::infinity());

        constexpr double kGradientEpsilon = 1.0e-14;
        if (gM < kGradientEpsilon || gN < kGradientEpsilon) {
          // Mathematically g = gN*gM/(gN+gM) and therefore g = 0 when
          // either gradient on one of the bodies is zero. A zero gradient
          // means there is no contact constraint, and therefore we
          // ignore it to avoid numerical problems in the discrete solver.
          continue;
        }

        // Effective hydroelastic pressure gradient g result of
        // compliant-compliant interaction, see [Masterjohn et al., 2021].
        // The expression below is mathematically equivalent to g =
        // gN*gM/(gN+gM) but it has the advantage of also being valid if
        // one of the gradients is infinity.
        const T g = 1.0 / (1.0 / gM + 1.0 / gN);

        // Position of quadrature point Q in the world frame (since mesh_W
        // is measured and expressed in W).
        const Vector3<T>& p_WQ = s.centroid(face);
        // For a triangle, its centroid has the fixed barycentric
        // coordinates independent of the shape of the triangle. Using
        // barycentric coordinates to evaluate field value could be
        // faster than using Cartesian coordiantes, especially if the
        // TriangleSurfaceMeshFieldLinear<> does not store gradients and
        // has to solve linear equations to convert Cartesian to
        // barycentric coordinates.
        const Vector3<T> tri_centroid_barycentric(1 / 3., 1 / 3., 1 / 3.);
        // Pressure at the quadrature point.
        const T p0 = s.is_triangle()
                         ? s.tri_e_MN().Evaluate(face, tri_centroid_barycentric)
                         : s.poly_e_MN().EvaluateCartesian(face, p_WQ);

        // Effective compliance in the normal direction for the given
        // discrete patch, refer to [Masterjohn et al., 2021] for details.
        // [Masterjohn, 2021] Masterjohn J., Guoy D., Shepherd J. and Castro
        // A., 2021. Discrete Approximation of Pressure Field Contact Patches.
        // Available at https://arxiv.org/abs/2110.04157.
        const T k = Ae * g;

        // phi < 0 when in penetration.
        const T phi0 = -p0 / g;

        if (k > 0) {
          const T fn0 = NAN;  // not used.
          const T d = NAN;    // not used.
          contact_pairs.push_back(
              {s.id_M(), s.id_N(), p_WQ, nhat_W, phi0, fn0, k, d, tau, mu});
        }
      }
    }
  }
}

template <typename T>
void CompliantContactManager<T>::
    CalcForceElementsContributionExcludingJointDamping(
        const systems::Context<T>& context, MultibodyForces<T>* forces) const {
  DRAKE_DEMAND(forces != nullptr);
  DRAKE_DEMAND(forces->CheckHasRightSizeForModel(plant()));
  const internal::PositionKinematicsCache<T>& pc =
      plant().EvalPositionKinematics(context);
  const internal::VelocityKinematicsCache<T>& vc =
      plant().EvalVelocityKinematics(context);

  forces->SetZero();      
  // TODO(amcastro-tri): revisit this code if joint damping becomes a force
  // element in the future.
  for (ForceElementIndex e(0); e < plant().num_force_elements(); ++e) {
    const ForceElement<T>& force_element = plant().get_force_element(e);
    force_element.CalcAndAddForceContribution(context, pc, vc, forces);    
  }
}

template <typename T>
void CompliantContactManager<T>::CalcNonContactForcesExcludingJointDamping(
    const systems::Context<T>& context, MultibodyForces<T>* forces) const {
  // Compute forces applied through force elements. Note that this resets
  // forces to empty so must come first.
  CalcForceElementsContributionExcludingJointDamping(context, forces);
  // Add contribution from externally applied forces through input ports.
  this->AddInForcesFromInputPorts(context, forces);
}

template <typename T>
void CompliantContactManager<T>::CalcNonContactForcesExcludingJointLimits(
    const systems::Context<T>& context, MultibodyForces<T>* forces) const {
  // Compute forces applied through force elements. Note that this resets
  // forces to empty so must come first.
  this->CalcForceElementsContribution(context, forces);
  this->AddInForcesFromInputPorts(context, forces);
}

template <typename T>
void CompliantContactManager<T>::CalcAccelerationsDueToNonContactForcesCache(
    const systems::Context<T>& context,
    AccelerationsDueToExternalForcesCache<T>* forward_dynamics_cache)
    const {

  StopWatch stop_watch;    

  // To overcame issue #12786, we use this additional cache entry
  // to detect algebraic loops.
  systems::CacheEntryValue& value =
      plant()
          .get_cache_entry(
              cache_indexes_.non_contact_forces_evaluation_in_progress)
          .get_mutable_cache_entry_value(context);
  bool& evaluation_in_progress = value.GetMutableValueOrThrow<bool>();
  if (evaluation_in_progress) {
    const char* error_message =
        "Algebraic loop detected. This situation is caused when connecting the "
        "input of your MultibodyPlant to the output of a feedback system which "
        "is an algebraic function of a feedthrough output of the plant. Ways "
        "to remedy this: 1. Revisit the model for your feedback system. "
        "Consider if its output can be written in terms of other inputs. 2. "
        "Break the algebraic loop by adding state to the controller, typically "
        "to 'remember' a previous input. 3. Break the algebraic loop by adding "
        "a zero-order hold system between the output of the plant and your "
        "feedback system. This effectively delays the input signal to the "
        "controller.";
    throw std::runtime_error(error_message);
  }
  // Mark the start of the computation. If within an algebraic
  // loop, pulling from the plant's input ports during the
  // computation will trigger the recursive evaluation of this
  // method and the exception above will be thrown.
  evaluation_in_progress = true;
  // If the exception above is triggered, we will leave this method and the
  // computation will no longer be "in progress". We use a scoped guard so that
  // we have a chance to mark it as such when we leave this scope.
  ScopeExit guard(
      [&evaluation_in_progress]() { evaluation_in_progress = false; });

  // N.B. Joint limits are modeled as constraints. Therefore here we only add
  // all other external forces.
  CalcNonContactForcesExcludingJointLimits(
      context, &forward_dynamics_cache->forces);

  const VectorX<T> diagonal_inertia =
      plant().EvalReflectedInertiaCache(context) +
      joint_damping_ * plant().time_step();

  // We compute the articulated body inertia including the contribution of the
  // additional diagonal elements arising from the implicit treatment of joint
  // damping.
  this->internal_tree().CalcArticulatedBodyInertiaCache(
      context, diagonal_inertia, &forward_dynamics_cache->abic);
  this->internal_tree().CalcArticulatedBodyForceBias(
      context, forward_dynamics_cache->abic, &forward_dynamics_cache->Zb_Bo_W);
  this->internal_tree().CalcArticulatedBodyForceCache(
      context, forward_dynamics_cache->abic, forward_dynamics_cache->Zb_Bo_W,
      forward_dynamics_cache->forces, &forward_dynamics_cache->aba_forces);
  this->internal_tree().CalcArticulatedBodyAccelerations(
      context, forward_dynamics_cache->abic, forward_dynamics_cache->aba_forces,
      &forward_dynamics_cache->ac);

  // Mark the end of the computation.
  evaluation_in_progress = false;

  stats_.free_motion_accelerations_time += stop_watch.Elapsed();
}

template <typename T>
void CompliantContactManager<T>::CalcFreeMotionVelocities(
    const systems::Context<T>& context, VectorX<T>* v_star) const {
  DRAKE_DEMAND(v_star != nullptr);
  StopWatch stop_watch;
  // N.B. Forces are evaluated at the previous time step state. This is
  // consistent with the explicit Euler and symplectic Euler schemes.
  // TODO(amcastro-tri): Implement free-motion velocities update based on the
  // theta-method, as in the SAP paper.
  const VectorX<T>& vdot0 =
      EvalAccelerationsDueToNonContactForcesCache(context).get_vdot();
  const double dt = this->plant().time_step();
  const VectorX<T>& x0 =
      context.get_discrete_state(this->multibody_state_index()).value();
  const auto v0 = x0.bottomRows(this->plant().num_velocities());
  *v_star = v0 + dt * vdot0;
  stats_.free_motion_velocities_time += stop_watch.Elapsed();
}

template <typename T>
void CompliantContactManager<T>::CalcLinearDynamicsMatrix(
    const systems::Context<T>& context, std::vector<MatrixX<T>>* A) const {
  DRAKE_DEMAND(A != nullptr);
  A->resize(tree_topology().num_trees());
  const int nv = plant().num_velocities();

  // TODO(amcastro-tri): implicitly include force elements such as joint
  // dissipation and/or stiffness.
  // TODO(amcastro-tri): consider placing the computation of the dense mass
  // matrix in a cache entry to minimize heap allocations or better yet,
  // implement a MultibodyPlant method to compute the per-tree mass matrices.
  MatrixX<T> M(nv, nv);
  plant().CalcMassMatrix(context, &M);
  M.diagonal() += plant().time_step() * joint_damping_;

  for (TreeIndex t(0); t < tree_topology().num_trees(); ++t) {
    const int tree_start = tree_topology().tree_velocities_start(t);
    const int tree_nv = tree_topology().num_tree_velocities(t);
    (*A)[t] = M.block(tree_start, tree_start, tree_nv, tree_nv);
  }
}

template <typename T>
const std::vector<internal::DiscreteContactPair<T>>&
CompliantContactManager<T>::EvalDiscreteContactPairs(
    const systems::Context<T>& context) const {
  return plant()
      .get_cache_entry(cache_indexes_.discrete_contact_pairs)
      .template Eval<std::vector<internal::DiscreteContactPair<T>>>(context);
}

template <typename T>
const multibody::internal::AccelerationKinematicsCache<T>&
CompliantContactManager<T>::EvalAccelerationsDueToNonContactForcesCache(
    const systems::Context<T>& context) const {
  return plant()
      .get_cache_entry(cache_indexes_.non_contact_forces_accelerations)
      .template Eval<AccelerationsDueToExternalForcesCache<T>>(context)
      .ac;
}

template <typename T>
void CompliantContactManager<T>::DoCalcContactSolverResults(
    const systems::Context<T>& context,
    contact_solvers::internal::ContactSolverResults<T>* contact_results) const {

  StopWatch contact_results_stopwatch;

  const ContactProblemCache<T>& contact_problem_cache =
      EvalContactProblemCache(context);
  const SapContactProblem<T>& sap_problem = *contact_problem_cache.sap_problem;

  // We use the velocity stored in the current context as initial guess.
  const VectorX<T>& x0 =
      context.get_discrete_state(this->multibody_state_index()).value();
  const auto v0 = x0.bottomRows(this->plant().num_velocities());

  // Solve contact problem.
  StopWatch stop_watch;
  SapSolver<T> sap;
  sap.set_parameters(sap_parameters_);
  SapSolverResults<T> sap_results;
  const drake::multibody::contact_solvers::internal::SapSolverStatus status =
      sap.SolveWithGuess(sap_problem, v0, &sap_results);
  if (status !=
      drake::multibody::contact_solvers::internal::SapSolverStatus::kSuccess) {
    throw std::runtime_error("SAP solver failed.");
  }
  stats_.solve_problem_time += stop_watch.Elapsed();
  stats_.num_iters += sap.get_statistics().num_iters;
  stats_.num_ls_iters += sap.get_statistics().num_line_search_iters;
  if constexpr (std::is_same_v<T, double>) {
    stats_.sap_stats.push_back(sap.get_statistics());
  }

  const std::vector<internal::DiscreteContactPair<T>>& discrete_pairs =
      EvalDiscreteContactPairs(context);
  const int num_contacts = discrete_pairs.size();

  PackContactSolverResults(sap_problem, num_contacts, sap_results,
                           contact_results);

  stats_.contact_results_time += contact_results_stopwatch.Elapsed();
}

template <typename T>
void CompliantContactManager<T>::PackContactSolverResults(
    const SapContactProblem<T>& problem, int num_contacts,
    const SapSolverResults<T>& sap_results,
    contact_solvers::internal::ContactSolverResults<T>* contact_results) const {
  DRAKE_DEMAND(contact_results != nullptr);

  StopWatch stop_watch;

  contact_results->Resize(plant().num_velocities(), num_contacts);
  contact_results->v_next = sap_results.v;
  // We added all contact constraints first and therefore we know the head of
  // the impulses corresponds to contact impulses.
  const VectorX<T> contact_forces =
      sap_results.gamma.head(3 * num_contacts) / plant().time_step();
  const VectorX<T> contact_velocities = sap_results.vc.head(3 * num_contacts);
  ExtractNormal(contact_forces, &contact_results->fn);
  ExtractTangent(contact_forces, &contact_results->ft);
  ExtractNormal(contact_velocities, &contact_results->vn);
  ExtractTangent(contact_velocities, &contact_results->vt);

  auto& tau_contact = contact_results->tau_contact;
  tau_contact.setZero();
  for (int i = 0; i < num_contacts; ++i) {
    const auto& c = problem.get_constraint(i);
    {
      const TreeIndex t(c.first_clique());
      const MatrixX<T>& Jic = c.first_clique_jacobian();
      const int v_start = tree_topology().tree_velocities_start(t);
      const int nv = tree_topology().num_tree_velocities(t);
      const auto fi = contact_forces.template segment<3>(3 * i);
      tau_contact.segment(v_start, nv) += Jic.transpose() * fi;
    }

    if (c.num_cliques() == 2) {
      const TreeIndex t(c.second_clique());
      const MatrixX<T>& Jic = c.second_clique_jacobian();
      const int v_start = tree_topology().tree_velocities_start(t);
      const int nv = tree_topology().num_tree_velocities(t);
      const auto fi = contact_forces.template segment<3>(3 * i);
      tau_contact.segment(v_start, nv) += Jic.transpose() * fi;
    }
  }

  stats_.pack_results_time += stop_watch.Elapsed();
}

template <typename T>
std::vector<RotationMatrix<T>>
CompliantContactManager<T>::AddContactConstraints(
    const systems::Context<T>& context, SapContactProblem<T>* problem) const {
  DRAKE_DEMAND(problem != nullptr);

  // Parameters used by SAP to estimate regularization, see [Castro et al.,
  // 2021].
  // TODO(amcastro-tri): consider exposing these parameters.
  constexpr double beta = 1.0;
  constexpr double sigma = 1.0e-3;

  const std::vector<internal::DiscreteContactPair<T>>& contact_pairs =
      EvalDiscreteContactPairs(context);
  const int num_contacts = contact_pairs.size();

  // Quick no-op exit.
  if (num_contacts == 0) return std::vector<RotationMatrix<T>>();

  std::vector<ContactPairKinematics<T>> contact_kinematics =
      CalcContactKinematics(context);

  std::vector<RotationMatrix<T>> R_WC;
  R_WC.reserve(num_contacts);
  for (int icontact = 0; icontact < num_contacts; ++icontact) {
    const auto& discrete_pair = contact_pairs[icontact];

    const T stiffness = discrete_pair.stiffness;
    const T dissipation_time_scale = discrete_pair.dissipation_time_scale;
    const T friction = discrete_pair.friction_coefficient;
    const T phi = contact_kinematics[icontact].phi;
    const auto& jacobian_blocks = contact_kinematics[icontact].jacobian;

    const typename SapFrictionConeConstraint<T>::Parameters parameters{
        friction, stiffness, dissipation_time_scale, beta, sigma};

    if (jacobian_blocks.size() == 1) {
      problem->AddConstraint(std::make_unique<SapFrictionConeConstraint<T>>(
          jacobian_blocks[0].tree, std::move(jacobian_blocks[0].J), phi,
          parameters));
    } else {
      problem->AddConstraint(std::make_unique<SapFrictionConeConstraint<T>>(
          jacobian_blocks[0].tree, jacobian_blocks[1].tree,
          std::move(jacobian_blocks[0].J), std::move(jacobian_blocks[1].J), phi,
          parameters));
    }
    R_WC.emplace_back(std::move(contact_kinematics[icontact].R_WC));
  }
  return R_WC;
}

template <typename T>
void CompliantContactManager<T>::AddLimitConstraints(
    const systems::Context<T>& context, SapContactProblem<T>* problem) const {
  DRAKE_DEMAND(problem != nullptr);

  // TODO(amcastro-tri): consider exposing this parameter.
  constexpr double kBeta = 0.1;

  const auto& joint_limits_parameters = this->joint_limits_parameters();

  // TODO: you might get away without joint_limits_parameters from the plant,
  // now that you are not using the stiffness and damping estimated by the
  // plant.
  for (size_t index = 0;
       index < joint_limits_parameters.joints_with_limits.size(); ++index) {
    const JointIndex joint_index =
        joint_limits_parameters.joints_with_limits[index];
    const double lower_limit = joint_limits_parameters.lower_limit[index];
    const double upper_limit = joint_limits_parameters.upper_limit[index];

    // N.B. MultibodyPlant estimates very conservative (soft) stiffness and
    // damping parameters to ensure that the explicit treatment of the compliant
    // forces used to impose limits does not become unstable. SAP however treats
    // these forces implicitly and therefore these parameters can be tighten for
    // stiffer limits. Here we let the stiffness parameter to a very high value
    // so that internally SAP limits the amount of regularization using the
    // kBeta parameter as described in the SAP paper, [Castro et al., 2021]. As
    // shown in the SAP paper, a dissipation time scale of the order of the time
    // step leads to a critically damped constraint.
    // TODO(amcastro-tri): allow users to specify joint limits stiffness and
    // damping.
    const double stiffness = 1.0e12;
    const double dissipation_time_scale = plant().time_step();
    //const double stiffness = 10000. * joint_limits_parameters.stiffness[index];
    //const double damping = joint_limits_parameters.damping[index];
    //const double dissipation_time_scale = damping / stiffness;
    const Joint<T>& joint = plant().get_joint(joint_index);

    const int velocity_index = joint.velocity_start();
    const TreeIndex tree_index =
        tree_topology().velocity_to_tree_index(velocity_index);
    const int tree_dof =
        velocity_index - tree_topology().tree_velocities_start(tree_index);
    const int tree_nv = tree_topology().num_tree_velocities(tree_index);

    const T& q0 = joint.GetOnePosition(context);
    typename SapLimitConstraint<T>::Parameters parameters{
        lower_limit, upper_limit, stiffness, dissipation_time_scale, kBeta};

    problem->AddConstraint(std::make_unique<SapLimitConstraint<T>>(
        tree_index, tree_dof, tree_nv, q0, std::move(parameters)));
  }
}

template <typename T>
void CompliantContactManager<T>::AddCouplerConstraints(
    const systems::Context<T>& context, SapContactProblem<T>* problem) const {
  DRAKE_DEMAND(problem != nullptr);

  // Previous time step positions.
  const VectorX<T> q0 = plant().GetPositions(context);  

  for (const CouplerConstraintInfo& info : coupler_constraints_info_) {
    const TreeIndex c0 = tree_topology().velocity_to_tree_index(info.q0);
    const TreeIndex c1 = tree_topology().velocity_to_tree_index(info.q1);

    // Sanity check.
    DRAKE_DEMAND(c0.is_valid() && c1.is_valid());

    // Constraint function.
    const T g0 = q0[info.q0] - info.gear_ratio * q0[info.q1];

    // TODO: expose this parameter.
    const double beta = 0.1;

    const typename SapCouplerConstraint<T>::Parameters parameters{
        info.gear_ratio, info.stiffness, info.dissipation_time_scale, beta};

    if (c0 == c1) {
      const int nv = tree_topology().num_tree_velocities(c0);
      MatrixX<T> J = MatrixX<T>::Zero(1, nv);
      // J = dg/dv
      J(0, info.q0) = 1.0;
      J(0, info.q1) = -info.gear_ratio;
      
      problem->AddConstraint(std::make_unique<SapCouplerConstraint<T>>(
          parameters, c0, J, g0));
    } else {
      const int nv0 = tree_topology().num_tree_velocities(c0);
      const int nv1 = tree_topology().num_tree_velocities(c1);
      MatrixX<T> J0 = MatrixX<T>::Zero(1, nv0);
      MatrixX<T> J1 = MatrixX<T>::Zero(1, nv1);
      J0(0, info.q0) = 1.0;
      J1(0, info.q1) = -info.gear_ratio;
      problem->AddConstraint(std::make_unique<SapCouplerConstraint<T>>(
          parameters, c0, c1, J0, J1, g0));
    }
  }

}

template <typename T>
void CompliantContactManager<T>::CalcContactProblemCache(
    const systems::Context<T>& context, ContactProblemCache<T>* cache) const {
  SapContactProblem<T>& problem = *cache->sap_problem;
  std::vector<MatrixX<T>> A;
  CalcLinearDynamicsMatrix(context, &A);
  VectorX<T> v_star;
  CalcFreeMotionVelocities(context, &v_star);
  // N.B. We exclude time to compute v* from actually making the problem.
  StopWatch stop_watch;
  problem.Reset(std::move(A), std::move(v_star));
  cache->R_WC = AddContactConstraints(context, &problem);
  AddLimitConstraints(context, &problem);
  AddCouplerConstraints(context, &problem);
  stats_.make_problem_time += stop_watch.Elapsed();
}

template <typename T>
const ContactProblemCache<T>&
CompliantContactManager<T>::EvalContactProblemCache(
    const systems::Context<T>& context) const {
  return plant()
      .get_cache_entry(cache_indexes_.contact_problem)
      .template Eval<ContactProblemCache<T>>(context);
}

template <typename T>
void CompliantContactManager<T>::DoCalcDiscreteValues(
    const drake::systems::Context<T>& context,
    drake::systems::DiscreteValues<T>* updates) const {
  
  StopWatch stop_watch;  

  const contact_solvers::internal::ContactSolverResults<T>& results =
      this->EvalContactSolverResults(context);

  // Previous time step positions.
  const int nq = plant().num_positions();
  const auto x0 =
      context.get_discrete_state(this->multibody_state_index()).get_value();
  const auto q0 = x0.topRows(nq);

  // Retrieve the solution velocity for the next time step.
  const VectorX<T>& v_next = results.v_next;

  // Update generalized positions.
  VectorX<T> qdot_next(plant().num_positions());
  plant().MapVelocityToQDot(context, v_next, &qdot_next);
  const VectorX<T> q_next = q0 + plant().time_step() * qdot_next;

  VectorX<T> x_next(plant().num_multibody_states());
  x_next << q_next, v_next;
  updates->set_value(this->multibody_state_index(), x_next);

  stats_.discrete_update_time += stop_watch.Elapsed();
}

template <typename T>
void CompliantContactManager<T>::ExtractModelInfo() {
  joint_damping_ = VectorX<T>::Zero(plant().num_velocities());
  for(JointIndex j(0); j < plant().num_joints(); ++j) {
    const Joint<T>& joint = plant().get_joint(j);
    const int velocity_start = joint.velocity_start();
    const int nv = joint.num_velocities();
    joint_damping_.segment(velocity_start, nv) = joint.damping_vector();
  }
  PRINT_VAR(joint_damping_.transpose());
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::internal::CompliantContactManager);
