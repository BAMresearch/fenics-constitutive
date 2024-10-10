from __future__ import annotations

import basix
import dolfinx as df
import numpy as np
import ufl
from petsc4py import PETSc

from .interfaces import IncrSmallStrainModel
from .maps import SubSpaceMap, build_subspace_map
from .stress_strain import ufl_mandel_strain
from scipy.linalg import logm, expm


def build_history(
    law: IncrSmallStrainModel, mesh: df.mesh.Mesh, q_degree: int
) -> dict[str, df.fem.Function] | None:
    """Build the history space and function(s) for the given law.

    Args:
        law: The constitutive law.
        mesh: Either the full mesh for a homogenous domain or the submesh.
        q_degree: The quadrature degree.

    Returns:
        The history function(s) for the given law.

    """
    if law.history_dim is None:
        return None

    history = {}
    for key, value in law.history_dim.items():
        match value:
            case int():
                Qh = ufl.VectorElement(
                    "Quadrature",
                    mesh.ufl_cell(),
                    q_degree,
                    quad_scheme="default",
                    dim=value,
                )
            case tuple():
                Qh = ufl.TensorElement(
                    "Quadrature",
                    mesh.ufl_cell(),
                    q_degree,
                    quad_scheme="default",
                    shape=value,
                )
        history_space = df.fem.FunctionSpace(mesh, Qh)
        history[key] = df.fem.Function(history_space)
    return history


class IncrSmallStrainProblem(df.fem.petsc.NonlinearProblem):
    """
    A nonlinear problem for incremental small strain models. To be used with
    the dolfinx NewtonSolver.

    Args:
        laws: A list of tuples where the first element is the constitutive law and the second
            element is the cells for the submesh. If only one law is provided, it is assumed
            that the domain is homogenous.
        u: The displacement field. This is the unknown in the nonlinear problem.
        bcs: The Dirichlet boundary conditions.
        q_degree: The quadrature degree (Polynomial degree which the quadrature rule needs to integrate exactly).
        form_compiler_options: The options for the form compiler.
        jit_options: The options for the JIT compiler.

    Note:
        If `super().__init__(R, u, bcs, dR)` is called within the __init__ method,
        the user cannot add Neumann BCs. Therefore, the compilation (i.e. call to
        `super().__init__()`) is done when `df.nls.petsc.NewtonSolver` is initialized.
        The solver will call `self._A = fem.petsc.create_matrix(problem.a)` and hence
        we override the property ``a`` of NonlinearProblem to ensure that the form is compiled.
    """

    def __init__(
        self,
        laws: list[tuple[IncrSmallStrainModel, np.ndarray]] | IncrSmallStrainModel,
        u: df.fem.Function,
        bcs: list[df.fem.DirichletBCMetaClass],
        q_degree: int,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        mesh_update = False,
        co_rotation = False,
    ):
        mesh = u.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)
        if isinstance(laws, IncrSmallStrainModel):
            laws = [(laws, cells)]

        constraint = laws[0][0].constraint
        assert all(
            law[0].constraint == constraint for law in laws
        ), "All laws must have the same constraint"

        gdim = mesh.ufl_cell().geometric_dimension()
        assert (
            constraint.geometric_dim() == gdim
        ), "Geometric dimension mismatch between mesh and laws"

        QVe = ufl.VectorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            dim=constraint.stress_strain_dim(),
        )
        QTe = ufl.TensorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            shape=(constraint.stress_strain_dim(), constraint.stress_strain_dim()),
        )
        Q_grad_u_e = ufl.TensorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            shape=(gdim, gdim),
        )
        QV = df.fem.FunctionSpace(mesh, QVe)
        QT = df.fem.FunctionSpace(mesh, QTe)

        self.mesh_update = mesh_update
        self.co_rotation = co_rotation
        self.laws: list[tuple[IncrSmallStrainModel, np.ndarray]] = []
        self.submesh_maps: list[SubSpaceMap] = []

        self._del_grad_u = []
        self._stress = []
        self._history_0 = []
        self._history_1 = []
        self._tangent = []

        self._time = 0.0  # time at the end of the increment

        # if len(laws) > 1:
        for law, cells in laws:
            self.laws.append((law, cells))

            # default case for homogenous domain
            submesh = mesh

            if len(laws) > 1:
                # ### submesh and subspace for strain, stress
                subspace_map, submesh, QV_subspace = build_subspace_map(
                    cells, QV, return_subspace=True
                )
                self.submesh_maps.append(subspace_map)
                self._stress.append(df.fem.Function(QV_subspace))

            # subspace for grad u
            Q_grad_u_subspace = df.fem.FunctionSpace(submesh, Q_grad_u_e)
            self._del_grad_u.append(df.fem.Function(Q_grad_u_subspace))

            # subspace for tanget
            QT_subspace = df.fem.FunctionSpace(submesh, QTe)
            self._tangent.append(df.fem.Function(QT_subspace))

            # subspaces for history
            history_0 = build_history(law, submesh, q_degree)
            history_1 = (
                {key: fn.copy() for key, fn in history_0.items()}
                if isinstance(history_0, dict)
                else history_0
            )
            self._history_0.append(history_0)
            self._history_1.append(history_1)

        self.stress_0 = df.fem.Function(QV)
        self.stress_1 = df.fem.Function(QV)
        self.tangent = df.fem.Function(QT)

        u_, du = ufl.TestFunction(u.function_space), ufl.TrialFunction(u.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        self.R_form = (
            ufl.inner(ufl_mandel_strain(u_, constraint), self.stress_1) * self.dxm
        )
        self.dR_form = (
            ufl.inner(
                ufl_mandel_strain(du, constraint),
                ufl.dot(self.tangent, ufl_mandel_strain(u_, constraint)),
            )
            * self.dxm
        )

        self._u = u
        self._u0 = u.copy()
        self._bcs = bcs
        self._form_compiler_options = form_compiler_options
        self._jit_options = jit_options

        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        self.q_points, _ = basix.make_quadrature(basix_celltype, q_degree)

        self.del_grad_u_expr = df.fem.Expression(
            ufl.nabla_grad(self._u - self._u0), self.q_points
        )

    @property
    def a(self) -> df.fem.FormMetaClass:
        """Compiled bilinear form (the Jacobian form)"""

        if not hasattr(self, "_a"):
            # ensure compilation of UFL forms
            super().__init__(
                self.R_form,
                self._u,
                self._bcs,
                self.dR_form,
                form_compiler_options=self._form_compiler_options
                if self._form_compiler_options is not None
                else {},
                jit_options=self._jit_options if self._jit_options is not None else {},
            )

        return self._a

    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values, but here
        we use it to update the stress, tangent and history.

        Args:
            x: The vector containing the latest solution

        """
        super().form(x)
        assert (
            x.array.data == self._u.vector.array.data
        ), "The solution vector must be the same as the one passed to the MechanicsProblem"

        # print('Mesh Dimension: ', np.shape(self._u.function_space.mesh.geometry.x))
        # print('U Dimension', np.shape(self._u.x.array))
        aaa = self._u.x.array.reshape(-1, 3)
        # print(aaa[:8,:])
        # print(self._u.interpola)



        if self.mesh_update:

            # Update the mesh geometry to the midpoint configuration
            ####################################
            # midpoint_displacement = 0.5 * (self._u.x.array - self._u0.x.array)
            ###########################################
            # Assuming `self._u` is the displacement function and its values are at P2 points (vertices + mid-edge points)
            dofmap = self._u.function_space.dofmap
            # Create a vector function space of order 1
            V_CG = df.fem.VectorFunctionSpace(self._u.function_space.mesh, ("CG", 1))
            u_CG0 = df.fem.Function(V_CG)
            u_CG = df.fem.Function(V_CG)

            u_CG0.interpolate(self._u0)
            u_CG.interpolate(self._u)

            # print('u_CG ', np.shape(u_CG.x.array))
            # print('u ', np.shape(self._u.x.array))

            midpoint_displacement = 0.5 *(u_CG.x.array - u_CG0.x.array)

            # print(u_CG1.x.array)
            # # dofmap.x.array.sort
            # # print(dofmap)
            # mesh = self._u.function_space.mesh
            #
            # # with df.io.XDMFFile(mesh.comm, "Mesh.xdmf", "w") as xdmf:
            # #     self._u.function_space.mesh.name = "mesh"
            # #     xdmf.write_mesh(mesh)
            # dof_coordinates = self._u.function_space.tabulate_dof_coordinates().reshape((-1, 3))
            # vertex_coords = mesh.geometry.x
            # print(vertex_coords)
            # _, unique_indices = np.unique(vertex_coords, axis=0, return_index=True)
            # print(unique_indices)
            #
            # # Getting the vertex-to-dof map for vertices only
            # vertex_to_dof = dofmap.list.array[:mesh.topology.index_map(0).size_local]
            # # print(vertex_to_dof)
            # vertex_displacements = midpoint_displacement.reshape(-1, 3)[unique_indices]
            #
            # # print(vertex_displacements)
            # mesh.geometry.x[unique_indices] += vertex_displacements

            ############################################
            # print(np.shape(self._u.x.array))
            # print(np.shape(self._u.function_space.mesh.geometry.x))
            # midpoint_displacement= midpoint_displacement.reshape(-1, 3)
            self._u.function_space.mesh.geometry.x[:] += midpoint_displacement.reshape(-1, 3)
            # TODO: mesh update for all constraints
            ######################################

        # print(self._u.function_space.mesh.geometry.x[:])
        # print(self._u.x.array.reshape(-1, 3))

        # if len(self.laws) > 1:
        for k, (law, cells) in enumerate(self.laws):
            # TODO: test this!
            # Replace with `self._del_grad_u[k].interpolate(...)` in 0.8.0
            self.del_grad_u_expr.eval(
                cells, self._del_grad_u[k].x.array.reshape(cells.size, -1)
            )
            # self.strain_rotate(del_grad_u=self._del_grad_u[k].x.array, angle = np.pi/4)
            self._del_grad_u[k].x.scatter_forward()
            # print('del_grad_u',np.shape(self._del_grad_u[k].x.array))
            np.set_printoptions(threshold=np.inf)
            # print('del_grad_u dimension:', np.shape(self._del_grad_u[k].x.array))
            if len(self.laws) > 1:
                self.submesh_maps[k].map_to_child(self.stress_0, self._stress[k])
                stress_input = self._stress[k].x.array
                tangent_input = self._tangent[k].x.array
            else:
                self.stress_1.x.array[:] = self.stress_0.x.array
                self.stress_1.x.scatter_forward()
                stress_input = self.stress_1.x.array
                if self.co_rotation:
                    self.stress_rotate(del_grad_u=self._del_grad_u[k].x.array, mandel_stress=stress_input)
                    # TODO: create a sperate function outside this class
                tangent_input = self.tangent.x.array

            history_input = None
            if law.history_dim is not None:
                history_input = {}
                for key in law.history_dim:
                    self._history_1[k][key].x.array[:] = self._history_0[k][key].x.array
                    history_input[key] = self._history_1[k][key].x.array
            law.evaluate(
                self._time,
                self._del_grad_u[k].x.array,
                stress_input,
                tangent_input,
                history_input,
            )
            if len(self.laws) > 1:
                self.submesh_maps[k].map_to_parent(self._stress[k], self.stress_1)
                self.submesh_maps[k].map_to_parent(self._tangent[k], self.tangent)

        if self.mesh_update:
            self._u.function_space.mesh.geometry.x[:] -= midpoint_displacement.reshape(-1, 3)
            # mesh.geometry.x[unique_indices] -= vertex_displacements
        self.stress_1.x.scatter_forward()
        self.tangent.x.scatter_forward()

    def update(self) -> None:
        """
        Update the current displacement, stress and history.
        """

        if self.mesh_update:

            # Update to current configuration

            # current_displacement = self._u.x.array - self._u0.x.array

            V_CG = df.fem.VectorFunctionSpace(self._u.function_space.mesh, ("CG", 1))
            u_CG0 = df.fem.Function(V_CG)
            u_CG = df.fem.Function(V_CG)

            u_CG0.interpolate(self._u0)
            u_CG.interpolate(self._u)

            current_displacement = u_CG.x.array - u_CG0.x.array

            # ###################################################
            # dofmap = self._u.function_space.dofmap
            # mesh = self._u.function_space.mesh
            # dof_coordinates = self._u.function_space.tabulate_dof_coordinates().reshape((-1, 3))
            # vertex_coords = mesh.geometry.x
            # _, unique_indices = np.unique(vertex_coords, axis=0, return_index=True)
            # # print(unique_indices)
            #
            # # Getting the vertex-to-dof map for vertices only
            # vertex_to_dof = dofmap.list.array[:mesh.topology.index_map(0).size_local]
            # # print(vertex_to_dof)
            # vertex_displacements = current_displacement.reshape(-1, 3)[unique_indices]
            # # vertex_displacements = current_displacement.reshape(-1, 3)[vertex_to_dof]
            ##########################################################################
            # Update the mesh geometry to the deformed configuration
            # # print(midpoint_displacement)
            # current_displacement = current_displacement.reshape(-1, 3)
            # self._u.function_space.mesh.geometry.x[unique_indices] += vertex_displacements
            self._u.function_space.mesh.geometry.x[:] += current_displacement.reshape(-1, 3)
            # print(self._u.function_space.mesh.geometry.x[:])

            # print(self._u.function_space.mesh.geometry.x[:])

        self._u0.x.array[:] = self._u.x.array
        self._u0.x.scatter_forward()

        self.stress_0.x.array[:] = self.stress_1.x.array
        print(self.stress_0.x.array)
        self.stress_0.x.scatter_forward()

        for k, (law, _) in enumerate(self.laws):
            if law.history_dim is not None:
                for key in law.history_dim:
                    self._history_0[k][key].x.array[:] = self._history_1[k][key].x.array
                    self._history_0[k][key].x.scatter_forward()


    # TODO: write a function that rotates the strain, stresses, tangent and history that is being input to the material law?
    # TODO: or just rotate the stresses being input to the material model
    def stress_rotate(self, del_grad_u, mandel_stress):
        # TODO the stress that we get here is mandel stress already. convert it into 3x3 form using appropriate expressions
        #I2 = np.zeros((3,3), dtype=np.float64)  # Identity of rank 2 tensor
        #I2[0, 0] = 1.0
        #I2[1, 1] = 1.0
        #I2[2, 2] = 1.0
        I2 = np.eye(3,3)
        shape = int(np.shape(del_grad_u)[0]/9)

        mandel_stress = mandel_stress.reshape(-1,6)
        # print(np.shape(del_grad_u))



        stress = np.zeros((shape, 3,3), dtype=np.float64)

        stress[:, 0,0] = mandel_stress[:, 0]
        stress[:, 1,1] = mandel_stress[:, 1]
        stress[:, 2,2] = mandel_stress[:, 2]
        stress[:, 0,1] = 1 / 2 ** 0.5 * (mandel_stress[:, 3])
        stress[:, 1,2] = 1 / 2 ** 0.5 * (mandel_stress[:, 4])
        stress[:, 0,2] = 1 / 2 ** 0.5 * (mandel_stress[:, 5])
        stress[:, 1,0] = stress[:, 0,1]
        stress[:, 2,1] = stress[:, 1,2]
        stress[:, 2,0] = stress[:, 0,2]


        # print(del_grad_u)
        # g = del_grad_u.reshape(-1, 9)
        g = del_grad_u.reshape(shape,3,3)
        strains = del_grad_u.reshape(shape,3,3)
        # print(g)
        #rotated_stress_matrix = []

        for n, eps in enumerate(g):
            # strain_increment = (eps + np.transpose(eps))/2
            rotation_increment = (eps - np.transpose(eps))/2
            # print(rotation_increment)
            # print('rotation increment', rotation_increment)
            Q_matrix = I2 + (np.linalg.inv(I2 - 0.5*rotation_increment)) @ rotation_increment

            # ########################################################################
            #
            # def skew_symmetric_to_vector(skew_sym_matrix):
            #     # Extract the vector from the skew-symmetric matrix
            #     return np.array([skew_sym_matrix[2, 1], skew_sym_matrix[0, 2], skew_sym_matrix[1, 0]])
            #
            # def vector_to_skew_symmetric(vector):
            #     # Convert a vector to a skew-symmetric matrix
            #     return np.array([[0, -vector[2], vector[1]],
            #                      [vector[2], 0, -vector[0]],
            #                      [-vector[1], vector[0], 0]])
            #
            # def rotation_matrix_from_axis_angle(axis, angle):
            #     # Compute the rotation matrix from axis and angle using Rodrigues' formula
            #     K = vector_to_skew_symmetric(axis)
            #     I = np.eye(3)
            #     return I + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            #
            # # Step 1: Extract the rotation axis and angle from the skew-symmetric matrix
            # rotation_vector = skew_symmetric_to_vector(rotation_increment)
            #
            # # The angle of rotation is the magnitude of the rotation vector
            # rotation_angle = np.linalg.norm(rotation_vector)
            #
            #
            # if rotation_angle != 0:
            #     # Step 2: Normalize the rotation vector to get the axis of rotation
            #     rotation_axis = rotation_vector / rotation_angle
            #
            #     # Step 3: Halve the rotation angle
            #     half_rotation_angle = rotation_angle / 2
            #
            #     # Step 4: Compute the rotation matrix for half the angle
            #     Q_matrix_half_angle = rotation_matrix_from_axis_angle(rotation_axis, half_rotation_angle)
            #
            #     print("rotation angle is ", rotation_increment)
            #
            # else:
            #     # If there's no rotation, the matrix is just the identity
            #     Q_matrix_half_angle = np.eye(3)
            #
            # # Q_matrix_half_angle now contains the rotation matrix for half the angle
            #
            # ##########################################################################

            # Logarithm of the rotation matrix
            log_Q = logm(Q_matrix)

            # Halve the rotation
            log_Q_half = 0.5 * log_Q

            # Exponentiate to get the half-angle rotation matrix
            Q_half = expm(log_Q_half)

            theta = np.arctan2(Q_matrix[1, 0], Q_matrix[0, 0])

            # Calculate half angle rotation matrix
            Q_matrix_half_angle = np.array([
                [np.cos(theta / 2), -np.sin(theta / 2), 0],
                [np.sin(theta / 2), np.cos(theta / 2), 0],
                [0, 0, 1]
            ])
            rot_stress = Q_half.T @ stress[n,:,:] @ Q_half
            # strains_rotated = Q_matrix @ strains[n,:,:] @ Q_matrix.T
            # print(strains_rotated-eps)
            # print(rotation_increment)
            stress[n,:,:] = rot_stress
            # strains[n,:,:] = strains_rotated
            #rotated_stress_matrix.append(rot_stress)

        #rotated_stress_matrix = np.array(rotated_stress_matrix)
        # print(np.shape(rotated_stress_matrix))
        rotated_stress_mandel = np.zeros((shape,6), dtype=np.float64)

        rotated_stress_mandel[:, 0] = stress[:, 0,0]
        rotated_stress_mandel[:, 1] = stress[:, 1,1]
        rotated_stress_mandel[:, 2] = stress[:, 2,2]
        rotated_stress_mandel[:, 3] = 2 ** 0.5 * stress[:, 0,1]
        rotated_stress_mandel[:, 4] = 2 ** 0.5 * stress[:, 1,2]
        rotated_stress_mandel[:, 5] = 2 ** 0.5 * stress[:, 0,2]

        # print('mandel stress rotated ################',rotated_stress_mandel)
        # mandel_stress = mandel_stress.flatten()
        mandel_stress[:,:] = rotated_stress_mandel
        # del_grad_u[:] = strains.flatten()
        # print(del_grad_u - g.flatten())

        # print('mandel stress : ',np.shape(mandel_stress))

        #return rotated_stress_mandel # TODO match stress shapes

    def strain_rotate(self, del_grad_u,angle):
        # TODO the stress that we get here is mandel stress already. convert it into 3x3 form using appropriate expressions
        # I2 = np.zeros((3,3), dtype=np.float64)  # Identity of rank 2 tensor
        # I2[0, 0] = 1.0
        # I2[1, 1] = 1.0
        # I2[2, 2] = 1.0
        # I2 = np.eye(3, 3)
        rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
        ])

        shape = int(np.shape(del_grad_u)[0] / 9)

        # mandel_stress = mandel_stress.reshape(-1, 6)
        # # print(np.shape(del_grad_u))
        #
        # stress = np.zeros((shape, 3, 3), dtype=np.float64)
        #
        # stress[:, 0, 0] = mandel_stress[:, 0]
        # stress[:, 1, 1] = mandel_stress[:, 1]
        # stress[:, 2, 2] = mandel_stress[:, 2]
        # stress[:, 0, 1] = 1 / 2 ** 0.5 * (mandel_stress[:, 3])
        # stress[:, 1, 2] = 1 / 2 ** 0.5 * (mandel_stress[:, 4])
        # stress[:, 0, 2] = 1 / 2 ** 0.5 * (mandel_stress[:, 5])
        # stress[:, 1, 0] = stress[:, 0, 1]
        # stress[:, 2, 1] = stress[:, 1, 2]
        # stress[:, 2, 0] = stress[:, 0, 2]

        # print(del_grad_u)
        # g = del_grad_u.reshape(-1, 9)
        g = del_grad_u.reshape(shape, 3, 3)
        strains = del_grad_u.reshape(shape, 3, 3)
        # print(g)
        # rotated_stress_matrix = []

        for n, eps in enumerate(g):
            # strain_increment = (eps + np.transpose(eps))/2
            # rotation_increment = (eps - np.transpose(eps)) / 2
            # print(rotation_increment)
            # print('rotation increment', rotation_increment)
            # Q_matrix = I2 + (np.linalg.inv(I2 - 0.5 * rotation_increment)) @ rotation_increment
            # rot_stress = Q_matrix @ stress[n, :, :] @ Q_matrix.T
            strains_rotated = rot_matrix @ strains[n, :, :] @ rot_matrix.T
            # print(strains_rotated-eps)
            # print(rotation_increment)
            # stress[n, :, :] = rot_stress
            strains[n, :, :] = strains_rotated
            # rotated_stress_matrix.append(rot_stress)

        # rotated_stress_matrix = np.array(rotated_stress_matrix)
        # print(np.shape(rotated_stress_matrix))
        # rotated_stress_mandel = np.zeros((shape, 6), dtype=np.float64)
        #
        # rotated_stress_mandel[:, 0] = stress[:, 0, 0]
        # rotated_stress_mandel[:, 1] = stress[:, 1, 1]
        # rotated_stress_mandel[:, 2] = stress[:, 2, 2]
        # rotated_stress_mandel[:, 3] = 2 ** 0.5 * stress[:, 0, 1]
        # rotated_stress_mandel[:, 4] = 2 ** 0.5 * stress[:, 1, 2]
        # rotated_stress_mandel[:, 5] = 2 ** 0.5 * stress[:, 0, 2]

        # print('mandel stress rotated ################',rotated_stress_mandel)
        # mandel_stress = mandel_stress.flatten()
        # mandel_stress[:, :] = rotated_stress_mandel
        del_grad_u[:] = strains.flatten()
        # print(del_grad_u - g.flatten())

        # print('mandel stress : ',np.shape(mandel_stress))
        print('##############called################')

        # return rotated_stress_mandel # TODO match stress shapes



