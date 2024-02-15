# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Stokes equations using Taylor-Hood elements
#
# This demo is implemented in {download}`demo_stokes.py`. It shows how
# to solve the Stokes problem using Taylor-Hood elements using different
# linear solvers.
#
# ## Equation and problem definition
#
# ### Strong formulation
#
# $$
# \begin{align}
#   - \nabla \cdot (\nabla u + p I) &= f \quad {\rm in} \ \Omega,\\
#   \nabla \cdot u &= 0 \quad {\rm in} \ \Omega.
# \end{align}
# $$
#
# with conditions on the boundary $\partial \Omega = \Gamma_{D} \cup
# \Gamma_{N}$ of the form:
#
# $$
# \begin{align}
#   u &= u_0 \quad {\rm on} \ \Gamma_{D},\\
#   \nabla u \cdot n + p n &= g \,   \quad\;\; {\rm on} \ \Gamma_{N}.
# \end{align}
# $$
#
# ```{note}
# The sign of the pressure has been changed from the usual
# definition. This is to generate have a symmetric system
# of equations.
# ```
#
# ### Weak formulation
#
# The weak formulation reads: find $(u, p) \in V \times Q$ such that
#
# $$
# a((u, p), (v, q)) = L((v, q)) \quad \forall  (v, q) \in V \times Q
# $$
#
# where
#
# $$
# \begin{align}
#   a((u, p), (v, q)) &:= \int_{\Omega} \nabla u \cdot \nabla v -
#            \nabla \cdot v \ p + \nabla \cdot u \ q \, {\rm d} x,
#   L((v, q)) &:= \int_{\Omega} f \cdot v \, {\rm d} x + \int_{\partial
#            \Omega_N} g \cdot v \, {\rm d} s.
# \end{align}
# $$
#
# ### Domain and boundary conditions
#
# We consider the lid-driven cavity problem with the following
# domain and boundary conditions:
#
# - $\Omega := [0,1]\times[0,1]$ (a unit square)
# - $\Gamma_D := \partial \Omega$
# - $u_0 := (1, 0)^\top$ at $x_1 = 1$ and $u_0 = (0, 0)^\top$ otherwise
# - $f := (0, 0)^\top$
#
#
# ## Implementation
#
# The Stokes problem using Taylor-Hood elements is solved using:
# 1. [Block preconditioner using PETSc MatNest and VecNest data
#    structures. Each 'block' is a standalone object.](#nested-matrix-solver)
# 1. [Block preconditioner with the `u` and `p` fields stored block-wise
#    in a single matrix](#monolithic-block-iterative-solver)
# 1. [Direct solver with the `u` and `p` fields stored block-wise in a
#    single matrix](#monolithic-block-direct-solver)
# 1. [Direct solver with the `u` and `p` fields stored block-wise in a
#    single matrix](#non-blocked-direct-solver)
#
# The required modules are first imported:

NUM_SAMPLES=600
import numpy as np

u_all=np.zeros((NUM_SAMPLES,1089))
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem.petsc import LinearProblem
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, la
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)
from tqdm import trange
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner

# We create a {py:class}`Mesh <dolfinx.mesh.Mesh>`, define functions for
# locating geometrically subsets of the boundary, and define a function
# for the  velocity on the lid:

# +
# Create mesh
msh = create_rectangle(
    MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [32, 32], CellType.triangle
)

xdmf = XDMFFile(msh.comm, "laplace.xdmf", "w")
xdmf.write_mesh(msh)


# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)), np.isclose(x[1], 0.0)
    )


# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)




# -

# Two {py:class}`function spaces <dolfinx.fem.FunctionSpace>` are
# defined using different finite elements. `P2` corresponds to a
# continuous piecewise quadratic basis (vector) and `P1` to a continuous
# piecewise linear basis (scalar).


P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
V= functionspace(msh, P2)

# Boundary conditions for the velocity field are defined:

# +
# No-slip condition on boundaries where x = 0, x = 1, and y = 0
# -

# The bilinear and linear forms for the Stokes equations are defined
# using a a blocked structure:

# +
# Define variational problem
for i in trange(50,NUM_SAMPLES+50):
# No-slip condition on boundaries where x = 0, x = 1, and y = 0
    noslip = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)  # type: ignore
    facets = locate_entities_boundary(msh, 1, noslip_boundary)
    bc0 = dirichletbc(noslip, locate_dofs_topological(V, 1, facets), V)
# Lid velocity
    def lid_velocity_expression(x):
        return np.stack((i/NUM_SAMPLES*np.ones(x.shape[1]), np.zeros(x.shape[1])))

    # Driving (lid) velocity condition on top boundary (y = 1)
    lid_velocity = Function(V)
    lid_velocity.interpolate(lid_velocity_expression)
    facets = locate_entities_boundary(msh, 1, lid)
    bc1 = dirichletbc(lid_velocity, locate_dofs_topological(V, 1, facets))

# Collect Dirichlet boundary conditions
    bcs = [bc0, bc1]

    u, = ufl.TrialFunction(V),
    v = ufl.TestFunction(V)
    f = Constant(msh, (PETSc.ScalarType(i/NUM_SAMPLES), PETSc.ScalarType(0)))  # type: ignore

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    problem = LinearProblem(a, L, bcs=[bc0,bc1], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    P1 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    u1 = Function(functionspace(msh, P1))
    u1.interpolate(uh)
    # Save solution to file in XDMF format for visualization, e.g. with
    # ParaView. Before writing to file, ghost values are updated using
    # `scatter_forward`.

    # Compute norms of the solution vectors
    #print(np.sum(np.abs(u.vector)))
    u1_arr=u1.x.array
    
    tmp=u1_arr.reshape(-1,2)
    u_normed=np.linalg.norm(tmp,axis=1)
    P1F=element("Lagrange", msh.basix_cell(), 1)
    V1F=functionspace(msh,P1F)

    v1f=Function(V1F)
    v1f.vector[:]=u_normed

    xdmf.write_function(v1f, i)

    u_all[i-50]=v1f.x.array
np.save("laplace.npy",u_all)

xdmf.close()