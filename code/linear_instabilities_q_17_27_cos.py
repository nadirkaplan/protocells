#This code solves the dimenionless equation(Eq. 25 from SI) -- \frac{\partial h_1}{\partial t} = \beta*(\frac{\partial^6 h_1}{\partial x^6}) - 3*(\frac{\partial^2 h_1}{\partial x^2})
#using finite element method for wave number q = 11*(\pi/2)
#The sixth order equation is written in the mixed form of 3 coupled second order equations.  
#\frac{\partial h_1}{\partial t} = \beta*(\frac{\partial^2 a}{\partial x^2}) - 3*b
#\frac{\partial^2 b}{\partial x^2} = a
#\frac{\partial^2 h_1}{\partial x^2} = b
#For the time integration Backwards differentiation formula of the 5th order(BDF5) was implemented.
#The solution is saved in the VTK file for later visualization. The generated ".pvd" files can be loaded into Paraview(5.9.0) to see the time evolution of the desired quantities. 


#importing required libraries
from fenics import*
import numpy as np
import matplotlib as plt
from time import perf_counter
from datetime import date
import pdb
from ufl import sin
from ufl import cos
from ufl import exp


t1_start = perf_counter()

T = 0.005  #final time
dt = Constant(1e-5)  

#defining all the constants
ccn = Constant(-1.)
ccn_1 = Constant(0.)
ccn_2 = Constant(0.)
ccn_3 = Constant(0.)
ccn_4 = Constant(0.)
ccr = Constant(1.)
A_h = Constant(2.081e-21)
pi = Constant(3.1415)
beta = Constant(1.e-5) 
q = Constant(11*pi/2)
print(float(q))

#defining the mesh
mesh = IntervalMesh(2000, 0, 1)
P1 = FiniteElement('P', mesh.ufl_cell(), 1)
element = MixedElement([P1,P1,P1])
V = FunctionSpace(mesh, element)
W = FunctionSpace(mesh,'P',1)


v1, v2, v3 = TestFunctions(V)

u = Function(V)
un = Function(V)
undx = Function(W)
un_1 = Function(V)
un_2 = Function(V)
un_3 = Function(V)
un_4 = Function(V)
dv = TrialFunction(V)

u0 = Expression(('0.001*cos(q*x[0])','-0.001*pow(q,2)*cos(q*x[0])', '0.001*pow(q,4)*cos(q*x[0])'), degree=1, q = q)

un = interpolate(u0, V)

u1, u2, u3 = split(u); #splitting u
#u1 = h_1
#u2 = b
#u3 = a
un1, un2, un3 = split(un);
un_11, un_12, un_13 = split(un_1);
un_21, un_22, un_23 = split(un_2);
un_31, un_32, un_33 = split(un_3);
un_41, un_42, un_43 = split(un_4);

def BDFparameters(nn, ccn, ccn_1, ccn_2, ccn_3, ccn_4, ccr):
    if nn==0:
        ccn.assign(Constant(-1.)); ccr.assign(Constant(1.)); ccn_1.assign(Constant(0.))
        ccn_2.assign(Constant(0.)); ccn_3.assign(Constant(0.)); ccn_4.assign(Constant(0.))                
    elif nn==1:
        ccn.assign(Constant(-4./3.)); ccr.assign(Constant(2./3.)); ccn_1.assign(Constant(1./3.))
        ccn_2.assign(Constant(0.)); ccn_3.assign(Constant(0.)); ccn_4.assign(Constant(0.))
    elif nn==2:
        ccn.assign(Constant(-18./11.)); ccr.assign(Constant(6./11.)); ccn_1.assign(Constant(9./11.))
        ccn_2.assign(Constant(-2./11.)); ccn_3.assign(Constant(0.)); ccn_4.assign(Constant(0.))
    elif nn==3:
        ccn.assign(Constant(-48./25.)); ccr.assign(Constant(12./25.)); ccn_1.assign(Constant(36./25.))
        ccn_2.assign(Constant(-16./25.)); ccn_3.assign(Constant(3./25.)); ccn_4.assign(Constant(0.))
    else:
        ccn.assign(Constant(-300./137.)); ccr.assign(Constant(60./137.)); ccn_1.assign(Constant(300./137.))
        ccn_2.assign(Constant(-200./137.)); ccn_3.assign(Constant(75./137.)); ccn_4.assign(Constant(-12./137.))

def BDFvariables(nn, u, un, un_1, un_2, un_3, un_4):
    if nn==0:
        un.assign(u)    
    elif nn==1:
        un_1.assign(un); un.assign(u)  
    elif nn==2:
        un_2.assign(un_1); un_1.assign(un); un.assign(u)  
    elif nn==3:
        un_3.assign(un_2); un_2.assign(un_1); un_1.assign(un); un.assign(u)  
    else:
        un_4.assign(un_3); un_3.assign(un_2); un_2.assign(un_1); un_1.assign(un); un.assign(u)
        
        
#boundary terms in weak form

n = FacetNormal(mesh)
g1 = Constant(0.)
g2 = Constant(0.)
g3 = Constant(0.)

boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1)

class BoundaryX0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x, 0, DOLFIN_EPS)

bx0 = BoundaryX0()
bx0.mark(boundary_markers, 0)

class BoundaryX1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x, 1., DOLFIN_EPS)

bx1 = BoundaryX1()
bx1.mark(boundary_markers, 1)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

#weak statement of the equations

F1 = (u1 + ccn*un1 + ccn_1*un_11 + ccn_2*un_21 + ccn_3*un_31 + ccn_4*un_41)*v1*dx + dt*(ccr*beta*inner(grad(u3),grad(v1))*dx + ccr*3*u2*v1*dx - ccr*beta*g3*v1*ds(0))
F2 = u2*v2*dx + inner(grad(u1),grad(v2))*dx - g1*v2*ds(0)
F3 = u3*v3*dx + inner(grad(u2),grad(v3))*dx - g2*v3*ds(0)

F  = F1 + F2 + F3

#creating vtk file for visualization

vtkfile_u1 = File('q_17_27/height.pvd')
vtkfile_u2 = File('q_17_27/second_derivative.pvd') 
vtkfile_u3 = File('q_17_27/fourth_derivative.pvd')
vtkfile_u4 = File('q_17_27/pressure.pvd')
vtkfile_u5 = File('q_17_27/velocity.pvd')

bc_u1 = DirichletBC(V.sub(0), Constant(0.), bx1)
bc_u2 = DirichletBC(V.sub(1), Constant(0.), bx1)
bc_u3 = DirichletBC(V.sub(2), Constant(0.), bx1)

t = 0
nn = 0

J = derivative(F, u, dv)

while t < T:
    
    
    _u1,_u2,_u3 = un.split()

    p = project((beta*un3-3*un1),W) #p = pressure
    p.rename('p','p')
    vel = project((3*un1.dx(0)-beta*un3.dx(0)),W)  #vel = flowspeed
    vel.rename('vel','vel')
    vtkfile_u1 << (_u1,t)
    vtkfile_u2 << (_u2,t)
    vtkfile_u3 << (_u3,t)
    vtkfile_u4 << (p,t)
    vtkfile_u5 << (vel,t)
    
    t  += float(dt)
    
    bcs = [bc_u1, bc_u2, bc_u3]
    
    BDFparameters(nn, ccn, ccn_1, ccn_2, ccn_3, ccn_4, ccr)
    
    problem = NonlinearVariationalProblem(F,u,bcs,J=J)
    solver  = NonlinearVariationalSolver(problem)
    
    
    prm = solver.parameters
    prm['nonlinear_solver']='newton'
    prm["newton_solver"]["absolute_tolerance"]= 1e-9   
    prm["newton_solver"]["relative_tolerance"] = 1e-9 
    prm["newton_solver"]["maximum_iterations"] = 5
    prm["newton_solver"]["error_on_nonconvergence"]=False
    
    (Nit2, conv2)=solver.solve()
    
    if conv2 == False:
        break
    
    
    nn += 1
    
    BDFvariables(nn, u, un, un_1, un_2, un_3, un_4)

t1_end = perf_counter()    

print("Total runtime in seconds =", t1_end-t1_start)  

pdb.set_trace()
