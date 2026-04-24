# Python class that implements the Q-tensor elasticity tissue model

import sys
import copy
sys.path.insert(1, '../LdG/')
sys.path.insert(1, '../Newton_Methods/')
#sys.path.insert(1, '../VTK/')

# from termcolor import colored, cprint

import numpy as np
import scipy as sp

#import netgen.gui
from ngsolve import *
#from xfem import *
#from xfem.lsetcurv import *

#from ngsolve.webgui import Draw# as webDraw

# SWW: this interferes with netgen gui interface
# import matplotlib.pyplot as plt

# from VTKwrite.interface import timeseries_unstructuredGrid
# from vtk_extract import *

#from nonlinear_min_funcs import *
from mesh_1D import *
#from LdG_energies import *

import json

import time as py_time

class GRB_model:
    def __init__(self,dim,logger=None):
        self.name = "Gamma-ray burst model simulation object"
        # what is the dimension
        self.dim = dim

        # store reference to the logger
        self.logger = logger

        # a dict with parameters and coefficients
        self.data = []

        # domain/mesh
        self.x_vec         = []
        self.Om_mesh       = []
        self.boundary_name = "left|right"

        # the finite element spaces
        self.fes_u  = []

        # grid functions
        self.dirichlet_bdy_u = []

        # self.fes_rhs_Om   = []
        # self.fes_rhs_Gm   = []
        # grid functions (Landau--de Gennes solution)
        self.gf_u           = []
        self.gf_u_prev      = []
        # grid function for weak anchoring condition
        self.gf_u_BC = []
        # grid function for initial condition of Q (in vector form)
        self.gf_u_init   = []
        # # grid functions (control variables)
        # self.CF_Q_Gm  = []
        # self.CF_qq_Gm = []
        # self.gf_qq_Gm = []

        # the energy functional
        #self.LdG_energy = []

        # we want to solve:
        #      a( (uu,qq), (ww,pp) ) = L( (ww,pp) ),
        # where the a(.,.) term may be nonlinear in the (uu,qq) argument;
        # note: L is a linear functional.
        # the (non-linear) a(.,.) form (Left-Hand-Side)
        self.a_form = []
        # the linear L(.) form (Right-Hand-Side)
        self.linform_RHS = []

        # keep track of time as a parameters
        self.t_val = Parameter(0.0)

        # a dict with parameters and coefficients
        self.solver_param = []

        # array of time-values
        self.time_vec     = []
        # # array of solution vectors vs. time indices
        # self.qq_time_array = []

        # VTK writer object
        self.ts_ugrid = []

    def __repr__(self):
        print(self)
        return ''

    def __str__(self):
        OUT_STR = ("Here is the Gamma-ray burst model simulation object data:" + "\n"
                 + "---------------------------------------------------------" + "\n"
                 + "Dimension: " + str(self.dim) + "\n"
                 + "Problem Data:" + "\n"
                 + str(self.data) + "\n"
                 # + "Unfitted Object:" + "\n"
                 # + str(self.BG_LS) + "\n"
                 # + "Finite Element Spaces:" + "\n"
                 # + str(self.fes_Q) + "\n"
                 # + str(self.fes_q_scalar) + "\n"
                 # + str(self.fes_uu) + "\n"
                 # + "grid functions: " + "gf_qq, gf_qq_prev, gf_qq_prev_norm, " + "\n"
                 # + "Boundary Condition: " + "gf_qq_box_BC, " + "\n"
                 # + "Initial Condition:  " + "gf_qq_init, " + "\n"
                 # + "CF functions for RHS for given exact soln: " + "CF_rhs_Om, CF_rhs_Gm, " + "\n"
                 # + "CF functions for normal anchoring function: " + "CF_Q_Gm, CF_qq_Gm, " + "\n"
                 # + "the non-linear a form: nonlinear_a_form" + "\n"
                 # + str(self.nonlinear_a_form) + "\n"
                 # + "the Newton a form: Newton_a_form" + "\n"
                 # + str(self.Newton_a_form) + "\n"
                 # + "the positive definite Hessian form: Pos_Def_a_form" + "\n"
                 # + str(self.Pos_Def_a_form) + "\n"
                 # + "Stokes object: Stokes_obj" + "\n"
                 # + str(self.Stokes_obj) + "\n"
                 # + "P1 FES and grid function for velocity on zero-level set: P1_vec_fes, gf_velocity" + "\n"
                 # + str(self.P1_vec_fes) + "\n"
                 # + str(self.gf_velocity) + "\n"
                 + "solver parameters: solver_param" + "\n"
                 + str(self.solver_param) + "\n"
                 + "array of time-values:" + "\n"
                 + str(self.time_vec) + "\n")
        return OUT_STR

    # set the internal data
    def set_data(self,prob_data=None,filename=""):
        # a dict with parameters and coefficients

        if prob_data is None:
            # default data
            init_cond_u_str = "CoefficientFunction( 0.0 )"
            bdy_cond_u_str  = "CoefficientFunction( 0.0 )"
            self.data = {'dim' : self.dim, 'FE_degree' : 1, 'linearsolve' : True, \
                         'time_scale_u' : 1.0, 'Lorentz_factor' : 1.0, \
                         'init_cond_u' : init_cond_u_str, 'bdy_cond_u' : bdy_cond_u_str, \
                         'init_time' : 0, 'num_steps' : 10, 'final_time' : 1.0, 'dt' : []}
        else:
            self.data = prob_data

        # check that data is valid
        if self.data['num_steps'] <= 1:
            self.logger.error("num_steps must be > 1!")
            sys.exit("num_steps must be > 1!")
        # add other checks....

        # create the array of time values (num_steps+1)
        self.time_vec = np.linspace(self.data['init_time'], self.data['final_time'], num=(self.data['num_steps']+1))

        # compute the time-step directly
        self.data['dt'] = float(self.time_vec[1] - self.time_vec[0])

        # # check time-step stability condition
        # #stab_cond = eps**2 - dt
        # stab_cond = (1/self.data['dt']) - ( dw_coefs['A'] + dw_coefs['B']**2 / (3*dw_coefs['C']) )/(self.data['eta_dw']**2)
        # if stab_cond < 0.0:
            # self.logger.warning(" ")
            # self.logger.warning("WARNING!")
            # self.logger.warning("time-step dt is not small enough to be energy stable!")
            # self.logger.warning("-----------------------------------------------------")
        # else:
            # self.logger.info(" ")
            # self.logger.info("Note:")
            # self.logger.info("time-step dt satisfies stability condition.")
            # self.logger.info("-----------------------------------------------------")

        if filename != "":
            # create json object from dictionary
            prob_data_json = json.dumps(self.data)
            # open file for writing, "w": filename="XXX.json"
            f_data = open(filename + ".json","w")
            # write json object to file
            f_data.write(prob_data_json)
            # close file
            f_data.close()

    # # just to have a default function
    # def default_u_BC_func(self,t_val,gf_u_BC):
        # #
        # gf_u_BC.Interpolate(0)

    # create the global mesh
    def define_interval_mesh(self,left_pt,right_pt,max_mesh_size):

        # --------------------------------------------------
        # Mesh: x in (left_pt, right_pt) (interval)
        # --------------------------------------------------
        domain_len = right_pt - left_pt
        num_elem = round(domain_len / max_mesh_size)
        self.x_vec = np.linspace(left_pt, right_pt, num=num_elem+1)
        self.Om_mesh = define_1D_mesh(self.x_vec)
        
        print(self.Om_mesh.GetBoundaries())

    # save the global mesh
    def save_Om_mesh(self,filename):
        #
        self.Om_mesh.ngmesh.Save(filename)

    # load the global mesh
    def load_Om_mesh(self,filename):
        #
        if self.dim==1:
            # make temporary mesh
            #self.Om_mesh = Mesh(unit_square.GenerateMesh(maxh=1.0))
            self.Om_mesh = Mesh(filename)
        else:
            self.logger.error("ERROR in dimension for loading mesh!")

        #self.Om_mesh.ngmesh.Load(filename)

    # define all finite element spaces
    def define_finite_element_spaces(self,dirichlet_u_BCs=[]):
        # make sure the mesh has been set!

        self.fes_u = H1(self.Om_mesh, order=self.data['FE_degree'], dirichlet=dirichlet_u_BCs)
        self.dirichlet_bdy_u = dirichlet_u_BCs

        self.gf_u_prev = GridFunction(self.fes_u, name="u_prev")
        self.gf_u_prev.Set(0.0)
        self.gf_u      = GridFunction(self.fes_u, name="u")
        self.gf_u.Set(0.0)
        self.gf_u_init = GridFunction(self.fes_u, name="u_init")
        self.gf_u_BC   = GridFunction(self.fes_u, name="u_BC")

        # set initial condition
        init_u_CF = eval(self.data['init_cond_u'])
        self.gf_u_init.Set(init_u_CF)

        # set boundary condition
        BC_u_CF = eval(self.data['bdy_cond_u'])
        self.gf_u_BC.Set(BC_u_CF)

    # # set the various pieces needed for the solver
    # def set_solver_info(self,solver_param=None,filename=""):
        # # a dict with parameters and coefficients

        # if solver_param is None:
            # # default parameters
            # self.solver_param = {'direct_method' : "sparsecholesky", 'iterative_method' : "", 'preconditioner' : "", \
                                 # 'func' : 'Newtons_Method', 'tol' : 1e-10, 'maxits' : 30}
        # else:
            # self.solver_param = solver_param

        # # check that data is valid
        # if (type(self.solver_param['tol']) != float):
            # self.logger.error("tol invalid!")
            # sys.exit("tol invalid!")
        # # add other checks....

        # if filename != "":
            # # create json object from dictionary
            # solver_param_json = json.dumps(self.solver_param)
            # # open file for writing, "w": filename="XXX.json"
            # f_data = open(filename,"w")
            # # write json object to file
            # f_data.write(solver_param_json)
            # # close file
            # f_data.close()

    # def define_model(self,Pm_Data,deriv_str=""):
        # # this will define either the energy, the (non-linear) first variation,
        # # or the linearized second variation
        # # SWW: NOT USED!

        # # setup weak formulation
        # qq = self.fes_V.TrialFunction()
        # pp = self.fes_V.TestFunction()

        # # setup the non-linear a form for evaluating the residual
        # # energy:
        # # a += ( (1/2)*W(\partial Q) + (1/dt)*(1/2)*|Q-Q_prev|^2 + (1/(eps**2)) * \psi(Q) - eta_Om * (U_Om:Q) ) * dx
        # # a += eta_Gm * (1/2)*|Q-U_Gm|^2 ds
        # # 1st variation (non-linear):
        # # a += ( bform(\partial Q, \partial P) + (1/dt)*(Q-Q_prev):P + (1/(eps**2)) * \psi'(Q):P - eta_Om * (U_Om:P) ) * dx
        # # a += eta_Gm * (Q-U_Gm):P ds
        # # 2nd variation (linearized):
        # # a += ( bform(\partial Q, \partial P) + (1/dt)*Q:P + (1/(eps**2)) * P:\psi''(\hat{Q}):Q ) * dx
        # # a += eta_Gm * Q:P ds

        # grad_energy_T0 = elastic_term(qq,pp,deriv_type=deriv_str,coefs=Pm_Data['Q_elastic_coefs'])
        # min_move_T0 = min_movement_term(Pm_Data['dt'],qq,self.gf_qq_prev,pp,deriv_type=deriv_str)
        # double_well_potential_T0 = double_well_bulk_potential_term(qq,self.gf_qq,pp,deriv_type=deriv_str,coefs=Pm_Data['dw_coefs'])
        # bulk_ctrl_T0 = bulk_control_term(qq,self.gf_rhs_Om,pp,deriv_type=deriv_str)
        # model_term_dx = grad_energy_T0 + min_move_T0 + (1/(Pm_Data['eta_dw']**2)) * double_well_potential_T0 \
                        # - Pm_Data['eta_Om'] * bulk_ctrl_T0
        # bdy_ctrl_T0 = bdy_control_term(qq,self.gf_rhs_Gm,pp,deriv_type=deriv_str)
        # model_term_ds = Pm_Data['eta_Gm'] * bdy_ctrl_T0

        # return model_term_dx, model_term_ds

    # def define_PDE_form_exact_soln(self):
        # # this defines the big bilinear and linear forms
        # # note: the internal data needs to be set before this is executed

        # # need the gfvv velocity from Stokes
        # div_vel = InnerProduct( Grad(self.Stokes_obj.gfvv), Id(self.dim) )
        # # divergence should be zero
        # W_ten = vort(self.Stokes_obj.gfvv)
        # S_ten = eps(self.Stokes_obj.gfvv)

        # FIX

        # # construct (bi-)linear forms
        # # a_form = RestrictedBilinearForm(self.background_fes, symmetric=False, element_restriction=self.els_outer,
                                              # # facet_restriction=None, check_unused=False)
        # self.nonlinear_a_form = RestrictedBilinearForm(self.fes_V, symmetric=False, \
                                # element_restriction=self.BG_LS.els_outer, \
                                # facet_restriction=self.BG_LS.facets_ring, check_unused=False)
        # self.linform_RHS = LinearForm(self.fes_V)

        # # setup weak formulation
        # qq = self.fes_V.TrialFunction()
        # QQ = Map_qq_to_QQ(qq,self.S0_basis)
        # pp = self.fes_V.TestFunction()
        # PP = Map_qq_to_QQ(pp,self.S0_basis)
        # h  = specialcf.mesh_size
        # dt = self.data['dt']
        # lcmole = self.data['beris_p0']
        # s0 = self.data['s0']

        # # time derivative
        # self.nonlinear_a_form += (1 / dt) * InnerProduct(qq, pp) * self.BG_LS.dx_Om
        # # "elastic" term
        # elastic_bi_form = elastic_term(qq,pp,deriv_type="1st_variation",coefs=self.data['Q_elastic_coefs'])
        # #self.nonlinear_a_form += InnerProduct(Grad(qq), Grad(pp)) * self.BG_LS.dx_Om
        # self.nonlinear_a_form += elastic_bi_form * self.BG_LS.dx_Om
        # # Beris-Edwards term
        # Beris_ten = W_ten * QQ - QQ * W_ten
        # # other terms
        # Beris_ten += lcmole*(S_ten * QQ + QQ * S_ten) - 2*lcmole*InnerProduct(S_ten, QQ)*QQ
        # self.nonlinear_a_form += -InnerProduct(Beris_ten, PP) * self.BG_LS.dx_Om
        # # double-well (bulk)
        # bulk_dbl_well_bi, bulk_dbl_well_lin = double_well_bulk_potential_term(qq,0*self.gf_qq,pp, \
                                              # deriv_type="1st_variation",coefs=self.data['bulk_coefs'])
        # self.nonlinear_a_form += (1/(self.data['bulk_eta']**2)) * bulk_dbl_well_bi * self.BG_LS.dx_Om

        # # weak anchoring (normal)
        # normal_anch_bi, normal_anch_lin = normal_anchoring(qq,self.CF_qq_Gm,pp,deriv_type="1st_variation")
        # self.nonlinear_a_form += self.data['anch_coefs']['w0'] * normal_anch_bi * self.BG_LS.ds_Gm
        # # weak anchoring (planar)
        # planar_anch_bi, planar_anch_lin = planar_anchoring(s0,qq,self.BG_LS.n_lset,pp,deriv_type="1st_variation")
        # self.nonlinear_a_form += self.data['anch_coefs']['w1'] * planar_anch_bi * self.BG_LS.ds_Gm
        # # double-well (bdy) part of planar anchoring
        # bdy_dbl_well_bi, bdy_dbl_well_lin = double_well_bdy_potential_term(s0,qq,0*self.gf_qq,pp, \
                                            # deriv_type="1st_variation")
        # self.nonlinear_a_form += (1/self.data['bdy_omega']) * bdy_dbl_well_bi * self.BG_LS.ds_Gm

        # # Nitsche term
        # self.nonlinear_a_form +=  (1/2) * InnerProduct(grad(qq)*self.Stokes_obj.gfvv, pp) * self.BG_LS.dx_Om
        # self.nonlinear_a_form += -(1/2) * InnerProduct(grad(pp)*self.Stokes_obj.gfvv, qq) * self.BG_LS.dx_Om
        # #self.nonlinear_a_form +=  (1/2) * div_vel * InnerProduct(qq, pp) * self.BG_LS.dx_Om
        # self.nonlinear_a_form +=  (1/2) * InnerProduct(self.Stokes_obj.gfvv, self.BG_LS.n_lset) \
                                        # * InnerProduct(qq, pp) * self.BG_LS.ds_Gm
        # # Ghost penalty stabilization (near the zero levelset)
        # self.nonlinear_a_form += (self.data['gamma_stab'] / h**2) * \
                                 # InnerProduct( (qq - qq.Other()) , (pp - pp.Other()) ) * self.BG_LS.dF

        # # RHS term
        # # forcing:
        # self.linform_RHS += InnerProduct(self.CF_rhs_Om, PP) * self.BG_LS.dx_Om
        # # time derivative
        # self.linform_RHS += (1 / dt) * InnerProduct(self.gf_qq_prev, pp) * self.BG_LS.dx_Om
        # # forcing:
        # self.linform_RHS += InnerProduct(self.CF_rhs_Gm, PP) * self.BG_LS.ds_Gm
        # # Beris-Edwards term
        # self.linform_RHS += (2/3)*lcmole * InnerProduct(S_ten, PP) * self.BG_LS.dx_Om
        # # weak anchoring (planar)
        # self.linform_RHS += self.data['anch_coefs']['w1'] * planar_anch_lin * self.BG_LS.ds_Gm

        # # Project the solution defined with respect to the last mesh deformation
        # # onto the the mesh with the current mesh deformation.
        # self.BG_LS.lsetmeshadap.ProjectOnUpdate(self.gf_qq)

    def define_PDE_form(self,logdiffcoef,t_esc,C_coef,B_coef,q_CF):
        # this defines the big bilinear and linear forms
        # note: the internal data needs to be set before this is executed

        # construct (bi-)linear forms
        # a_form = RestrictedBilinearForm(self.background_fes, symmetric=False, element_restriction=self.els_outer,
                                              # facet_restriction=None, check_unused=False)
        self.a_form = BilinearForm(self.fes_u, symmetric=False)
        # self.nonlinear_a_form = RestrictedBilinearForm(self.fes_V, symmetric=False, \
                                # element_restriction=self.BG_LS.els_outer, \
                                # facet_restriction=self.BG_LS.facets_ring, check_unused=False)
        self.linform_RHS = LinearForm(self.fes_u)

        self.t_val.Set(0.0)

        # test and trial functions
        u = self.fes_u.TrialFunction()
        v = self.fes_u.TestFunction()

        # parameters
        #h  = specialcf.mesh_size
        dt = self.data['dt']
        # ts_uu = self.data['time_scale_uu']
        # ts_Q  = self.data['time_scale_Q']

        self.a_form += (1/dt) * u*v *dx
        self.a_form += C_coef*u*v * dx
        self.a_form += -B_coef*u*grad(v)[0] * dx
        self.a_form += logdiffcoef * grad(u)[0]*grad(v)[0] * dx

        self.linform_RHS += ( (1/dt)*self.gf_u_prev*v + q_CF*v ) * dx

    def evaluate_residual(self,gf_u_input):
        # this applies the weak form to a given gf_u
        # returns a vector

        res_a = gf_u_input.vec.CreateVector()
        #res_lin = gf_u_input.vec.CreateVector()
        self.a_form.Assemble()
        res_a.data = self.a_form.mat * gf_u_input.vec
        #self.a_form.Apply(gf_u_input.vec, res_a)
        
        out_res = gf_u_input.vec.CreateVector()
        self.linform_RHS.Assemble()
        out_res.data = res_a.data - self.linform_RHS.vec
        
        return out_res

    # run a forward simulation
    def run_forward_sim(self, gf_uT, scene=None):
        # make sure all parameters, forms, and control functions are set!

        num_time_pts = self.time_vec.size
        dt = self.data['dt']

        self.t_val.Set(0.0)
        
        # set the initial condition
        self.gf_u_prev.Set(self.gf_u_init)
        self.gf_u.Set(self.gf_u_init)
        gf_uT.Set(self.gf_u, mdcomp=0)

        with TaskManager():
            for kk in range(1, num_time_pts):
                t_kk = self.time_vec[kk]
                self.t_val.Set(t_kk)
                
                self.a_form.Assemble()
                self.linform_RHS.Assemble()
                
                # no Dirichlet boundary conditions put in yet...
                
                inv = self.a_form.mat.Inverse(self.fes_u.FreeDofs(), inverse="umfpack")
                self.gf_u.vec.data = inv * self.linform_RHS.vec
                self.gf_u_prev.Set(self.gf_u)
                
                # # check residual
                # res_a = self.gf_u.vec.CreateVector()
                # res_a.data = self.a_form.mat * self.gf_u.vec
                # out_res = self.gf_u.vec.CreateVector()
                # res_a_np = res_a.FV().NumPy()[:]
                # linform_np = self.linform_RHS.vec.FV().NumPy()[:]
                # #out_res.data = res_a.data - self.linform_RHS.vec.data
                # RES_np = res_a_np - linform_np
                # print(np.linalg.norm(RES_np))
                
                # store the time-dependent solution
                gf_uT.Set(self.gf_u, mdcomp=kk)

                time_str = "t = " + "{:3f}".format(t_kk)
                print("\r" + time_str, end="")


        # for kk in range(1, num_time_pts):
            # t_kk = self.time_vec[kk]
            # if t_val is not None:
                # t_val.Set(t_kk)

            # solve_str = "Q-tissue Sim.: Solve at t = " + "{:5f}".format(t_kk)
            # self.logger.info(solve_str)
            # print("\r" + solve_str, end="")

            # # set the mesh deformation to last displacement
            # self.gf_mesh_deform.Interpolate(self.gf_uu)

            # # manually move gf_uu, gf_qq to gf_uu_prev, gf_qq_prev
            # self.gf_uu_prev.Interpolate(self.gf_uu)
            # self.gf_qq_prev.Interpolate(self.gf_qq)

            # # # compute velocity
            # # self.BG_LS.background_mesh.SetDeformation(self.BG_LS.mesh_deformation)
            # # #self.Stokes_obj.coef_g.Interpolate(Vel_func)
            # # self.Stokes_obj.coef_g.Interpolate(self.gf_velocity)
            # # self.BG_LS.background_mesh.UnsetDeformation()

            # # # do not put with TaskManager() above the level set stuff
            # # # (there is interference with KDTree)
            # # with TaskManager():

            # active_dofs = self.fes_uu_X_Q.FreeDofs()

            # #init_time = py_time.time()

            # if self.data['linearsolve']:
                # with TaskManager():
                    # # when assembling, it accounts for updated mesh deformation
                    # self.nonlinear_a_form.Assemble()
                    # self.linform_RHS.Assemble()

                # # include Dirichlet conditions
                # self.gf_qq.Set(self.zero_qq)
                # self.gf_qq.Interpolate(self.gf_qq_BC, definedon=self.dirichlet_bdy_Q)
                # self.gf_uu.Set(self.zero_uu)
                # self.uu_BC_func(t_kk,self.gf_uu_BC)
                # self.gf_uu.Interpolate(self.gf_uu_BC, definedon=self.dirichlet_bdy_uu)
                # RHS_vec = self.linform_RHS.vec \
                        # - self.nonlinear_a_form.mat * self.gf_soln_on_uu_X_Q.vec

                # # Solve linear system
                # #self.gf_qq.vec.data = self.nonlinear_a_form.mat.Inverse(active_dofs, inverse="umfpack") * self.linform_RHS.vec
                # #self.gf_qq.vec.data += self.nonlinear_a_form.mat.Inverse(active_dofs, inverse="umfpack") * RHS_vec
                # self.gf_soln_on_uu_X_Q.vec.data += LinearSolve(self.nonlinear_a_form,RHS_vec,active_dofs,direct_type="umfpack")
            # else:
                # # do something else (Newton?)
                # # self.linform_RHS.Assemble()

                # # include Dirichlet conditions
                # #self.gf_qq.Set(self.zero_qq)
                # #self.gf_qq.Interpolate(self.gf_qq_box_BC, definedon=self.dirichlet_bdy)
                # # SWW: do not destroy the initial guess for Newton's method
                # self.gf_qq.Interpolate(self.gf_qq_BC, definedon=self.dirichlet_bdy_Q)
                # self.uu_BC_func(t_kk,self.gf_uu_BC)
                # self.gf_uu.Interpolate(self.gf_uu_BC, definedon=self.dirichlet_bdy_uu)

                # #RHS_vec = self.linform_RHS.vec

                # Newtons_Method(self.nonlinear_a_form, self.nonlinear_a_form, active_dofs, \
                               # self.gf_soln_on_uu_X_Q, direct_type="umfpack", iter_method="", \
                               # precond="", tol=1e-10, maxits=30, logger=self.logger)

                # # Newtons_Method_2(self.nonlinear_a_form,RHS_vec,active_dofs,self.gf_qq, \
                                 # # direct_type="umfpack",logger=self.logger)

            # # SWW: do not move the TaskManager down...

            # # store the time-dependent solution
            # gf_uuT.Set(self.gf_uu, mdcomp=kk)
            # gf_qqT.Set(self.gf_qq, mdcomp=kk)

            # scene.Redraw()
        # #

    # this will write the VTK files
    def output_vtk(self, gf_uuT, gf_qqT, gf_QQ_comp, vtkout):
        # this requires the forward sim to already have run

        num_time_pts = self.time_vec.size

        ERROR

        MAT_fes = MatrixValued( self.fes_q_scalar )
        gf_QQ = GridFunction(MAT_fes)

        # print("Write VTK files:")
        # print("----------------")
        self.logger.info("Write VTK files:")
        self.logger.info("----------------")

        for kk in range(num_time_pts):
            t_kk = self.time_vec[kk]

            # load in current velocity
            self.gf_uu.Interpolate(gf_uuT.MDComponent(kk))

            # get current mesh deformation for current level set function
            self.gf_mesh_deform.Interpolate(self.gf_uu)

            # load in the forward solution variables
            self.gf_qq.Set(gf_qqT.MDComponent(kk))

            QQ_CF = 0*self.S0_basis[0]
            for cc in range(self.num_comp):
                QQ_CF += self.gf_qq.components[cc]*self.S0_basis[cc]
            gf_QQ.Set(QQ_CF)
            # convert to QQ components
            if self.dim==2:
                gf_QQ_comp['00'].Set(gf_QQ[0,0])
                gf_QQ_comp['01'].Set(gf_QQ[0,1])
            elif self.dim==3:
                ERROR
            else:
                ERROR

            solve_str = "Load Forward Sim at t = " + "{:5f}".format(t_kk)
            self.logger.info(solve_str)
            print("\r" + solve_str, end="")

            # NOTE: we can ignore this, because VTK can do its own mesh deformation...
            # # SWW: need to set the mesh deformation for VTK to see it!!!
            # self.Om_mesh.SetDeformation(self.gf_mesh_deform)

            # write to vtu file
            vtkout.Do(time = kk)

            # self.Om_mesh.UnsetDeformation()
        #
