import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
import torch.optim as optim
import numpy as np
import gurobipy

import diffcp
import time
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import \
    dims_to_solver_dict
import numpy as np
import time
import multiprocessing as mp


### MIDRO layer

def to_numpy(x):
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()


def to_torch(x, dtype, device):
    # convert numpy array to torch tensor
    return torch.from_numpy(x).type(dtype).to(device)

class RegressionLayer(nn.Module):
    def __init__(self):
        super(RegressionLayer, self).__init__()
        self.dim=4
        N=1
        self.fc11 = nn.Linear(N, 60)
        self.fc12 = nn.Linear(60,60)
        self.fc13 = nn.Linear(60, self.dim)

        self.fc21 = nn.Linear(N, 60)
        self.fc22 = nn.Linear(60,60)
        self.fc23 = nn.Linear(60, 1)

        self.fc31 = nn.Linear(N, 60)
        self.fc32 = nn.Linear(60,60)
        self.fc33 = nn.Linear(60, self.dim)

        self.fc41 = nn.Linear(N, 60)
        self.fc42 = nn.Linear(60,60)
        self.fc43 = nn.Linear(60, 1)
        

    def forward(self, x):

        y1 = F.relu(self.fc11(x))
        y1 = F.relu(self.fc12(y1))
        y1 = self.fc13(y1).reshape(-1,self.dim,1)

        y2 = F.relu(self.fc21(x))
        y2 = F.relu(self.fc22(y2))
        y2 = self.fc23(y2).reshape(-1,1,1)

        y3 = F.relu(self.fc31(x))
        y3 = F.relu(self.fc32(y3))
        y3 = self.fc33(y3).reshape(-1,self.dim,1)

        y4 = F.relu(self.fc41(x))
        y4 = F.relu(self.fc42(y4))
        y4 = self.fc43(y4).reshape(-1,1,1)


        return y1,y2,y3,y4
def Surrogate( energy_coef, xd ,proba, obj_opt, loss_func, loss_grad_func ,y,param_order_dro,param_ids_dro,
              param_order_int, param_ids_int ,variables,var_dict,compiler,cone_dims,solver_args):
    # batch represent the first dimension size of xd, real_batch is the first dimension size of params
    # xd of size batch * size_of(integer varibale)
    # proba of size batch * 1
    
    class Surrogateded(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            # print(params)
            
            # params is a tuple, each its item is a batch of parameters
            ctx.sample_num=len(proba)//len(params[0])
            # print(ctx.sample_num)
            
            ctx.proba=proba
            ctx.obj_opt=np.repeat(obj_opt,ctx.sample_num,axis=0)
           
            ctx.y=[np.repeat(to_numpy(p),ctx.sample_num,axis=0) for p in y]

            ctx.dtype = params[0].dtype
            ctx.device = params[0].device
            ctx.real_batch=len(params[0])
            
            params_numpy = [np.repeat(to_numpy(p),ctx.sample_num,axis=0) for p in params]
            
            params_numpy=params_numpy + xd # 改成+list
            # print(params_numpy,'asdad',ctx.proba)
            
            
            ctx.batch_sizes = []
            param_order=param_order_dro+param_order_int
            param_ids=param_ids_dro+param_ids_int
            for i, (p, q) in enumerate(zip(params_numpy, param_order)):
                                
                batch_size = len(proba)

                ctx.batch_sizes.append(batch_size)        
            
            ctx.batch_size = len(proba)
            ctx.batch_sizes = np.array(ctx.batch_sizes)
            
            ctx.batch = np.any(ctx.batch_sizes > 0)
            # start = time.time()
            As, bs, cs, cone_dicts, ctx.shapes = [], [], [], [], []
            for i in range(ctx.batch_size):
                params_numpy_i = [
                    p if sz == 0 else p[i]
                    for p, sz in zip(params_numpy, ctx.batch_sizes)]
                # print(params_numpy_i)
                # print(params_numpy_i)
                c, _, neg_A, b = compiler.apply_parameters(
                    dict(zip(param_ids, params_numpy_i)),
                    keep_zeros=True)
                A = -neg_A  # cvxpy canonicalizes -A
                As.append(A)
                bs.append(b)
                cs.append(c)
                cone_dicts.append(cone_dims)
                ctx.shapes.append(A.shape)
            # info['canon_time'] = time.time() - start

            try:
                xs, _, _, _, ctx.DT_batch = diffcp.solve_and_derivative_batch(
                    As, bs, cs, cone_dicts, **solver_args)
            except diffcp.SolverError as e:
                print(
                    "Please consider re-formulating your problem so that "
                    "it is always solvable or increasing the number of "
                    "solver iterations.")
                raise e
            
            sol = [[] for _ in range(len(variables))]
            for i in range(ctx.batch_size):
                sltn_dict = compiler.split_solution(
                    xs[i], active_vars=var_dict)
                for j, v in enumerate(variables):
                    sol[j].append(sltn_dict[v.id])
            sol = [np.stack(s, axis=0) for s in sol]
            ctx.sol=sol
            # print(sol)

            return torch.ones((ctx.real_batch,1,1))
        
        @staticmethod
        def backward(ctx, *grad_input):
            # print(grad_input)
            grad_input_numpy=np.repeat(to_numpy(grad_input[0]),repeats=ctx.sample_num,axis=0)
            E_xds=np.exp(-(ctx.sol[-1]-ctx.obj_opt)/energy_coef)        # size batch*1
            # print(E_xds)
            loss_xds=loss_func(xd,ctx.sol[0:-1],ctx.y)       # size batch*1, input xd and xc are list
            loss_grad=loss_grad_func(xd,ctx.sol[0:-1],ctx.y)        # loss gradient to continuous variables, output same size as batch continous variables
            # print(loss_grad)

            z_batch=(E_xds / ctx.proba).reshape(ctx.real_batch,-1).sum(axis=1,keepdims=True) # size real_batch*1
            z_batch=np.repeat(z_batch,ctx.sample_num,axis=0) # size batch*1
            z_batch=z_batch.reshape(-1,1,1)

            g22_batch= (E_xds * loss_xds) / (ctx.proba * z_batch)
            g1_coe= - g22_batch / energy_coef

            g22_ave=g22_batch.reshape(ctx.real_batch,-1).sum(axis=1,keepdims=True)
            g22_ave=np.repeat(g22_ave,ctx.sample_num,axis=0)
            g22_ave=g22_ave.reshape(-1,1,1)
          
            g2_coe=- g22_ave * E_xds/ (z_batch * ctx.proba * energy_coef)

            g_obj_coe= (g1_coe - g2_coe) * grad_input_numpy

            # g_obj_coe=np.zeros(g_obj_coe.shape)

            g_con_coe= [grad_input_numpy * E_xds * p  / (ctx.proba * z_batch) for p in loss_grad]

            dvars_numpy = g_con_coe+[g_obj_coe]
            # print(dvars_numpy)
            
            dxs, dys, dss = [], [], []
            for i in range(ctx.batch_size):
                
                del_vars = {}
                for v, dv in zip(variables, [dv[i] for dv in dvars_numpy]):
                    del_vars[v.id] = dv
                # print(del_vars)
                dxs.append(compiler.split_adjoint(del_vars))
                dys.append(np.zeros(ctx.shapes[i][0]))
                dss.append(np.zeros(ctx.shapes[i][0]))

            dAs, dbs, dcs = ctx.DT_batch(dxs, dys, dss)

            # differentiate from cone problem data to cvxpy parameters
            # start = time.time()
            grad = [[] for _ in range(len(param_ids_dro))]
            for i in range(ctx.batch_size):
                del_param_dict = compiler.apply_param_jac(
                    dcs[i], -dAs[i], dbs[i])
                if i % ctx.sample_num ==0:
                    for j, pid in enumerate(param_ids_dro):
                        grad[j] += [to_torch(del_param_dict[pid],
                                            ctx.dtype, ctx.device).unsqueeze(0)]
                else:
                    for j, pid in enumerate(param_ids_dro):
                        grad[j][i // ctx.sample_num] =grad[j][i // ctx.sample_num] + to_torch(del_param_dict[pid],
                                            ctx.dtype, ctx.device).unsqueeze(0)

            grad = [torch.cat(g, 0) for g in grad]
            # print('HI',tuple(grad))

            return tuple(grad)
        
    return Surrogateded.apply


class MIDROLayer(nn.Module):

    def __init__(self,  problem, parameters_dro,parameters_int, variables,energy_coef,
                    loss_func,loss_grad_func ):
        # parameters_dro, parameters_int, and variables lists
        # parameters_dro is parameter of the ambiguity set, and parameters_int is the integer variables
        # the last item in variable is always the objective value of the DRO problem 
        # loss_func takes input (x_d,x_c,y) in batch, and returns loss in size batch_size * 1
        # loss_grad_func takes input (x_d,x_c,y) in batch, and returns the gradient with respect to x_c, is of size batch * shape_of_x_c
        super(MIDROLayer, self).__init__()
        self.xdp=torch.tensor([[0.0],[0.0],[0.0],[1.0]])
        self.loss=loss_func
        self.loss_grad=loss_grad_func
        self.y=torch.tensor([[0.0]])
        self.proba=torch.tensor([[1.0]])
        
        self.param_order_dro = parameters_dro
        self.param_order_int = parameters_int
        self.variables = variables
        self.var_dict = {v.id for v in self.variables}
        data, _, _ = problem.get_problem_data(
        solver=cp.SCS, solver_opts={'use_quad_obj': False})
        self.compiler = data[cp.settings.PARAM_PROB]
        self.param_ids_dro = [p.id for p in self.param_order_dro]
        self.param_ids_int = [p.id for p in self.param_order_int]
        self.cone_dims = dims_to_solver_dict(data["dims"])
        self.solver_args={}
        self.energy=energy_coef

    def data_updata(self,xdp,prob,obj_opt,y):
        self.xdp=xdp
        self.proba=prob
        self.y=y
        self.obj_opt=obj_opt

    def forward(self, *params):
        # *params is a list where each of its items corresponds to the parameters, and the first dimension is the batch
        result=Surrogate(self.energy,self.xdp,self.proba ,self.obj_opt, self.loss, self.loss_grad , self.y ,             
            param_order_dro=self.param_order_dro,
            param_ids_dro=self.param_ids_dro,
            param_order_int=self.param_order_int,
            param_ids_int=self.param_ids_int,
            variables=self.variables,
            var_dict=self.var_dict,
            compiler=self.compiler,
            cone_dims=self.cone_dims,
            solver_args=self.solver_args)
        
        return result(*params)

### Projection layer
    
def projectionfn( knns ,param_order,
            param_id,
            param_order_knn,
            param_id_knn,
            variables,
            var_dict,
            compiler,
            cone_dims,
            solver_args):
    # batch represent the first dimension size of xd, real_batch is the first dimension size of params
    # xd of size batch * size_of(integer varibale)
    # proba of size batch * 1
    
    class projectionfnfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):

            # params is a tuple, each its item is a batch of parameters

            ctx.dtype = params[0].dtype
            ctx.device = params[0].device
            ctx.real_batch=len(params[0])
            params_numpy = [to_numpy(p) for p in params]
            params_numpy=params_numpy+knns
            # print(params_numpy)
            ctx.batch_sizes = []
            param_orders=param_order+param_order_knn
            param_ids=param_id+param_id_knn
            for i, (p, q) in enumerate(zip(params_numpy, param_orders)):
                
                batch_size = len(params[0])

                ctx.batch_sizes.append(batch_size)

            
            ctx.batch_size = len(params[0])
            ctx.batch_sizes = np.array(ctx.batch_sizes)
            # print(ctx.batch_sizes)
            ctx.batch = np.any(ctx.batch_sizes > 0)
            # start = time.time()
            As, bs, cs, cone_dicts, ctx.shapes = [], [], [], [], []
            for i in range(ctx.batch_size):
                params_numpy_i = [
                    p if sz == 0 else p[i]
                    for p, sz in zip(params_numpy, ctx.batch_sizes)]
                # print(params_numpy_i)
                c, _, neg_A, b = compiler.apply_parameters(
                    dict(zip(param_ids, params_numpy_i)),
                    keep_zeros=True)
                A = -neg_A  # cvxpy canonicalizes -A
                As.append(A)
                bs.append(b)
                cs.append(c)
                cone_dicts.append(cone_dims)
                ctx.shapes.append(A.shape)

            try:
                xs, _, _, _, ctx.DT_batch = diffcp.solve_and_derivative_batch(
                    As, bs, cs, cone_dicts, **solver_args)
            except diffcp.SolverError as e:
                print(
                    "Please consider re-formulating your problem so that "
                    "it is always solvable or increasing the number of "
                    "solver iterations.")
                raise e
            
            sol = [[] for _ in range(len(variables))]
            for i in range(ctx.batch_size):
                sltn_dict = compiler.split_solution(
                    xs[i], active_vars=var_dict)
                for j, v in enumerate(variables):
                    sol[j].append(to_torch(
                        sltn_dict[v.id], ctx.dtype, ctx.device).unsqueeze(0))
            sol = [torch.cat(s, 0) for s in sol]


            return tuple(sol)

        @staticmethod
        def backward(ctx, *dvars):

            # print(dvars)
            dvars_numpy = [to_numpy(dvar) for dvar in dvars]
            
            dxs, dys, dss = [], [], []
            for i in range(ctx.batch_size):
                del_vars = {}
                for v, dv in zip(variables, [dv[i] for dv in dvars_numpy]):
                    del_vars[v.id] = dv
                dxs.append(compiler.split_adjoint(del_vars))
                dys.append(np.zeros(ctx.shapes[i][0]))
                dss.append(np.zeros(ctx.shapes[i][0]))

            dAs, dbs, dcs = ctx.DT_batch(dxs, dys, dss)

            grad = [[] for _ in range(len(param_id))]
            for i in range(ctx.batch_size):
                # print(i)
                del_param_dict = compiler.apply_param_jac(
                    dcs[i], -dAs[i], dbs[i])
                for j, pid in enumerate(param_id):
                    grad[j] += [to_torch(del_param_dict[pid],
                                         ctx.dtype, ctx.device).unsqueeze(0)]
            grad = [torch.cat(g, 0) for g in grad]
            # print('a')

            return tuple(grad)
        
    return projectionfnfn.apply

class ProjectionLayer(nn.Module):

    def __init__(self, problem , params , knn_params , proj_params):
        super(ProjectionLayer, self).__init__()
        self.knns=np.ones((20,1))
        self.param_order = params
        self.param_order_knn = knn_params
        self.variables = proj_params
        self.var_dict = {v.id for v in self.variables}
        data, _, _ = problem.get_problem_data(
        solver=cp.SCS, solver_opts={'use_quad_obj': False})
        self.compiler = data[cp.settings.PARAM_PROB]
        self.param_ids = [p.id for p in self.param_order]
        self.param_ids_knn = [p.id for p in self.param_order_knn]
        self.cone_dims = dims_to_solver_dict(data["dims"])
        self.solver_args={'solve_method': 'ECOS'}

    def updata_knn(self,knns):
        self.knns=knns

    def forward(self, *input):

        res=projectionfn( self.knns,param_order=self.param_order,
            param_id=self.param_ids,
            param_order_knn=self.param_order_knn,
            param_id_knn=self.param_ids_knn,
            variables=self.variables,
            var_dict=self.var_dict,
            compiler=self.compiler,
            cone_dims=self.cone_dims,
            solver_args=self.solver_args )
        

        return res(*input)


