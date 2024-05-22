import cvxpy as cp
import torch
import torch.optim as optim
import numpy as np

import time
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import \
    dims_to_solver_dict
import numpy as np
import time
from multiprocessing import Pool
import torch.optim as optim
import numpy as np
import torch.utils.data as data
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors


from my_layer_news import *

def to_numpy(x):
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()


def to_torch(x, dtype, device):
    # convert numpy array to torch tensor
    return torch.from_numpy(x).type(dtype).to(device)

class Newsvendor():

    def __init__(self,amb_type):

        self.amb_type=amb_type

        self.dim=4
        self.num_piece=16
        self.a=0.25 * np.array([[[1.0]],[[2.0]],[[3.0]],[[4.0]]]) # buy price
        self.a_d=0.25 * np.array([[[0.95]],np.array(2)*[[0.95]],
                            np.array(3)*[[0.95]],np.array(4)*[[0.95]]])
        self.c=np.array([[[2.0]],[[4.0]],[[6.0]],[[8.0]]]) # recur price
        self.d=np.array([[[0.5]],[[1.0]],[[1.5]],[[2.0]]]) # holding cost
        self.v_d=np.array([[[10.0]],[[13.0]],[[16.0]],[[19.0]]])

        self.a_to=torch.tensor([[1.0]]) # buy price
        self.a_d_to=torch.tensor([[0.96],[0.92],[0.88]])
        self.c_to=torch.tensor([[2.0]]) # recur price
        self.d_to=torch.tensor([[0.5]]) # holding cost
        self.v_d_to=torch.tensor([[4.0],[7.0],[10.0]])
        integer_all=[]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                        for l in range(2):
                            integer_all.append(np.array([[l,k,j,i]]))
        self.integer_all=np.stack(integer_all)
        self.integer_total=len(self.integer_all)

    def get_dro_problem(self):

        xs=[cp.Variable((1,1)) for _ in range(self.dim)]
        xd=[cp.Parameter((1,1)) for _ in range(self.dim)]

        coeff=[]
        x_cost=[]
        cds=[[self.c[i],-self.d[i]] for i in range(self.dim)]

               
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        coeff.append(np.vstack([cds[0][i],cds[1][j],cds[2][k],cds[3][l]])) # uncertainty coefficent of each piece
                        x_cost.append(-cds[0][i] @ (xs[0] + self.v_d[0].T @ xd[0]) -cds[1][j] @ (xs[1] + self.v_d[1].T @ xd[1])
                                    -cds[2][k] @ (xs[2] + self.v_d[2].T @ xd[2])-cds[3][l] @ (xs[3] + self.v_d[3].T @ xd[3]))
        
        if self.amb_type=='SOC-I':
            h=[cp.Parameter((self.dim,1)) for _ in range(1)]
            sigams=[cp.Parameter((1,1)) for _ in range(1)]

            rs=cp.Variable((1,1))

            beta=cp.Variable((1,1))

            Axi=[-np.identity(self.dim),np.identity(self.dim),-np.identity(self.dim)] # the first self.num_SOC items are coeffient 
            # matrix with SOC ambiguity set constraints, and the last two represent the support
            Au=[-np.ones((1,1))]
            Av=[-np.identity(self.dim),-np.identity(self.dim),np.ones((1,self.dim))]

            b=[-h[-1],h[-1],np.zeros((1,1))]

            eta=[[cp.Variable((self.dim,1)),cp.Variable((self.dim,1)),cp.Variable((self.dim,1))
                 ,cp.Variable((1,1))] for _ in range(self.num_piece)] # dual vaiable eta
            
            constraints=[]
            constraints=constraints+[xs[i]>=0 for i in range(self.dim)] # put constraints for x here
            constraints.append(beta>=0)
            for i in range(self.num_piece):
                constraints=constraints+[eta[i][j]>=0 for j in range(len(eta[i]))]
                constraints=constraints+[coeff[i]== cp.sum([Axi[j].T @ eta[i][j] for j in range(len(Axi))])]
                
                constraints=constraints+[-beta == Au[-1].T @ eta[i][-1]]
                constraints=constraints+[np.zeros((self.dim,1)) == cp.sum([Av[j].T @ eta[i][j+1] for j in range(len(Av))]) ]
            
                constraints.append(rs>=x_cost[i]-cp.sum([b[j].T @ eta[i][j+1] for j in range(len(b))]))

            obj=cp.Variable((1,1))
            constraints.append(obj>=rs +  beta.T @ sigams[0] + cp.sum([self.a[i] @ xs[i]+(self.a_d[i] * self.v_d[i]).T @ xd[i] 
                                                                for i in range(self.dim)]))

            objective = cp.Minimize(obj)
            
            problems = cp.Problem(objective, constraints)

            assert problems.is_dpp()
            return problems,[h[0],sigams[0]],xd,xs+[obj] # return the problem, parameters, and the variables
        
        if self.amb_type=='SOC-II':
            h=[cp.Parameter((self.dim,1)) for _ in range(1)]
            sigams=[cp.Parameter((1,1)) for _ in range(1)]

            rs=cp.Variable((1,1))

            beta=cp.Variable((1,1))

            Axi=[-np.identity(self.dim),np.concatenate([np.identity(self.dim),np.zeros((2,self.dim))])] # the first self.num_SOC items are coeffient 
            # matrix with SOC ambiguity set constraints, and the last two represent the support
            Au=[np.concatenate([np.zeros((self.dim,1)),np.array([[0.5],[-0.5]])])]
            b=[cp.vstack([ -h[-1] , np.array([[-0.5],[-0.5]]) ])]

            eta=[[cp.Variable((self.dim,1)),cp.Variable((self.dim+2,1))]
                 for _ in range(self.num_piece)] # dual vaiable eta
            
            constraints=[]
            constraints=constraints+[xs[i]>=0 for i in range(self.dim)] # put constraints for x here
            constraints.append(beta>=0)
            for i in range(self.num_piece):
                constraints=constraints+[eta[i][0]>=0,cp.SOC(eta[i][-1][-1],eta[i][-1][0:-1],axis=0)]
                constraints=constraints+[coeff[i]== cp.sum([Axi[j].T @ eta[i][j] for j in range(len(Axi))])]
                
                constraints=constraints+[-beta== Au[-1].T @ eta[i][-1] ]
                constraints.append(rs>=x_cost[i]-b[-1].T @ eta[i][-1])

            obj=cp.Variable((1,1))
            constraints.append(obj>=rs +  beta.T @ sigams[0] + cp.sum([self.a[i] @ xs[i]+(self.a_d[i] * self.v_d[i]).T @ xd[i] 
                                                                for i in range(self.dim)]))

            objective = cp.Minimize(obj)
            
            problems = cp.Problem(objective, constraints)

            assert problems.is_dpp()
            return problems,[h[0],sigams[0]],xd,xs+[obj] # return the problem, parameters, and the variables
        
        if self.amb_type=='SOC-III':
            h=[cp.Parameter((self.dim,1)),cp.Parameter((self.dim,1))]
            sigams=[cp.Parameter((1,1)) for _ in range(2)]

            rs=cp.Variable((1,1))

            beta=[cp.Variable((1,1)) for _ in range(2)]

            Axi=[-np.identity(self.dim),np.identity(self.dim),-np.identity(self.dim),np.zeros((1,1)),
                 np.concatenate([np.identity(self.dim),np.zeros((2,self.dim))]) ] # the first self.num_SOC items are coeffient 
            # matrix with SOC ambiguity set constraints, and the last two represent the support
            Au=[-np.ones((1,1)),np.concatenate([np.zeros((self.dim,1)),np.array([[0.5],[-0.5]])])]
            Av=[-np.identity(self.dim),-np.identity(self.dim),np.ones((1,self.dim))]

            b=[-h[0],h[0],np.zeros((1,1)),cp.vstack([ -h[1] , np.array([[-0.5],[-0.5]]) ])]

            eta=[[cp.Variable((self.dim,1)),cp.Variable((self.dim,1)),cp.Variable((self.dim,1))
                 ,cp.Variable((1,1)),cp.Variable((self.dim+2,1))] for _ in range(self.num_piece)] # dual vaiable eta
            
            constraints=[]
            constraints=constraints+[xs[i]>=0 for i in range(self.dim)] # put constraints for x here
            constraints=constraints+[beta[i]>=0 for i in range(len(beta))]
            for i in range(self.num_piece):
                constraints=constraints+[eta[i][j]>=0 for j in range(4)]+[cp.SOC(eta[i][-1][-1],eta[i][-1][0:-1],axis=0)]
                constraints=constraints+[coeff[i]== cp.sum([Axi[j].T @ eta[i][j] for j in range(len(Axi))])]
                
                constraints=constraints+[-beta[0]== Au[0].T @ eta[i][-2], -beta[1]==Au[1].T @ eta[i][-1]]
                
                constraints=constraints+[np.zeros((self.dim,1)) == cp.sum([Av[j].T @ eta[i][j+1] for j in range(len(Av))]) ]
            
                constraints.append(rs>=x_cost[i]-cp.sum([b[j].T @ eta[i][j+1] for j in range(len(b))]))

            obj=cp.Variable((1,1))
            constraints.append(obj>=rs +  beta[0].T @ sigams[0]+beta[1].T @ sigams[1] +
                               cp.sum([self.a[i] @ xs[i]+(self.a_d[i] * self.v_d[i]).T @ xd[i] for i in range(self.dim)]))

            objective = cp.Minimize(obj)
            
            problems = cp.Problem(objective, constraints)

            assert problems.is_dpp()
            return problems,[h[0],sigams[0],h[1],sigams[1]],xd,xs+[obj] # return the problem, parameters, and the variables

    def get_proj_problem(self,k):

        h=[cp.Parameter((self.dim,1)),cp.Parameter((self.dim,1))]
        sigams=[cp.Parameter((1,1)) for _ in range(2)]
        knn_v=[cp.Parameter((k,1)) for _ in range(self.dim)]
        knn_stack=cp.hstack(knn_v)

        h_proj=[cp.Variable((self.dim,1)),cp.Variable((self.dim,1))]
        sigams_proj=[cp.Variable((1,1)) for _ in range(2)]

        obj=cp.Variable((1,1))

        constraints=[ cp.mean(cp.norm( knn_stack.T - h_proj[0],p=1,axis=0  ) ) <=  sigams_proj[0]]
        constraints=constraints+[ cp.sum(cp.square( knn_stack.T - h_proj[1]  ) ) / k <=  sigams_proj[1]]
        constraints.append(obj >= cp.sum([5 * cp.pnorm(h[i]-h_proj[i],2)**2+cp.pnorm(sigams[i]-sigams_proj[i],2)**2
                                              for i in range(2)]) )
        
        objective = cp.Minimize(obj)

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        params=[h[0],sigams[0],h[1],sigams[1]]
        proj_params=[h_proj[0],sigams_proj[0],h_proj[1],sigams_proj[1],obj]

        return problem,params,knn_v,proj_params
        
        
    def solve_model(self,input_data,num_sols):

        xs=[cp.Variable((1,1)) for _ in range(self.dim)]
        xd=[cp.Variable((1,1),boolean=True) for _ in range(self.dim)]

        coeff=[]
        x_cost=[]
        cds=[[self.c[i],-self.d[i]] for i in range(self.dim)]
        

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        coeff.append(np.vstack([cds[0][i],cds[1][j],cds[2][k],cds[3][l]])) # uncertainty coefficent of each piece
                        x_cost.append(-cds[0][i] @ (xs[0] + self.v_d[0].T @ xd[0]) -cds[1][j] @ (xs[1] + self.v_d[1].T @ xd[1])
                                    -cds[2][k] @ (xs[2] + self.v_d[2].T @ xd[2])-cds[3][l] @ (xs[3] + self.v_d[3].T @ xd[3]))

        if self.amb_type=='SOC-I':
            h,sigams=input_data
            h=[h]
            sigams=[sigams]
            # f=[cp.Parameter((self.dim,1)) for _ in range(1)]

            rs=cp.Variable((1,1))

            beta=cp.Variable((1,1))

            Axi=[-np.identity(self.dim),np.identity(self.dim),-np.identity(self.dim)] # the first self.num_SOC items are coeffient 
            # matrix with SOC ambiguity set constraints, and the last two represent the support
            Au=[-np.ones((1,1))]
            Av=[-np.identity(self.dim),-np.identity(self.dim),np.ones((1,self.dim))]

            b=[-h[-1],h[-1],np.zeros((1,1))]

            eta=[[cp.Variable((self.dim,1)),cp.Variable((self.dim,1)),cp.Variable((self.dim,1))
                 ,cp.Variable((1,1))] for _ in range(self.num_piece)] # dual vaiable eta
            
            constraints=[]
            constraints=constraints+[xs[i]>=0 for i in range(self.dim)]
            constraints.append(beta>=0)
            for i in range(self.num_piece):
                constraints=constraints+[eta[i][j]>=0 for j in range(len(eta[i]))]
                constraints=constraints+[coeff[i]== cp.sum([Axi[j].T @ eta[i][j] for j in range(len(Axi))])]
                
                constraints=constraints+[-beta== Au[-1].T @ eta[i][-1]]
                constraints=constraints+[np.zeros((self.dim,1)) == cp.sum([Av[j].T @ eta[i][j+1] for j in range(len(Av))]) ]
            
                constraints.append(rs>=x_cost[i]-cp.sum([b[j].T @ eta[i][j+1] for j in range(len(b))]))

            obj=cp.Variable((1,1))
            constraints.append(obj>=rs +  beta.T @ sigams[0] + cp.sum([self.a[i] @ xs[i]+(self.a_d[i] * self.v_d[i]).T @ xd[i] 
                                                                for i in range(self.dim)]))
            
        if self.amb_type=='SOC-II':
            h,sigams=input_data
            h=[h]
            sigams=[sigams]

            rs=cp.Variable((1,1))

            beta=cp.Variable((1,1))

            Axi=[-np.identity(self.dim),np.concatenate([np.identity(self.dim),np.zeros((2,self.dim))])] # the first self.num_SOC items are coeffient 
            # matrix with SOC ambiguity set constraints, and the last two represent the support
            Au=[np.concatenate([np.zeros((self.dim,1)),np.array([[0.5],[-0.5]])])]
            b=[cp.vstack([ -h[-1] , np.array([[-0.5],[-0.5]]) ])]

            eta=[[cp.Variable((self.dim,1)),cp.Variable((self.dim+2,1))]
                 for _ in range(self.num_piece)] # dual vaiable eta
            
            constraints=[]
            constraints=constraints+[xs[i]>=0 for i in range(self.dim)]
            constraints.append(beta>=0)
            for i in range(self.num_piece):
                constraints=constraints+[eta[i][0]>=0,cp.SOC(eta[i][-1][-1],eta[i][-1][0:-1],axis=0)]
                constraints=constraints+[coeff[i]== cp.sum([Axi[j].T @ eta[i][j] for j in range(len(Axi))])]
                
                constraints=constraints+[-beta== Au[-1].T @ eta[i][-1] ]
                constraints.append(rs>=x_cost[i]-b[-1].T @ eta[i][-1])

            obj=cp.Variable((1,1))
            constraints.append(obj>=rs +  beta.T @ sigams[0] + cp.sum([self.a[i] @ xs[i]+(self.a_d[i] * self.v_d[i]).T @ xd[i] 
                                                                for i in range(self.dim)]))
            
        if self.amb_type=='SOC-III':
            h0,sigams0,h1,sigams1=input_data
            h=[h0,h1]
            sigams=[sigams0,sigams1]

            rs=cp.Variable((1,1))

            beta=[cp.Variable((1,1)) for _ in range(3)]

            Axi=[-np.identity(self.dim),np.identity(self.dim),-np.identity(self.dim),np.zeros((1,1)),
                 np.concatenate([np.identity(self.dim),np.zeros((2,self.dim))]) ] # the first self.num_SOC items are coeffient 
            # matrix with SOC ambiguity set constraints, and the last two represent the support
            Au=[-np.ones((1,1)),np.concatenate([np.zeros((self.dim,1)),np.array([[0.5],[-0.5]])])]
            Av=[-np.identity(self.dim),-np.identity(self.dim),np.ones((1,self.dim))]

            b=[-h[0],h[0],np.zeros((1,1)),cp.vstack([ -h[1] , np.array([[-0.5],[-0.5]]) ])]

            eta=[[cp.Variable((self.dim,1)),cp.Variable((self.dim,1)),cp.Variable((self.dim,1))
                 ,cp.Variable((1,1)),cp.Variable((self.dim+2,1))] for _ in range(self.num_piece)] # dual vaiable eta
            
            constraints=[]
            constraints=constraints+[xs[i]>=0 for i in range(self.dim)] 
            constraints=constraints+[beta[i]>=0 for i in range(len(beta))]
            for i in range(self.num_piece):
                constraints=constraints+[eta[i][j]>=0 for j in range(4)]+[cp.SOC(eta[i][-1][-1],eta[i][-1][0:-1],axis=0)]
                constraints=constraints+[coeff[i]== cp.sum([Axi[j].T @ eta[i][j] for j in range(len(Axi))])]
                
                constraints=constraints+[-beta[0]== Au[0].T @ eta[i][-2], -beta[1]==Au[1].T @ eta[i][-1]]
                
                constraints=constraints+[np.zeros((self.dim,1)) == cp.sum([Av[j].T @ eta[i][j+1] for j in range(len(Av))]) ]
            
                constraints.append(rs>=x_cost[i]-cp.sum([b[j].T @ eta[i][j+1] for j in range(len(b))]))


            obj=cp.Variable((1,1))
            constraints.append(obj>=rs +  beta[0].T @ sigams[0]+beta[1].T @ sigams[1] + cp.sum([self.a[i] @ xs[i]+(self.a_d[i] * self.v_d[i]).T @ xd[i] 
                                                                for i in range(self.dim)]))

        objective = cp.Minimize(obj)
        
        problems = cp.Problem(objective, constraints)

        problems.solve(solver=cp.GUROBI,PreMIQCPForm=1)

        xd_opt=[[] for _ in range(self.dim)]
        xc_opt=[[] for _ in range(self.dim)]
        obj_opt=[]
        for i in range(self.dim):
            xd_opt[i].append(np.around(xd[i].value))
            xc_opt[i].append(xs[i].value)
        obj_opt.append(obj.value)

        for t in range(num_sols-1):
            constraints=constraints+[cp.sum([xd[i].T @ (1- 2 * xd_opt[i][-1] ) +  xd_opt[i][-1] 
                                            for i in range(self.dim)]) >=1]
            problems = cp.Problem(objective, constraints)
            problems.solve(solver=cp.GUROBI,OptimalityTol=1e-4,PreMIQCPForm=1)
            for i in range(self.dim):
                xd_opt[i].append(np.around(xd[i].value))
                xc_opt[i].append(xs[i].value)
            obj_opt.append(obj.value)
        output=[np.stack(xd_opt[i]) for i in range(self.dim)]+[np.stack(xc_opt[i]) for i in range(self.dim)]+[np.stack(obj_opt)]

        return tuple(output)
        
        
    def verify_model(self,input_data):
        return self.solve_model(input_data,1)
        
    def sampling(self,t_sol,energy_coef,M):
    # the probability of getting a point outside T is set to the half of the largest one
    
        decay=np.array([[0.5]])

        xd1,xd2,xd3,xd4,xc1,xc2,xc3,xc4,obj_or=t_sol
        xd1=np.around(xd1)
        xd2=np.around(xd2)
        xd3=np.around(xd3)
        xd4=np.around(xd4)
        obj=obj_or-obj_or[0]
        obj=np.exp(-obj / np.array(energy_coef))

        obj_least=decay * obj[-1]
        # print(obj)
        map=[(np.array([[1]]) @ xd1[i] + np.array([[2]]) @ xd2[i] +
                np.array([[4]]) @ xd3[i] + np.array([[8]]) @ xd4[i]).astype(int).reshape(-1) for i in range(len(obj))]

        frac=0.4
        if (self.integer_total-len(obj_or)) * obj_least >= frac * np.sum(obj,axis=0,keepdims=True):
            obj_least = frac * np.sum(obj,axis=0,keepdims=True) / (self.integer_total-len(obj_or))
        
        p=np.concatenate([obj_least/(np.sum(obj)+(self.integer_total-len(obj)) * obj_least) 
                            for _ in range(self.integer_total)]).reshape(-1)
        p[map]=np.concatenate([obj[i] / (np.sum(obj)+(self.integer_total-len(obj)) * obj_least) 
                                for i in range(len(obj))]).reshape(len(map),-1)
        
        p=p/np.sum(p)
        index = np.random.choice([i for i in range(self.integer_total)], M , p=p.reshape(-1))
        
        sample_output=self.integer_all[index]
        sample_output1=sample_output[:,:,0].reshape(-1,1,1)
        sample_output2=sample_output[:,:,1].reshape(-1,1,1)
        sample_output3=sample_output[:,:,2].reshape(-1,1,1)
        sample_output4=sample_output[:,:,3].reshape(-1,1,1)

        proba=p[index].reshape(-1,1,1)
        return sample_output1,sample_output2,sample_output3,sample_output4,proba,obj_or[0]
         
    def loss_func(self,xds,xcs,ys):
    
        output=[]
        batch_size=len(ys[0])
        for i in range(batch_size):
            outs=np.array([[0.0]])
            for j in range(self.dim):
                xd=xds[j][i]
                xc=xcs[j][i]
                y=ys[j][i]
                outs=outs+(self.a_d[j] * self.v_d[j]).T @ xd+self.a[j] @ xc +np.max([self.c[j] @ (y-self.v_d[j].T @ xd - xc),
                                                                            self.d[j] @ (-y+self.v_d[j].T @ xd +xc)])
            output.append(outs)
            
        return np.stack(output,axis=0)
    
    def loss_grad_func(self,xds,xcs,ys):
        
        output=[[] for _ in range(self.dim)]
        batch_size=len(ys[0])
        # print(batch_size)
        for i in range(batch_size):
            for j in range(self.dim):
                xd=xds[j][i]
                xc=xcs[j][i]
                y=ys[j][i]
                if self.c[j] @ (y-self.v_d[j].T @ xd - xc) >= self.d[j] @ (-y+self.v_d[j].T @ xd +xc):
                    gradxc= self.a[j] - self.c[j]
                else:
                    gradxc= self.a[j] + self.d[j]
                output[j].append(gradxc)

        output= [np.stack(p) for p in output]

        return output
    
def knnsearch(nbrs,cali_data,feature):
    feature=to_numpy(feature)
    distances, indices = nbrs.kneighbors(feature)
    return [np.stack([cali_data[indices[i],j+1].reshape(-1,1) for i in range(len(feature))]) for j in range(cali_data.shape[1]-1)]

def knn_pre_train(nbrs,cali_data,feature):
    feature=to_numpy(feature)
    distances, indices = nbrs.kneighbors(feature)
    return [cali_data[indix,1:] for indix in indices]

def testing_loss(reg_net,amb_set,proj_net,loss_func,test_loader,dim,nbrs,cali_data,my_proj,pool,myprob):
    with torch.no_grad():
        cost=torch.tensor([[0]])
        for step, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
            x=batch_x.reshape(-1,1).to(torch.float32)
            y=[batch_y[:,i].reshape(-1,1,1).to(torch.float32) for i in range(dim)]
            asd=reg_net(x)
            knns=knnsearch(nbrs,cali_data,x)
            proj_net.updata_knn(knns)
            app=my_proj(*asd)
            if amb_set=='SOC-III':
                params=zip(list(zip(*[to_numpy(app[i]) for i in range(len(app)-1)])),[1 for _ in range(len(x))])
            if amb_set=='SOC-I':
                params=zip(list(zip(*[to_numpy(app[i]) for i in range(2)])),[1 for _ in range(len(x))])
            if amb_set=='SOC-II':
                params=zip(list(zip(*[to_numpy(app[i+2]) for i in range(2)])),[1 for _ in range(len(x))])
        
            res = pool.starmap(myprob.solve_model, params)
            xds_opt=[np.stack([p[i][0] for p in res]) for i in range(dim) ]
            xcs_opt=[np.stack([p[i+dim][0] for p in res]) for i in range(dim)]
            # print(xds_opt[0],xcs_opt[0].shape)
            cost=cost+loss_func(xds_opt,xcs_opt,[to_numpy(p) for p in y]).mean()
    return cost/(step+1)

def training(epoches,training_data_size,num_pool):
    pool=Pool(num_pool)
    record_df=[[] for _ in range(3)]
    record_pre=[[] for _ in range(3)]
    amb_all=['SOC-II','SOC-I','SOC-III']

    k=int(training_data_size/20)
    dim=4
    num_train=training_data_size
    num_valid=num_train
    num_test=2000
    num_total=num_train+num_valid+num_test

    v=np.random.uniform(0,1,[num_total,1])
    q=[[],[],[],[]]
    q_n=[]
    for i in v:
        mean=np.array([8+6*np.square(i-0.8),11+6*np.square(i-0.2)]).reshape(-1)
        cov=np.array([[1/(0.5+1*np.abs(i-0.8)),np.array([1.0])],[np.array([1.0]),1/(0.5+1*np.abs(i-0.2))]]).reshape(2,2)
        cov=cov.T @ cov
        q_n.append(np.random.multivariate_normal(mean,cov,1))
        q[0].append(np.random.normal(8+6*np.square(i-0.2),1/(1+8*np.abs(i-0.2)),1))
        q[1].append(np.random.normal(11+6*np.square(i-0.4),1/(1+8*np.abs(i-0.4)),1))
        q[2].append(np.random.normal(14+6*np.square(i-0.6),1/(1+8*np.abs(i-0.6)),1))
        q[3].append(np.random.normal(17+6*np.square(i-0.8),1/(1+8*np.abs(i-0.8)),1))
    q=[np.concatenate(p,axis=0).reshape(-1,1) for p in q]
    q=np.hstack(q)
    q_n=q
    cali_data=np.hstack((v[0:num_train],q_n[0:num_train]))

    train_data=[(v[i],q_n[i]) for i in range(num_train)]
    valid_data=[(v[num_train+i],q_n[num_train+i]) for i in range(num_valid)]
    test_data=[(v[num_train+num_valid+i],q_n[num_train+num_valid+i]) for i in range(num_test)]
    loader = data.DataLoader(
        dataset=train_data,
        batch_size=50,
        shuffle=True,
        num_workers=2
    )

    valid_loader = data.DataLoader(
        dataset=valid_data,
        batch_size=200,
        shuffle=True,
        num_workers=2
    )

    test_loader = data.DataLoader(
        dataset=test_data,
        batch_size=200,
        shuffle=True,
        num_workers=2
    )
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(cali_data[:,0].reshape(-1,1))
    net_reg=RegressionLayer()

    lr=0.1
    optimizer = optim.Adam(net_reg.parameters(),lr=lr)

    for epoch in range(40):
        cost_total=torch.tensor([0.0])
        for step, (batch_x, batch_y) in tqdm(enumerate(loader)):
            
            x=to_torch(np.random.uniform(0,1,[200,1]),torch.float32,'cpu')
            asd=net_reg(x)
            knns=knn_pre_train(nbrs,cali_data,x)
            knns_cpu=[to_torch(knn,torch.float32,'cpu') for knn in knns]
            cost=torch.tensor([0.0])
            for i in range(k):
                knn_sig=torch.stack([knn[i].reshape(-1,1) for knn in knns_cpu],axis=0)
                cost=cost+torch.abs(asd[0]-knn_sig).sum()+torch.square(asd[2]-knn_sig).sum()

            cost=cost/200
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                cost_total=cost_total+cost
            print('pre-training 1 epoch=',epoch,'step=',step,'training_loss=',cost_total/(step+1))
        print('pre-training 1 epoch=',epoch+1,'training_loss=',cost_total/4)

        if (epoch+1) % 20 ==0:
            lr=lr/5
            optimizer = optim.Adam(net_reg.parameters(),lr=lr)

    lr=0.1
    optimizer = optim.Adam(net_reg.parameters(),lr=lr)

    for epoch in range(40):
        cost_total=torch.tensor([0.0])
        cost1=torch.tensor([0.0])
        cost2=torch.tensor([0.0])
        for step, (batch_x, batch_y) in tqdm(enumerate(loader)):
            x=to_torch(np.random.uniform(0,1,[200,1]),torch.float32,'cpu')
            asd=net_reg(x) 
            h1=to_numpy(asd[0])
            h2=to_numpy(asd[2])
            h1=to_torch(h1,torch.float32,'cpu')
            h2=to_torch(h2,torch.float32,'cpu')
            knns=knn_pre_train(nbrs,cali_data,x)
            knns_cpu=[to_torch(knn,torch.float32,'cpu') for knn in knns]
            knns_mean=torch.stack([torch.mean(knn.T,axis=1,keepdims=True) for knn in knns_cpu])
            for i in range(k):
                knn_sig=torch.stack([knn[i].reshape(-1,1) for knn in knns_cpu],axis=0)
                cost1=cost1+torch.abs(h1-knn_sig).sum(axis=1,keepdim=True)
                cost2=cost2+torch.square(h2-knn_sig).sum(axis=1,keepdim=True)
            
            cost1=cost1/k
            cost2=cost2/k
            losss=torch.square(cost1-asd[1]).mean()+torch.square(cost2-asd[3]).mean()
            losss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                cost_total=cost_total+losss
            print('pre-training 2 epoch=',epoch,'step=',step,'training_loss=',cost_total/(step+1))
        print('pre-training 2 epoch=',epoch+1,'training_loss=',cost_total/4)
        if (epoch+1) % 20 ==0:
            lr=lr/5
            optimizer = optim.Adam(net_reg.parameters(),lr=lr)
    torch.save(net_reg,'my_net')

    for amb in range(3):
        net_reg=torch.load('my_net')
        energy_coef=3
        amb_set=amb_all[amb]
        myprob=Newsvendor(amb_set)
        problem_pro,params_pro,knn_params_pro,proj_params_pro=myprob.get_proj_problem(k)
        my_proj=ProjectionLayer(problem_pro,params_pro,knn_params_pro,proj_params_pro)
        pro,param_dro,param_int,vari=myprob.get_dro_problem()
        my_dro=MIDROLayer(pro,param_dro,param_int,vari,energy_coef,myprob.loss_func,myprob.loss_grad_func)
        cost_pre=testing_loss(net_reg,amb_set,my_proj,myprob.loss_func,test_loader,dim,nbrs,cali_data,my_proj,pool,myprob).mean()
        print('Prediction-focused learning: ambiguity set=',amb_set, 'testing loss=',cost_pre)
        record_pre[amb].append(cost_pre)
        K=4
        M=7
        lamda=0.0
        lr=0.0001
        optimizer = optim.Adam(net_reg.parameters(),lr=lr)
        for epoch in range(epoches):
            cost_total=np.array([0.0])
            proj_loss=torch.tensor([0.0])
            for step, (batch_x, batch_y) in tqdm(enumerate(loader)):
                x=batch_x.reshape(-1,1).to(torch.float32)
                y=[batch_y[:,i].reshape(-1,1,1).to(torch.float32) for i in range(dim)]
                asd=net_reg(x) 
                knns=knnsearch(nbrs,cali_data,x)
                my_proj.updata_knn(knns)
                app=my_proj(*asd)
                if amb_set=='SOC-III':
                    params=zip(list(zip(*[to_numpy(app[i]) for i in range(len(app)-1)])),[K for _ in range(len(x))])
                if amb_set=='SOC-I':
                    params=zip(list(zip(*[to_numpy(app[i]) for i in range(2)])),[K for _ in range(len(x))])
                if amb_set=='SOC-II':
                    params=zip(list(zip(*[to_numpy(app[i+2]) for i in range(2)])),[K for _ in range(len(x))])

                res1 = pool.starmap(myprob.solve_model, params)
                ast=zip(res1,[energy_coef for _ in range(len(res1))],[M for _ in range(len(res1))])
                ressult=pool.starmap(myprob.sampling, ast)
                xds=[np.concatenate([p[i] for p in ressult]) for i in range(dim)]
                probas=np.concatenate([p[-2] for p in ressult])
                obj_opt=np.stack([p[-1] for p in ressult])
                my_dro.data_updata(xds,probas,obj_opt,y)
                if amb_set=='SOC-III':
                    (my_dro(*[app[i] for i in range(len(app)-1)]) + lamda * app[-1]).mean().backward()
                if amb_set=='SOC-I':
                    (my_dro(*[app[i] for i in range(2)]) + lamda * app[-1]).mean().backward()
                if amb_set=='SOC-II':
                    (my_dro(*[app[i+2] for i in range(2)]) + lamda * app[-1]).mean().backward()
                t5=time.time()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    xds_opt=[np.stack([p[i][0] for p in res1]) for i in range(dim) ]
                    xcs_opt=[np.stack([p[i+dim][0] for p in res1]) for i in range(dim)]
                    loss=myprob.loss_func(xds_opt,xcs_opt,[to_numpy(p) for p in y]).mean()
                    proj_loss=proj_loss+app[-1].mean()
                    cost_total=cost_total+loss
                print('epoch=',epoch,'step=',step,'training_loss=',cost_total/(step+1),'projection_loss=',proj_loss/(step+1))

            print('epoch=',epoch,'training_loss=',cost_total/(step+1),'projection_loss=',proj_loss/(step+1),'valid_loss=',
                          testing_loss(net_reg,amb_set,my_proj,myprob.loss_func,valid_loader,dim,nbrs,cali_data,my_proj,pool,myprob).mean())

            if epoch % 20 ==0:
                if epoch > 0:
                    lr=lr/5
                    optimizer = optim.Adam(net_reg.parameters(),lr=lr)
                    energy_coef=energy_coef/3
                    pro,param_dro,param_int,vari=myprob.get_dro_problem()
                    my_dro=MIDROLayer(pro,param_dro,param_int,vari,energy_coef,myprob.loss_func,myprob.loss_grad_func)
        cost_df=testing_loss(net_reg,amb_set,my_proj,myprob.loss_func,test_loader,dim,nbrs,cali_data,my_proj,pool,myprob).mean()
        print('Decision-focused learning: ambiguity set=',amb_set, 'testing loss=',cost_df)           
        record_df[amb].append(cost_df)
    pool.close()
    pool.join()