import cvxpy as cp
import torch
import torch.optim as optim
import numpy as np
from my_layer_MI import *
from sklearn.neighbors import NearestNeighbors
import torch.utils.data as data
from tqdm import tqdm
from multiprocessing import Pool
from my_util_func import *


class MI_portfolio():

    def __init__(self,n_z,dim,int_dim,amb_set,K,M,energy_coef,num_pool,set_seed=100):
        self.n_z=n_z
        self.dim=dim
        self.int_dim=int_dim
        self.amb_set=amb_set
        self.int_propo=(1/self.dim) * np.ones([self.int_dim,1])
        self.integer_total=2 ** self.int_dim
        
        self.set_seed=set_seed
        self.K=K
        self.M=M
        self.energy_coef=energy_coef
        self.num_pool=num_pool

        if amb_set=='SOC30':
            self.maxtri=np.zeros([int(self.dim/2),self.dim])
            for i in range(int(self.dim/2)):
                self.maxtri[i,2*i]=1
                self.maxtri[i,2*i+1]=1

        if amb_set=='SOC15':
            self.maxtri=np.zeros([int(self.dim/4),self.dim])
            for i in range(int(self.dim/4)):
                self.maxtri[i,4*i]=1
                self.maxtri[i,4*i+1]=1
                self.maxtri[i,4*i+2]=1
                self.maxtri[i,4*i+3]=1


    def get_dro_problem(self):
        """
        Get DR portfolio management problem

        Returns
        problems: cvxpy problem object. DRO problem
        [sigma,h[-1]]: List of cvxpy parameter object. Ambiguity set parameter
        [xd]: List of cvxpy variable object. Output integer decisions
        [xc,obj]: List of cvxpy variable object. Output continuous decisions, objective value
        """

        xs=cp.Variable((self.dim-self.int_dim,1))
        xd=cp.Parameter((self.int_dim,1))
        x=cp.vstack([cp.multiply(self.int_propo,xd),xs])

        constraints=[xs>=0 , cp.sum(x,axis=0) == 1]

        if self.amb_set=='SOC_full':
            h=[cp.Parameter((self.dim,1))]
            sigma=cp.Parameter((self.dim,1))

            rs=cp.Variable((1,1))
            beta=[cp.Variable((self.dim,1))]

            Au=[np.concatenate([np.zeros((self.dim,1)),np.array([[0.5],[-0.5]])])]

            eta=[cp.Variable((1+2,1)) for _ in range(self.dim)] # dual vaiable eta

            constraints=constraints+[beta[i]>=0 for i in range(len(beta))]
            constraints=constraints+[cp.SOC(eta[i][-1],eta[i][0:-1],axis=0) for i in range(self.dim)]
            constraints=constraints+[-x[j]==eta[j][0] for j in range(self.dim)]
            constraints=constraints+[-beta[-1][j]==np.array([[0.0],[0.5],[-0.5]]).T @ eta[j] for j in range(self.dim)]
            constraints=constraints+[rs>=cp.sum( [ h[0][j] @ eta[j][0] - np.array([[0.0],[-0.5],[-0.5]]) .T @ eta[j] for j in range(self.dim) ])]

            obj=cp.Variable((1,1))
            constraints.append(obj>=rs + beta[-1].T @ sigma )

            objective = cp.Minimize(obj)

            problems = cp.Problem(objective, constraints)
            assert problems.is_dpp()
            return problems,[sigma,h[-1]],[xd],[xs,obj] # return the problem, parameters, and the variables

        if self.amb_set=='SOC30':
            h=[cp.Parameter((self.dim,1))]
            sigma=cp.Parameter((int(self.dim/2),1))

            rs=cp.Variable((1,1))
            beta=[cp.Variable((int(self.dim/2),1))]

            Au=[np.concatenate([np.zeros((self.dim,1)),np.array([[0.5],[-0.5]])])]

            eta=[cp.Variable((2+2,1)) for _ in range(int(self.dim/2))] # dual vaiable eta

            constraints=constraints+[beta[i]>=0 for i in range(len(beta))]
            constraints=constraints+[cp.SOC(eta[i][-1],eta[i][0:-1],axis=0) for i in range(len(eta))]
            constraints=constraints+[-x[2*j]==eta[j][0] for j in range(int(self.dim/2))]+[-x[2*j+1]==eta[j][1] for j in range(int(self.dim/2))]
            constraints=constraints+[-beta[-1][j]==np.array([[0.0],[0.0],[0.5],[-0.5]]).T @ eta[j] for j in range(len(eta))]
            constraints=constraints+[rs>=cp.sum( [ h[0][2*j] @ eta[j][0] +h[0][2*j+1] @ eta[j][1]- 
                                                  np.array([[0.0],[0.0],[-0.5],[-0.5]]).T @ eta[j] for j in range(len(eta)) ])]

            obj=cp.Variable((1,1))
            constraints.append(obj>=rs + beta[-1].T @ sigma )

            objective = cp.Minimize(obj)

            problems = cp.Problem(objective, constraints)
            assert problems.is_dpp()
            return problems,[sigma,h[-1]],[xd],[xs,obj] # return the problem, parameters, and the variables
        
        if self.amb_set=='SOC15':
            h=[cp.Parameter((self.dim,1))]
            sigma=cp.Parameter((int(self.dim/4),1))

            rs=cp.Variable((1,1))
            beta=[cp.Variable((int(self.dim/4),1))]

            Au=[np.concatenate([np.zeros((self.dim,1)),np.array([[0.5],[-0.5]])])]

            eta=[cp.Variable((4+2,1)) for _ in range(int(self.dim/4))] # dual vaiable eta

            constraints=constraints+[beta[i]>=0 for i in range(len(beta))]
            constraints=constraints+[cp.SOC(eta[i][-1],eta[i][0:-1],axis=0) for i in range(len(eta))]
            constraints=constraints+[-x[4*j]==eta[j][0] for j in range(int(self.dim/4))]+[-x[4*j+1]==eta[j][1] for j in range(int(self.dim/4))]
            constraints=constraints+[-x[4*j+2]==eta[j][2] for j in range(int(self.dim/4))]+[-x[4*j+3]==eta[j][3] for j in range(int(self.dim/4))]
            constraints=constraints+[-beta[-1][j]==np.array([[0.0],[0.0],[0.0],[0.0],[0.5],[-0.5]]).T @ eta[j] for j in range(len(eta))]
            constraints=constraints+[rs>=cp.sum( [ h[0][2*j] @ eta[j][0] +h[0][2*j+1] @ eta[j][1]+h[0][2*j+2] @ eta[j][2] +h[0][2*j+3] @ eta[j][3]- 
                                                  np.array([[0.0],[0.0],[0.0],[0.0],[-0.5],[-0.5]]).T @ eta[j] for j in range(len(eta)) ])]

            obj=cp.Variable((1,1))
            constraints.append(obj>=rs + beta[-1].T @ sigma )

            objective = cp.Minimize(obj)

            problems = cp.Problem(objective, constraints)
            assert problems.is_dpp()
            return problems,[sigma,h[-1]],[xd],[xs,obj] # return the problem, parameters, and the variables


    def solve_model(self,input_data,num_sols):
        xs=cp.Variable((self.dim-self.int_dim,1))
        xd=cp.Variable((self.int_dim,1),boolean=True)
        x=cp.vstack([cp.multiply(self.int_propo,xd),xs])
        constraints=[xs>=0 , cp.sum(x,axis=0) == 1]

        if self.amb_set=='SOC_full':
            sigma,h=input_data

            rs=cp.Variable((1,1))
            beta=[cp.Variable((self.dim,1))]

            Au=[np.concatenate([np.zeros((self.dim,1)),np.array([[0.5],[-0.5]])])]

            eta=[cp.Variable((1+2,1)) for _ in range(self.dim)] # dual vaiable eta

            constraints=constraints+[beta[i]>=0 for i in range(len(beta))]
            constraints=constraints+[cp.SOC(eta[i][-1],eta[i][0:-1],axis=0) for i in range(self.dim)]
            constraints=constraints+[-x[j]==eta[j][0] for j in range(self.dim)]
            constraints=constraints+[-beta[-1][j]==np.array([[0.0],[0.5],[-0.5]]).T @ eta[j] for j in range(self.dim)]
            constraints=constraints+[rs>=cp.sum( [ h[j] @ eta[j][0] - np.array([[0.0],[-0.5],[-0.5]]) .T @ eta[j] for j in range(self.dim) ])]

            obj=cp.Variable((1,1))
            constraints.append(obj>=rs + beta[-1].T @ sigma )

        if self.amb_set=='SOC30':
            sigma,h=input_data

            rs=cp.Variable((1,1))
            beta=[cp.Variable((int(self.dim/2),1))]


            eta=[cp.Variable((2+2,1)) for _ in range(int(self.dim/2))] # dual vaiable eta

            constraints=constraints+[beta[i]>=0 for i in range(len(beta))]
            constraints=constraints+[cp.SOC(eta[i][-1],eta[i][0:-1],axis=0) for i in range(len(eta))]
            constraints=constraints+[-x[2*j]==eta[j][0] for j in range(int(self.dim/2))]+[-x[2*j+1]==eta[j][1] for j in range(int(self.dim/2))]
            constraints=constraints+[-beta[-1][j]==np.array([[0.0],[0.0],[0.5],[-0.5]]).T @ eta[j] for j in range(len(eta))]

            constraints=constraints+[rs>=cp.sum( [ h[2*j] @ eta[j][0] +h[2*j+1] @ eta[j][1]- 
                                                  np.array([[0.0],[0.0],[-0.5],[-0.5]]).T @ eta[j] for j in range(len(eta)) ])]

            obj=cp.Variable((1,1))
            constraints.append(obj>=rs + beta[-1].T @ sigma )

        if self.amb_set=='SOC15':
            sigma,h=input_data

            rs=cp.Variable((1,1))
            beta=[cp.Variable((int(self.dim/4),1))]


            eta=[cp.Variable((4+2,1)) for _ in range(int(self.dim/4))] # dual vaiable eta

            constraints=constraints+[beta[i]>=0 for i in range(len(beta))]
            constraints=constraints+[cp.SOC(eta[i][-1],eta[i][0:-1],axis=0) for i in range(len(eta))]
            constraints=constraints+[-x[4*j]==eta[j][0] for j in range(int(self.dim/4))]+[-x[4*j+1]==eta[j][1] for j in range(int(self.dim/4))]
            constraints=constraints+[-x[4*j+2]==eta[j][2] for j in range(int(self.dim/4))]+[-x[4*j+3]==eta[j][3] for j in range(int(self.dim/4))]
            constraints=constraints+[-beta[-1][j]==np.array([[0.0],[0.0],[0.0],[0.0],[0.5],[-0.5]]).T @ eta[j] for j in range(len(eta))]

            constraints=constraints+[rs>=cp.sum( [ h[2*j] @ eta[j][0] +h[2*j+1] @ eta[j][1]+h[2*j+2] @ eta[j][2] +h[2*j+3] @ eta[j][3]- 
                                                  np.array([[0.0],[0.0],[0.0],[0.0],[-0.5],[-0.5]]).T @ eta[j] for j in range(len(eta)) ])]

            obj=cp.Variable((1,1))
            constraints.append(obj>=rs + beta[-1].T @ sigma )
        

        objective = cp.Minimize(obj)

        problems = cp.Problem(objective, constraints)

        problems.solve(solver=cp.GUROBI,OptimalityTol=1e-4,PreMIQCPForm=1)

        xd_opt=[]
        xc_opt=[]
        obj_opt=[]
        xd_opt.append(np.around(xd.value))
        xc_opt.append(xs.value)
        obj_opt.append(obj.value)

        for t in range(num_sols-1):
            constraints=constraints+[(np.ones([self.int_dim,1])-xd_opt[-1] @ np.array([[2.0]])).T @ xd + 
                                     np.ones([self.int_dim,1]).T @ xd_opt[-1] >=1]
            problems = cp.Problem(objective, constraints)
            problems.solve(solver=cp.GUROBI,OptimalityTol=1e-4,PreMIQCPForm=1)
            xd_opt.append(np.around(xd.value))
            xc_opt.append(xs.value)
            obj_opt.append(obj.value)
        output=[np.stack(xd_opt),np.stack(xc_opt),np.stack(obj_opt)]

        return tuple(output)

    def get_proj_problem(self,k):
        """
        Get projection problem

        Input:
        k: Integer. k in the k-nearest-neighborhood

        Returns
        problem: cvxpy problem object. Projection problem
        params: List of cvxpy parameter object. Input ambiugity set parameter
        proj_params: List of cvxpy variable object. Output projected ambiugity set parameter
        """
        
        if self.amb_set=='SOC_full':
            h=[cp.Parameter((self.dim,1))]
            sigma=[cp.Parameter((self.dim,1))]
            knn_v=cp.Parameter((k,self.dim))

            h_proj=[cp.Variable((self.dim,1))]
            sigma_proj=[cp.Variable((self.dim,1))]
            obj=cp.Variable((1,1))


            constraints=[cp.square(knn_v.T-h_proj[-1]).mean(axis=1,keepdims=True)<=sigma_proj[0]]

            constraints=constraints+[obj >= cp.sum([5 * cp.pnorm(h[i]-h_proj[i],2) + 
                                        cp.pnorm(sigma[i]-sigma_proj[i],2) for i in range(len(h))]) ]
            
            objective = cp.Minimize(obj)

            problem = cp.Problem(objective, constraints)
            assert problem.is_dpp()

        if self.amb_set=='SOC30':
            h=[cp.Parameter((self.dim,1))]
            sigma=[cp.Parameter((int(self.dim/2),1))]
            knn_v=cp.Parameter((k,self.dim))

            h_proj=[cp.Variable((self.dim,1))]
            sigma_proj=[cp.Variable((int(self.dim/2),1))]
            obj=cp.Variable((1,1))

            
            constraints=[self.maxtri @ cp.square(knn_v.T-h_proj[-1]).mean(axis=1,keepdims=True)
                         <=sigma_proj[0]]

            constraints=constraints+[obj >= cp.sum([5 * cp.pnorm(h[i]-h_proj[i],2) + 
                                        cp.pnorm(sigma[i]-sigma_proj[i],2) for i in range(len(h))]) ]

            objective = cp.Minimize(obj)

            problem = cp.Problem(objective, constraints)
            assert problem.is_dpp()

        if self.amb_set=='SOC15':
            h=[cp.Parameter((self.dim,1))]
            sigma=[cp.Parameter((int(self.dim/4),1))]
            knn_v=cp.Parameter((k,self.dim))

            h_proj=[cp.Variable((self.dim,1))]
            sigma_proj=[cp.Variable((int(self.dim/4),1))]
            obj=cp.Variable((1,1))

            constraints=[self.maxtri @ cp.square(knn_v.T-h_proj[-1]).mean(axis=1,keepdims=True)
                         <=sigma_proj[0]]

            constraints=constraints+[obj >= cp.sum([5 * cp.pnorm(h[i]-h_proj[i],2) + 
                                        cp.pnorm(sigma[i]-sigma_proj[i],2) for i in range(len(h))]) ]

            
            objective = cp.Minimize(obj)

            problem = cp.Problem(objective, constraints)
            assert problem.is_dpp()

        params=[sigma[0],h[0],knn_v]
        proj_params=[sigma_proj[0],h_proj[0],obj]

        return problem,params,proj_params
    

    def update_knn(self,k=20,algorithm='kd_tree'):
        """Define kNN method
        """
        self.k=k
        self.nbrs=NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(self.cali_data_side)

    
    def knnsearch(self,nbrs,cali_data_return,feature):
        """Search for kNN given new feature
        Inputs
        nbrs: kNN method
        cali_data_return: assert return
        feature: New feature
        """
        distances, indices = nbrs.kneighbors(feature)
        return [cali_data_return[indix,:] for indix in indices]
    
    def get_data(self,batch_size=100, data_size=[2500,500,1000]):
        """Generate Data
        data_size=[training data size, validation data size, testing data size]
        """

        self.batch_size=batch_size
        side_train_s, side_val_s, side_test_s, returns_train_s, returns_val_s, returns_test_s=synthetic_nl(self.n_z, self.dim, data_size, self.set_seed)
        side_val_s=side_val_s.to(torch.float32)

        side_train_s=side_train_s.to(torch.float32)

        side_test_s=side_test_s.to(torch.float32)

        returns_train_s=returns_train_s.to(torch.float32)

        returns_val_s=returns_val_s.to(torch.float32)

        returns_test_s=returns_test_s.to(torch.float32)

        self.train_loader = data.DataLoader(
            dataset=list(zip(side_train_s,returns_train_s.unsqueeze(2))),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        self.valid_loader=data.DataLoader(
            dataset=list(zip(side_val_s,returns_val_s.unsqueeze(2))),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        self.test_loader=data.DataLoader(
            dataset=list(zip(side_test_s,returns_test_s.unsqueeze(2))),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        
        self.cali_data_side=side_train_s
        self.cali_data_return=returns_train_s
    
        
    def sampling(self,t_sol,energy_coef,M):
        """Construct the proposal distribution and sampling

        Input
        t_sol: tuple. T solutions to the DRO problem, format: (integer decison, continuous decision, objective value)
        energy_coef: energy coefficient \lambda in (10)
        M: Integer. Number of samples we want
        
        Return
        sample_total: array. Each row corrresponds to an integer decision sampled from the proposal distribution
        prob_total: array. Each row is the probability of the corrresponding sample
        obj_or[0]: smallest objective value of t_sol"""
        decay=np.array([[0.5]])

        xd,xc,obj_or=t_sol

        obj=obj_or-obj_or[0]
        obj=np.exp(-obj / np.array(energy_coef))
        obj_least=decay * obj[-1]

        frac=0.2

        if (self.integer_total-len(obj_or)) * obj_least >= frac * np.sum(obj,axis=0,keepdims=True):
            E_outside =  frac * np.sum(obj)
        else:
            E_outside = (self.integer_total-len(obj_or)) * obj_least

        p_inside=np.concatenate([obj.reshape(-1,1),E_outside.reshape(-1,1)],axis=0)
        p_inside=p_inside/np.sum(p_inside)

        p_inside=p_inside.reshape(-1,1)

        index = np.random.choice([i for i in range(len(xd)+1)], M , p=p_inside.reshape(-1))

        cout_out = index == len(xd)

        cout_in = index != len(xd)

        index_out = index[cout_in]

        sample_in = xd[index_out]

        sample_out = np.random.randint(2,size=[cout_out.sum(),self.int_dim,1])

        sample_total = np.concatenate([sample_in,sample_out],axis=0)

        prob_in = p_inside[index_out] + (1/self.integer_total) * np.ones([cout_in.sum(),1]) @ p_inside[-1].reshape(-1,1) 

        prob_out = np.ones([cout_out.sum(),1]) * (1/self.integer_total) 

        prob_total=np.concatenate([prob_in,prob_out],axis=0).reshape(-1,1,1)

        return sample_total, prob_total, obj_or[0]
        
         
    def loss_func(self,xds,xcs,ys):

        output=[]
        batch_size=len(xds[0])
        for i in range(batch_size):
            x=np.concatenate([xds[0][i],xcs[0][i]],axis=0).reshape(-1,1)
            output.append( - ys[0][i].reshape(-1,1).T @ x )
        return np.stack(output,axis=0)

    
    def loss_grad_func(self,xds,xcs,ys):

        output = - ys[0][:,self.int_dim:,:]

        return [output]
    
    def pretraining(self,net_reg,epoches=10):
        """Prediction-focused pretraining"""
        lr=0.1
        optimizer = optim.Adam(net_reg.parameters(),lr=lr)

        for epoch in range(epoches):
            cost_total=torch.tensor([0.0])
            for step, (batch_x, batch_y) in tqdm(enumerate(self.train_loader)):
            
                y_pred=net_reg(batch_x)
                knns=self.knnsearch(self.nbrs, self.cali_data_return, batch_x.reshape(-1, self.n_z))
                cost=torch.tensor([0.0])
                for i in range(self.k):
                    knn_sig=torch.stack([knn[i].reshape(self.dim,1) for knn in knns])
                    cost=cost+torch.square(y_pred[-1]-knn_sig).sum()
                cost=cost/(self.batch_size*self.k)
                cost.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    cost_total=cost_total+cost

                    print('pre-training 1 epoch=',epoch,'step=',step,'training_loss=',cost_total/(step+1))

            if (epoch+1) % 5 ==0:
                lr=lr/5
                optimizer = optim.Adam(net_reg.parameters(),lr=lr)

        lr=0.1
        optimizer = optim.Adam(net_reg.parameters(),lr=lr)

        for epoch in range(epoches):
            cost_total=torch.tensor([0.0])
            for step, (batch_x, batch_y) in tqdm(enumerate(self.train_loader)):
                cost1=torch.zeros([batch_x.shape[0],self.dim,1])
                cost2=0
                y_pred=net_reg(batch_x)
                knns=self.knnsearch(self.nbrs,self.cali_data_return,batch_x.reshape(-1, self.n_z))

                h_pred=to_numpy(y_pred[-1])
                h_pred=to_torch(h_pred,torch.float64,'cpu')

                for i in range(self.k):
                    knn_sig=torch.stack([knn[i].reshape(self.dim,1) for knn in knns])
                    cost1=cost1+torch.square(h_pred-knn_sig)
                
                cost1=cost1/self.k
                if self.amb_set=='SOC_full':
                    cost2=cost1
                if self.amb_set=='SOC30':
                    cost2=cost1[:,0::2,:]+cost1[:,1::2,:]
                if self.amb_set=='SOC15':
                    cost2=cost1[:,0::4,:]+cost1[:,1::4,:]+cost1[:,2::4,:]+cost1[:,3::4,:]
                losss=torch.square(cost2-y_pred[0]).mean()
                losss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    cost_total=cost_total+losss

                    print('pre-training 2 epoch=',epoch,'step=',step,'training_loss=',cost_total/(step+1))
            if (epoch+1) % 5 ==0:
                lr=lr/5
                optimizer = optim.Adam(net_reg.parameters(),lr=lr)
        return 0

    def train_once(self,net_reg,proj_layer,my_dro,loader,optimizer,epoch):
        """Decision-focused training for a single epoch"""
        pool=Pool(self.num_pool)
        cost_total=np.array([0.0])
        proj_loss=torch.tensor([0.0])
        for step, (batch_x, batch_y) in tqdm(enumerate(loader)):
            optimizer.zero_grad()

            # Learning layer
            y_pred=net_reg(batch_x) 

            # Projection later
            knns=self.knnsearch(self.nbrs,self.cali_data_return,batch_x.reshape(-1, self.n_z))
            knns=np.stack(knns)
            knns=to_torch(knns,batch_x.dtype,batch_x.device) # kNN search
            y_proj=proj_layer(y_pred[0],y_pred[1],knns,solver_args = {'solve_method': 'SCS'}) # project to feasible region

            # DRO Layer
            params=zip(list(zip(*[to_numpy(y_proj[i]) for i in range(2)])),[self.K for _ in range(len(batch_x))])
            sols = pool.starmap(self.solve_model, params) # get T solutions
            temp=zip(sols,[self.energy_coef for _ in range(len(sols))],[self.M for _ in range(len(sols))])
            samplings=pool.starmap(self.sampling, temp) # get M samples
            xds=[np.concatenate([p[0] for p in samplings])]
            probas=np.concatenate([p[-2] for p in samplings])
            obj_opt=np.stack([p[-1] for p in samplings])
            my_dro.data_updata(xds,probas,obj_opt,[batch_y-1])
            my_dro(*[y_proj[0],y_proj[1]]).mean().backward() #DRO Layer
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                xds_opt=[np.stack([p[0][0]* (1/self.dim) for p in sols])]
                xcs_opt=[np.stack([p[1][0] for p in sols])]
                loss=self.loss_func(xds_opt,xcs_opt,[to_numpy(batch_y)]).mean()
                proj_loss=proj_loss+y_proj[-1].mean()
                cost_total=cost_total+loss
            print('training epoch=',epoch,'step=',step+1,'training_loss=',-loss.item(),'projection_loss=',proj_loss/(step+1))
        pool.close()
        pool.join()
        return 10 * cost_total/(step+1)
    
    def training(self,net_reg,proj_layer,my_dro,epoches):
        """Decision-focused training"""
        lr=0.00005
        optimizer = optim.Adam(net_reg.parameters(),lr=lr)

        for epoch in range(epoches):
            cost_total=self.train_once(net_reg,proj_layer,my_dro,self.train_loader,optimizer,epoch)
            print('epoch=',epoch,'training_loss=',-cost_total,'test_loss=',
                    self.validate(net_reg,proj_layer,self.valid_loader))

            if (epoch+1) % 20 ==0:
                    lr=lr/5
                    optimizer = optim.Adam(net_reg.parameters(),lr=lr)
            if (epoch+1) % 30 ==0:
                    self.energy_coef=self.energy_coef/3
                    pro,param_dro,param_int,vari=self.get_dro_problem()
                    my_dro=MIDROLayer(pro,param_dro,param_int,vari,self.energy_coef,self.loss_func,self.loss_grad_func)

    def validate(self,net_reg,proj_layer,loader):
        pool=Pool(self.num_pool)
        with torch.no_grad():
            cost_total=np.array([0.0])
            proj_loss=torch.tensor([0.0])
            for step, (batch_x, batch_y) in tqdm(enumerate(loader)):
                y_pred=net_reg(batch_x)
                knns=self.knnsearch(self.nbrs,self.cali_data_return,batch_x.reshape(-1, self.n_z))
                knns=np.stack(knns)
                knns=to_torch(knns,batch_x.dtype,batch_x.device)
                y_proj=proj_layer(y_pred[0],y_pred[1],knns,solver_args = {'solve_method': 'SCS'})
                params=zip(list(zip(*[to_numpy(y_proj[i]) for i in range(2)])),[1 for _ in range(len(batch_x))])
                sols = pool.starmap(self.solve_model, params)
                xds_opt=[np.stack([p[0][0] * (1/self.dim) for p in sols])]
                xcs_opt=[np.stack([p[1][0] for p in sols])]
                loss=self.loss_func(xds_opt,xcs_opt,[to_numpy(batch_y)]).mean()
                proj_loss=proj_loss+y_proj[-1].mean()
                cost_total=cost_total+loss
        pool.close()
        pool.join()
        return - 10 * cost_total/(step+1)
    
    
    def test_score(self,net_reg,proj_layer):
        return self.validate(net_reg,proj_layer,self.test_loader)

