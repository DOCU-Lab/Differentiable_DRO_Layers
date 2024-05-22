import cvxpy as cp
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch.utils.data as data
import torch.optim as optim
from tqdm import tqdm
from my_util_func import *

from my_layer import *

class portfolio_continuous():

    def __init__(self,n_z=5,n_y=40,set_seed=100):
        """
        Inputs
        dim: Integer. Number of assets
        """
        self.n_z=n_z
        self.dim=n_y
        self.set_seed=set_seed
        torch.manual_seed(self.set_seed)
        self.net_reg=RegressionLayer(self.n_z,self.dim)

    def get_dro_problem(self):
        """
        Get DR portfolio management problem

        Returns
        problems: cvxpy problem object. DRO problem
        [sigma,h[-1]]: List of cvxpy parameter object. Ambiguity set parameter
        [x]: List of cvxpy variable object. Output decisions
        """
            
        h=[cp.Parameter((self.dim,1))]
        sigma=cp.Parameter((self.dim,1))
        x=cp.Variable((self.dim,1))
        rs=cp.Variable((1,1))
        beta=[cp.Variable((self.dim,1))]
        eta=[cp.Variable((1+2,1)) for _ in range(self.dim)] # dual vaiable eta
        constraints=[cp.sum(x,axis=0)==1,x>=0]+[beta[i]>=0 for i in range(len(beta))]
        constraints=constraints+[cp.SOC(eta[i][-1],eta[i][0:-1],axis=0) for i in range(self.dim)]
        constraints=constraints+[-x[j]==eta[j][0] for j in range(self.dim)]
        constraints=constraints+[-beta[-1][j]==np.array([[0.0],[0.5],[-0.5]]).T @ eta[j] for j in range(self.dim)]
        constraints=constraints+[rs>=cp.sum( [ h[0][j] @ eta[j][0] - np.array([[0.0],[-0.5],[-0.5]]) .T @ eta[j] for j in range(self.dim) ])]
        obj=cp.Variable((1,1))
        constraints.append(obj>=rs + beta[-1].T @ sigma )
        objective = cp.Minimize(obj)
        problems = cp.Problem(objective, constraints)
        assert problems.is_dpp()
        return problems,[sigma,h[-1]],[x] # return the problem, parameters, and the variables



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
            dataset=list(zip(side_train_s,returns_train_s)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        self.valid_loader=data.DataLoader(
            dataset=list(zip(side_val_s,returns_val_s)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        self.test_loader=data.DataLoader(
            dataset=list(zip(side_test_s,returns_test_s)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        self.cali_data_side=side_train_s
        self.cali_data_return=returns_train_s
    
    def pretraining(self,epoches=10):
        """Prediction-focused pretraining
        pre-training 1: training \mu
        pre-training 2: training \sigma"""
        lr=0.1
        optimizer = optim.Adam(self.net_reg.parameters(),lr=lr)

        for epoch in range(epoches):
            cost_total=torch.tensor([0.0])
            for step, (batch_x, batch_y) in enumerate(self.train_loader):

                asd=self.net_reg(batch_x) 
                knns=self.knnsearch(self.nbrs,self.cali_data_return,batch_x.reshape(-1, self.n_z))
                cost=torch.tensor([0.0])
                for i in range(self.k):
                    knn_sig=torch.stack([knn[i].reshape(self.dim,1) for knn in knns])
                    cost=cost+torch.square(asd[-1]-knn_sig).sum()
                cost=cost/(self.batch_size*self.k)
                cost.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    cost_total=cost_total+cost
                    print('pre-training 1, epoch=',epoch,'step=',step,'training_loss=',cost_total/(step+1))


            if (epoch+1) % 5 ==0:
                lr=lr/5
                optimizer = optim.Adam(self.net_reg.parameters(),lr=lr)

        lr=0.1
        optimizer = optim.Adam(self.net_reg.parameters(),lr=lr)

        for epoch in range(epoches):
            cost_total=torch.tensor([0.0])
            for step, (batch_x, batch_y) in enumerate(self.train_loader):
                cost1=torch.zeros([batch_x.shape[0],self.dim,1])

                asd=self.net_reg(batch_x)
                knns=self.knnsearch(self.nbrs,self.cali_data_return,batch_x.reshape(-1, self.n_z))

                h_pred=to_numpy(asd[-1])
                h_pred=to_torch(h_pred,torch.float64,'cpu')

                for i in range(self.k):
                    knn_sig=torch.stack([knn[i].reshape(self.dim,1) for knn in knns])
                    cost1=cost1+torch.square(h_pred-knn_sig)
                
                cost1=cost1/self.k

                losss=torch.square(cost1-asd[0]).mean()
                losss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    cost_total=cost_total+losss

                    print('pre-training 2 epoch=',epoch,'step=',step,'training_loss=',cost_total/(step+1))
            if (epoch+1) % 5 ==0:
                lr=lr/5
                optimizer = optim.Adam(self.net_reg.parameters(),lr=lr)
        torch.save(self.net_reg,'PFL_net')


    def train_once(self,proj_layer,cvxpylayer,loss,loader,optimizer,epoch):
        cost_total=torch.tensor([0.0])
        proj_loss=torch.tensor([0.0])
        for step, (batch_x, batch_y) in tqdm(enumerate(loader)):
            optimizer.zero_grad()
            asd=self.net_reg(batch_x)
            knns=self.knnsearch(self.nbrs,self.cali_data_return,batch_x.reshape(-1,self.n_z))
            knns=np.stack(knns)
            knns=to_torch(knns,batch_x.dtype,batch_x.device)
            app=proj_layer(asd[0],asd[1],knns,solver_args = {'solve_method': 'SCS'})
            decis=cvxpylayer(app[0],app[1],solver_args = {'solve_method': 'SCS'})
            cost=loss(decis,batch_y-1)
            cost.backward()
            steps=step
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                proj_loss=proj_loss+app[-1].mean()
                cost_real=cost=loss(decis,batch_y)
                cost_total=cost_total+cost_real.item()
            print('training epoc=',epoch,'step=',steps+1,'training_score=',-cost_real.item(),'projection_loss=',proj_loss/(steps+1))
        return -cost_total
        
    def training(self,proj_layer,cvxpylayer,epoches):
        """Decision-focused training"""
        lr=0.00005
        optimizer = optim.Adam(self.net_reg.parameters(),lr=lr)
        for epoch in range(epoches):
            cost_total=torch.tensor([0.0])
            cost_total=self.train_once(proj_layer,cvxpylayer,loss,self.train_loader,optimizer,epoch)

            print('epoch=',epoch,'training_score=',cost_total/(25),'valid_score=',self.valid_score(proj_layer,cvxpylayer))

    def valid_loss(self,proj_layer,cvxpylayer,loader):
        with torch.no_grad():
            cost_tal=torch.tensor([0.0])
            for step, (batch_x, batch_y) in tqdm(enumerate(loader)):
                asd=self.net_reg(batch_x)
                knns=self.knnsearch(self.nbrs,self.cali_data_return,batch_x.reshape(-1, self.n_z))
                knns=np.stack(knns)
                knns=to_torch(knns,batch_x.dtype,batch_x.device)
                app=proj_layer(asd[0],asd[1],knns,solver_args = {'solve_method': 'SCS'})
                decis=cvxpylayer(app[0],app[1],solver_args = {'solve_method': 'SCS'})
                cost=loss(decis,batch_y)
                cost_tal=cost_tal+cost
                steps=step
            return -cost_tal/(steps+1)
        

    def valid_show(self,proj_layer,cvxpylayer,loader):
        with torch.no_grad():
            cost_tal=[]
            for step, (batch_x, batch_y) in tqdm(enumerate(loader)):
                asd=self.net_reg(batch_x)
                knns=self.knnsearch(self.nbrs,self.cali_data_return,batch_x.reshape(-1, self.n_z))
                knns=np.stack(knns)
                knns=to_torch(knns,batch_x.dtype,batch_x.device)
                app=proj_layer(asd[0],asd[1],knns,solver_args = {'solve_method': 'SCS'})
                decis=cvxpylayer(app[0],app[1],solver_args = {'solve_method': 'SCS'})
                cost=loss_show(decis,batch_y)
                cost_tal.append(cost)
            cost_tal=np.concatenate(cost_tal).reshape(1,-1)
            return -cost_tal
        
    def valid_score(self,proj_layer,cvxpylayer):
        return self.valid_loss(proj_layer,cvxpylayer,self.valid_loader)

    def test_score(self,proj_layer,cvxpylayer):
        return self.valid_show(proj_layer,cvxpylayer,self.test_loader)