from cvxpylayers.torch import CvxpyLayer
from my_layer_MI import *
from my_problem_MI import *

#=========================================================================================================
# Experiment 1: Varing problem dimension
#=========================================================================================================
n_z=5 # dimension of covariate z
amb_set='SOC_full'
energy_coef=10 # \lambda in (10)
k=20 # k in kNN method
batch_size=100
data_size=[2500,500,1000] # [training data, validation data, testing data]
K=3 # solve the DRO problem for K times to construct the proposal distribution
M=4 # sample M samples from the proposal distribution
num_pool=4 # number of thread to solve DRO problem
epoches=30 # training epoches

set_seed=100


if __name__ == '__main__':
    for n_y in [20,40,60]: # number of assets

        torch.manual_seed(set_seed)
        net_reg=RegressionLayer(n_z,n_y,amb_set)

        int_dim=int(n_y/5) # number of assets with integer dicision

        Mi_PM=MI_portfolio(n_z,n_y,int_dim,amb_set,K,M,energy_coef,num_pool,set_seed)

        Mi_PM.get_data(batch_size, data_size)
        Mi_PM.update_knn(k=k,algorithm='kd_tree') # define kNN search method


        Mi_PM.pretraining(net_reg,epoches=10) # Prediction-focused pretraining

        Mi_PM.update_knn(k=k,algorithm='kd_tree') # define kNN search method
        
        pro,param_dro,param_int,vari=Mi_PM.get_dro_problem()
        dro_layer=MIDROLayer(pro,param_dro,param_int,vari,energy_coef,Mi_PM.loss_func,Mi_PM.loss_grad_func) # define DRO Layer with mixed-integer decision

        problem,params,proj_params=Mi_PM.get_proj_problem(k) 
        proj_layer=CvxpyLayer(problem,params,proj_params) # define Projection later

        PFL_test_score=Mi_PM.test_score(net_reg,proj_layer) # testing score
        print('Experiment 1: dim=',n_y,'PFL=',PFL_test_score)

        Mi_PM.training(net_reg,proj_layer,dro_layer,epoches) # Decision-focused training
        DFL_test_score=Mi_PM.test_score(net_reg,proj_layer) # testing score
        print('Experiment 1: dim=',n_y,'DFL=',DFL_test_score)



#=========================================================================================================
# Experiment 2: Varing ambiguity set
#=========================================================================================================
n_z=5 # dimension of covariate z
n_y=60 # number of assets
int_dim=12 # number of assets with integer dicision
energy_coef=10 # \lambda in (10)
k=20 # k in kNN method
batch_size=100
data_size=[2500,500,1000]
K=3 # solve the DRO problem for K times to construct the proposal distribution
M=4 # sample M samples from the proposal distribution
num_pool=5 # number of thread to solve DRO problem
epoches=30# training epoches

set_seed=100

if __name__ == '__main__':

    PFL_scores=[]
    DFL_scores=[]
    for amb_set in ['SOC_15','SOC30','SOC_full']: # type of ambiguity set, corresponding to (97), (98), and (99) in the paper

        torch.manual_seed(set_seed)
        net_reg=RegressionLayer(n_z,n_y,amb_set)

        Mi_PM=MI_portfolio(n_z,n_y,int_dim,amb_set,K,M,energy_coef,num_pool,set_seed)

        Mi_PM.get_data(batch_size, data_size)
        Mi_PM.update_knn(k=k,algorithm='kd_tree') # define kNN search method

        Mi_PM.get_data(batch_size, data_size, set_seed)
        Mi_PM.update_knn(k=k,algorithm='kd_tree')

        Mi_PM.pretraining(net_reg,epoches=10)

        Mi_PM.update_knn(k=k,algorithm='kd_tree') # define kNN search method

        pro,param_dro,param_int,vari=Mi_PM.get_dro_problem()
        dro_layer=MIDROLayer(pro,param_dro,param_int,vari,energy_coef,Mi_PM.loss_func,Mi_PM.loss_grad_func)

        problem,params,proj_params=Mi_PM.get_proj_problem(k,'no_update')
        proj_layer=CvxpyLayer(problem,params,proj_params)


        PFL_test_score=Mi_PM.test_score(net_reg,proj_layer)
        print('Experiment 2: ambiguity set='+amb_set,'PFL=',PFL_scores)

        Mi_PM.training(proj_layer,dro_layer,epoches)
        DFL_test_score=Mi_PM.test_score(net_reg,proj_layer)
        print('Experiment 2: ambiguity set='+amb_set,'DFL=',DFL_scores)