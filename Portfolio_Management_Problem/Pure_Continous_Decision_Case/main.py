from cvxpylayers.torch import CvxpyLayer
import numpy as np
import numpy as np
from my_layer import *
from my_problem import *
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, sys.path[0]+"/Costa/")

from Costa import costa


n_z=5 # dimension of covariate z
n_y=40
k=20 # k in kNN method
batch_size=100 # batch size in training
data_size=[2500,500,1000] # [training data, validation data, testing data]
set_seed=100

con_PM=portfolio_continuous(n_z,n_y,set_seed)
problems,param,vari=con_PM.get_dro_problem()
drolayer = CvxpyLayer(problems,param,vari) # Define DRO Layer

problem,params,proj_params=con_PM.get_proj_problem(k)
proj_layer=CvxpyLayer(problem,params,proj_params)# Define Projection Layer

if __name__ == '__main__':
    con_PM.get_data(batch_size, data_size)
    con_PM.update_knn(k=k,algorithm='kd_tree')

    # Prediction-focued pretraining
    con_PM.pretraining(epoches=10)
    PFL_test_score=con_PM.test_score(proj_layer,drolayer)

    # Decision-focued training
    con_PM.training(proj_layer,drolayer,epoches=60)
    DFL_test_score=con_PM.test_score(proj_layer,drolayer)

    # Compare with method in "Costa, G. and Iyengar, G. N. Distributionally robust end-to-end portfolio construction. Quantitative Finance, 23(10): 1465–1482, 2023"
    dr_net_2layer=costa.Costa_training(n_z,n_y,60,data_size,set_seed)
    Costa_test_score=costa.Costa_testing_score(dr_net_2layer,n_z=5,n_y=40,data_size=[2500,500,1000],set_seed=100)

    # Print results
    print('Prediction-focused learning score=',PFL_test_score.mean(),'Decision-focused learning score=',DFL_test_score.mean(),
          'Costa & Iyengar (2023) score:',Costa_test_score.mean())

    # Wealth evolution plot
    color=['#2171b5','#238b45','#f6ce48']
    methodss=[1+0.1*DFL_test_score,
          1+0.1*DFL_test_score,
          1+0.1*Costa_test_score]
    method_cumu=[np.cumprod(methodss[i]) for i in range(3)]
    plt.figure(1,[7,3.5])
    plt.plot(range(1000),100*method_cumu[0].reshape(-1),'o-',color = color[0],markersize=0.01,label='DFL-SOC')
    plt.plot(range(1000),100*method_cumu[1].reshape(-1),'o-',color = color[1],markersize=0.01,label='PFL-SOC')
    plt.plot(range(1000),100*method_cumu[2].reshape(-1),'o-',color = color[2],markersize=0.01,label='Costa \& Iyengar (2023)')
    plt.xlabel('Day')
    plt.ylabel('Total wealth')
    plt.legend(loc='best')
    plt.show()

