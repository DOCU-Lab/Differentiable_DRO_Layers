from my_problem_news import *

epoches,num_pool = 60,3

if __name__ == '__main__':
    for training_data_size in [100,200,400,800]:

        training(epoches,training_data_size,num_pool)
