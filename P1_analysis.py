'''
@Auther: Shixiang WANG
@EID: sxwang6
@Date: 16/02/2022

@Discription:
CS5487 Programming Assignment1-Part1-Polynommial function-(b, c ,d ,e)
b:  For each regression method, use the sample data (sampx, sampy) to estimate the parameters of a 5th order polynomial function. 
    Make a plot of the estimated function using polyx as inputs, along with the sample data. 
    For BR, also plot the standard deviation around the mean. 
    What is the mean-squared error between the learned function outputs and the true function outputs (polyy), averaged over all input values in polyx? For algorithms with hyperparameters, select some values that tend to work well.

c:  Repeat (b), but reduce the amount of training data available, by selecting a subset of the samples (e.g., 10%, 25%, 50%, 75%). 
    Plot the estimated functions. Which models are more robust with less data, and which tend to overﬁt? 
    Make a plot of error versus training size, and comment on any important trends and ﬁndings. (You should run multiple trials with diﬀerent random subsets, and take the average error).

d:  Add some outliers output values (e.g., add large numbers to a few values in sampy), and repeat b Which methods are robust to the presence of outliers, and which are most sensitive? Why?

e:  Repeat b but estimate a higher-order polynomial (e.g., 10th order). 
    Which models tend to overﬁt the data when learning a more complex model, and which models do not? 
    Verify your observations by examining the estimated parameter values.
'''

import numpy as np
import matplotlib.pyplot as plt
import P1_model as model

def error_analysis(K=5, Question_b = False, Question_c =False, Outliers = 0.0):
    """Error analysis: different training size, outliers

    Parameters
    ----------
    K, optional
        the highest order of polynomial, by default 5
    Question_b, optional
        if Question_b is true,  it will show part1 question b's solution, by default False
    Question_c, optional
        if Question_c is true,  it will show part1 question c's solution, by default False
    Outliers, optional
        if Outliers > 0.0,  it will add some outliers ouput values (e.g., add large number to a few values in sampy) and show part1 question d's solution, by default 0.0
    """
    ratio = [0.15, 0.25, 0.50, 0.75, 1.0]
    num_trials = 5
    models = ['LS', 'RLS', 'LASSO', 'BR', 'RR']
    mse_matrix = np.ndarray((len(models), len(ratio)), dtype= np.float64)
    shuffle = False if Question_b else True

    for m in models:

        for r in ratio:

            samplex, sampley = model.dataloader(sample = True, shuffle = shuffle, ratio = r)
            if Outliers: sampley = model.add_outliers(sampley, ratio = Outliers)
            polyx, polyy = model.dataloader(sample = False, shuffle = False, ratio = 1)
            sample_phi, polyx_phi = model.transformer(samplex, K), model.transformer(polyx, K)
            # prediction = 0
            mse, theta, mu = 0, 0 ,0

            for j in range(num_trials):

                if m == 'LS':
                    theta = model.LS_train(sample_phi, sampley)
                    prediction = model.LS_prediction(polyx_phi, theta)
                elif m == 'RLS':
                    theta = model.RLS_train(sample_phi, sampley)
                    prediction = model.RLS_prediction(polyx_phi, theta)
                elif m == 'LASSO':
                    theta = model.LASSO_train(sample_phi, sampley)
                    prediction = model.LASSO_prediction(polyx_phi, theta)
                elif m == 'BR':
                    mu, var = model.BR_train(sample_phi, sampley, alpha = 1)
                    prediction, variance = model.BR_prediction(polyx_phi, mu, var)
                    BR_py = np.random.normal(prediction, np.sqrt(np.abs(variance)), size=None)
                else:
                    sample_phi, poly_phi = model.transformer(samplex, k = 5), model.transformer(polyx, k = 5)
                    theta = model.RR_train(sample_phi, sampley)
                    prediction = model.RR_prediction(poly_phi, theta)

                mse += model.MSE(prediction, polyy)

            mse_matrix[models.index(m)][ ratio.index(r)] = mse / num_trials

            if Question_c:
                print(f'Model: {m}, Mean Square Error is :{mse_matrix[models.index(m)][ratio.index(1.0)]}')
                plt.figure(models.index(m)+1)
                plt.title(f'Model: {m}, Kth = {K}, ratio = {r}')
                plt.xlabel('X-Input')
                plt.ylabel('Y-Output')
                if m == 'BR':
                    plt.plot(polyx, BR_py,'.')
                else:
                    plt.plot(polyx, prediction,'.-')
                plt.plot(samplex, sampley,'*')
                plt.show()               
        
        if Question_b:
            print(f'Model: {m}, Mean Square Error is :{mse_matrix[models.index(m)][ratio.index(1.0)]}')
            plt.figure(models.index(m)+1)
            if Outliers:
                plt.title(f'Model: {m}, Kth = {K}, Outliers-ratio = {Outliers}')
            else:
                plt.title(f'Model: {m}, Kth = {K}')
            plt.xlabel('X-Input')
            plt.ylabel('Y-Output')
            if m == 'BR':
                plt.plot(polyx, BR_py,'.')
            else:
                plt.plot(polyx, prediction,'.-')
            plt.plot(samplex, sampley,'*')
            plt.show()

    if Question_c:
        colors = ['b', 'c', 'g', 'm', 'r']
        fig, axs = plt.subplots(5, 1, sharex = True, sharey = False)
        for i in range(len(models)):
            axs[i].plot(ratio, mse_matrix[i, :], color = colors[i], label=models[i])
            axs[i].legend()
        fig.text(0.5, 0.03, "Training Size Ratio", ha="center", va="center")
        fig.text(0.03, 0.5, "Mean Squared Error", ha="center", va="center", rotation = 90)
        fig.suptitle("Training Size vs Mean square Error")
        plt.show()


if __name__ == '__main__':
    # question b
    print("******question b start******")
    error_analysis(K = 5, Question_b = True)
    print("******question b finish******")

    # question c
    print("******question c start******")
    error_analysis(K = 5, Question_c = True)
    print("******question c finish******")

    # question d
    print("******question d start******")
    error_analysis(K = 5, Question_b = True, Outliers = 0.2)
    print("******question d finish******")
    
    # question e
    print("******question e start******")
    error_analysis(K = 10, Question_b = True)
    print("******question e finish******")