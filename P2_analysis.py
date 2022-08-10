'''
@Auther: Shixiang WANG
@EID: sxwang6
@Date: 17/02/2022

@Discription:
CS5487 Programming Assignment1-Part2-A real world regression problem-counting people
a:  Let us ﬁrst look at using the features directly, i.e., set φ (x) = x. Use the training set (trainx, trainy), estimate a function with some of the regression algorithms above. 
    Use the test set inputs testx to predict the output, and compare the predictions to the true output testy. 
    (You can round the function outputs so they are counting numbers). 
    Calculate the mean-absolute error and mean-squared error. 
    Which method works the best? Plot the test predictions and the true counts, and discuss any interesting ﬁndings.
b:  Now try some other feature transformations. 
    For example, you can create a simple 2nd order polynomial as φ (x) = [x1,···,x9,x1^2,···,x9^2 ]'. This could be expanded to include more crossterms xixj. Try some other non-linear transformation of the features. Can you improve your results from (a)?
'''

import P1_model as model
import numpy as np
import matplotlib.pyplot as plt

def dataloader(sample = True) -> tuple:
    """load train and test data

    Parameters
    ----------
    sample, optional
        if sample is true, the function will return train data, else test data, by default True

    Returns
    -------
        return input values and output observations
    """    
    path = 'PA-1-data-text/count_data_train' if sample else 'PA-1-data-text/count_data_test'
    with open(path + 'x.txt') as f:
        feature = f.readline()
        feature = feature.split()
        x = np.array([[float(value) for value in feature]])
        for feature in f:
            feature = feature.split()
            x = np.concatenate((x, np.array([[float(value) for value in feature]])), axis=0)
    y = []
    with open(path + 'y.txt') as f:
        for feature in f:
            y.append(float(feature))
    y = np.array(y)
    y = y.reshape(len(y), 1)
    return x, y

def transformer(feature) -> np.ndarray:
    """transform the input feature into  2nd order polynomial as φ (x) = [x1,···,x9,x1^2,···,x9^2 ]

    Parameters
    ----------
    feature
        original input feature

    Returns
    -------
        2nd order polynomial feature
    """    
    tmp = feature.copy()
    squarex = np.square(feature)
    x = np.concatenate((tmp, squarex), axis = 0)
    return x


def error_analysis(Question_b = False) -> None:
    """Error analysis: different models, original feature and polynomial feature

    Parameters
    ----------
    Question_b, optional
        if question_b is true, it will show my PA1-part2-question b's solution, by default False, it will show my question a's solution.
    """    

    models = ['LS', 'RLS', 'LASSO', 'BR', 'RR']

    samplex, sampley = dataloader()
    polyx,  polyy = dataloader(sample=False)

    if Question_b:
        samplex = transformer(samplex)
        polyx = transformer(polyx)

    for m in models:

        mse, mae, theta, mu = 0, 0 ,0, 0
        if m == 'LS':
            theta = model.LS_train(samplex, sampley)
            prediction = model.LS_prediction(polyx, theta)
        elif m == 'RLS':
                theta = model.RLS_train(samplex, sampley)
                prediction = model.RLS_prediction(polyx, theta)
        elif m == 'LASSO':
                theta = model.LASSO_train(samplex, sampley)
                prediction = model.LASSO_prediction(polyx, theta)
        elif m == 'BR':
                mu, var = model.BR_train(samplex, sampley, alpha = 1)
                prediction, _ = model.BR_prediction(polyx, mu, var)
        else:
            theta = model.RR_train(samplex, sampley)
            prediction = model.RR_prediction(polyx, theta)
        
        prediction = np.round(prediction)
        mse = model.MSE(prediction, polyy)
        mae = model.MAE(prediction, polyy)
        print(f'Model:{m}, mse: {mse}, mae: {mae}')
        plt.figure(models.index(m)+1)
        plt.title(f'Model: {m}')
        plt.plot(prediction, 'y', label='prediction')
        plt.plot(polyy, 'm', label='observation')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # question a
    print("******question a start******")
    error_analysis()
    print("******question a finish******")

    # question b
    print("******question b start******")
    error_analysis(Question_b = True)
    print("******question b finish******")