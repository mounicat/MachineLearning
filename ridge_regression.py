import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10

def linear_regression_no_ridge(data,power):
    predictors=['x']
    model = LinearRegression()
    if power>=2:
        predictors.extend(['x_%d'%i for i in range(2,power+1)])
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret

def ridge_regression(data, predictors, alpha,models_to_plot={}): 
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],data['y'],'.')
        plt.plot(data['x'],y_pred,'r^')
        plt.title('Plot for alpha: %.3g'%alpha)
        
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
  #  print "alpha : ", alpha
    return ret

if __name__ == "__main__":
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
    training_data = pd.read_csv("train_data.csv",dtype=dtype_dict)
 #   training_data = training_data[:50]
    training_data = training_data.sort_values(['sqft_living','price'])
    l2_small_penalty = 1.5e-5
    x =  pd.DataFrame({'x':training_data.sqft_living})
    y =  pd.DataFrame({'y':training_data.price})

    data = pd.DataFrame(np.column_stack([x,y]),columns=['sqft_living','price'])

    data['x'] =  training_data['sqft_living']
    data['y'] =  training_data['price']
   # plt.plot(data['x'],data['y'],'.')
   # plt.show()
    for i in range(2,16):  
        colname = 'x_%d'%i    
        data[colname] = data['x']**i
   # print data.head()
    print linear_regression_no_ridge(data,1)
    print linear_regression_no_ridge(data,5)
    print linear_regression_no_ridge(data,15)


    #Initialize the dataframe for storing coefficients.
    alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
    ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
    coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)
    models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
    print "After ridge"
    predictors=['x']
    predictors.extend(['x_%d'%i for i in range(2,16)])
    alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
    for i in range(10):
        coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)  
    plt.show()
