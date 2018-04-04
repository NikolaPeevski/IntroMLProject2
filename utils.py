import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import linear_model, datasets, model_selection
from pylab import *

from sklearn.metrics import mean_squared_error, r2_score

def loadDataSet():
    """
    Simply loading the dataSet
    """
    return pd.read_csv('KidneyData2.csv')

def clearConsole():
    """
    Hacky way of having a clean console
    """
    print("\n" * 100)
    return

def exampleRegression(dataSet, column, target):
    #http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html really good mat
    dataSet_X = dataSet.values[:, np.newaxis, column]
    print(len(dataSet_X))
    dataSet_X_train = dataSet_X[: 125] #125 = 80%
    dataSet_X_test = dataSet_X[-31:] #31 = 20%
    print(dataSet_X_train)
    print(dataSet_X_test)
    

    dataSet_y = dataSet.values[: np.newaxis, target]
    dataSet_y_train = dataSet_y[: 125]
    dataSet_y_test = dataSet_y[-31:]
    print(dataSet_y_test)

    regr = linear_model.LinearRegression()
    regr.fit(dataSet_X_train, dataSet_y_train)

    dataSet_y_pred = regr.predict(dataSet_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(dataSet_y_test, dataSet_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(dataSet_y_test, dataSet_y_pred))
    # Plot outputs
    plt.scatter(dataSet_X_test, dataSet_y_test,  color='black')
    plt.plot(dataSet_X_test, dataSet_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

def errorCheck(X, y):

    # Attribte definition
    X_train = X[: 125] #125 = 80%
    X_test = X[-31:] #31 = 20%

    # Target defenition
    y_train = y[: 125] #125 = 80%
    y_test = y[-31:] #31 = 20%

    #Creating regression model
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    return mean_squared_error(y_test, y_pred)

def glm_validate(X,y,cvf=10):
    ''' Validate linear regression model using 'cvf'-fold cross validation.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns MSE averaged over 'cvf' folds.

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds        
    '''
    from sklearn import model_selection, linear_model
    CV = model_selection.KFold(n_splits=cvf)
    validation_error=np.empty(cvf)
    f=0
    for train_index, test_index in CV.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        m = linear_model.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        validation_error[f] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
        f=f+1
    return validation_error.mean()

def feature_selector_lr(X,y,cvf=10,features_record=None,loss_record=None,display=''):
    ''' Function performs feature selection for linear regression model using
        'cvf'-fold cross validation. The process starts with empty set of
        features, and in every recurrent step one feature is added to the set
        (the feature that minimized loss function in cross-validation.)

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds

        Returns:
        selected_features   indices of optimal set of features
        features_record     boolean matrix where columns correspond to features
                            selected in subsequent steps
        loss_record         vector with cv errors in subsequent steps
        
        Example:
        selected_features, features_record, loss_record = ...
            feature_selector_lr(X_train, y_train, cvf=10)
            
    ''' 

    # first iteration error corresponds to no-feature estimator
    if loss_record is None:
        loss_record = np.array([np.square(y-y.mean()).sum()/y.shape[0]])
    if features_record is None:
        features_record = np.zeros((X.shape[1],1))

    # Add one feature at a time to find the most significant one.
    # Include only features not added before.
    selected_features = features_record[:,-1].nonzero()[0]
    min_loss = loss_record[-1]
    if display is 'verbose':
        print(min_loss)
    best_feature = False
    for feature in range(0,X.shape[1]):
        if np.where(selected_features==feature)[0].size==0:
            trial_selected = np.concatenate((selected_features,np.array([feature])),0).astype(int)
            # validate selected features with linear regression and cross-validation:
            trial_loss = glm_validate(X[:,trial_selected],y,cvf)
            if display is 'verbose':
                print(trial_loss)
            if trial_loss<min_loss:
                min_loss = trial_loss 
                best_feature = feature

    # If adding extra feature decreased the loss function, update records
    # and go to the next recursive step
    if best_feature is not False:
        features_record = np.concatenate((features_record, np.array([features_record[:,-1]]).T), 1)
        features_record[best_feature,-1]=1
        loss_record = np.concatenate((loss_record,np.array([min_loss])),0)
        selected_features, features_record, loss_record = feature_selector_lr(X,y,cvf,features_record,loss_record)
        
    # Return current records and terminate procedure
    return selected_features, features_record, loss_record

def bmplot(yt, xt, X):
    ''' Function plots matrix X as image with lines separating fields. '''
    imshow(X,interpolation='none',cmap='bone')
    xticks(range(0,len(xt)), xt)
    yticks(range(0,len(yt)), yt)
    for i in range(0,len(yt)):
        axhline(i-0.5, color='black')
    for i in range(0,len(xt)):
        axvline(i-0.5, color='black')

def rlr_validate(X,y,lambdas,cvf=10):
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.

        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''
    from sklearn import model_selection
    CV = model_selection.KFold(cvf)
    M = X.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    for train_index, test_index in CV.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            w[:,f,l] = linalg.lstsq(XtX+lambdas[l] * np.eye(M),Xty)[0].squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).sum()/y_train.shape[0]
            test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).sum()/y_test.shape[0]
    
        f=f+1

    opt_val_err = np.min(np.mean(test_error,0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,0))]
    train_err_vs_lambda = np.mean(train_error,0)
    test_err_vs_lambda = np.mean(test_error,0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,1))
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda


def crossValidation(X, y, attributeNames, K = 10):
    ''' Simple cross validation for an X matrix and y target'''
    print(X)
    print(y)
    N, M = X.shape
    print(X)

    CV = model_selection.KFold(n_splits=K, shuffle=True)
    Features = np.zeros((M,K))
    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))
    Error_train_fs = np.empty((K,1))
    Error_test_fs = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))

    k=0

    for train_index, test_index in CV.split(X):

        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
        internal_cross_validation = 10  

        # Compute squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

        # Compute squared error with all features selected (no feature selection)
        m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
        Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

        # Compute squared error with feature subset selection
        #textout = 'verbose';
        textout = ''
        selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)

        Features[selected_features,k]=1
        # .. alternatively you could use module sklearn.feature_selection
        if len(selected_features) is 0:
            print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
        else:
            m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
            Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
            Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
        
            figure(k)
            subplot(1,2,1)
            plot(range(1,len(loss_record)), loss_record[1:])
            xlabel('Iteration')
            ylabel('Squared error (crossvalidation)')    
            
            subplot(1,3,3)
            bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
            clim(-1.5,0)
            xlabel('Iteration')
        
        print('Cross validation fold {0}/{1}'.format(k+1,K))
        print('Train indices: {0}'.format(train_index))
        print('Test indices: {0}'.format(test_index))
        print('Features no: {0}\n'.format(selected_features.size))

        k+=1
    # Display results
    print('\n')
    print('Linear regression without feature selection:\n')
    print('- Training error: {0}'.format(Error_train.mean()))
    print('- Test error:     {0}'.format(Error_test.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    print('Linear regression with feature selection:\n')
    print('- Training error: {0}'.format(Error_train_fs.mean()))
    print('- Test error:     {0}'.format(Error_test_fs.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

    figure(k)
    subplot(1,3,2)
    bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
    clim(-1.5,0)
    xlabel('Crossvalidation fold')
    ylabel('Attribute')

    f=2 # cross-validation fold to inspect
    ff=Features[:,f-1].nonzero()[0]
    if len(ff) is 0:
        print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X[:,ff], y)
        
        y_est= m.predict(X[:,ff])
        residual=y-y_est
        
        figure(k+1, figsize=(12,6))
        title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
        for i in range(0,len(ff)):
            subplot(2,np.ceil(len(ff)/2.0),i+1)
            plot(X[:,ff[i]],residual,'.')
            xlabel(attributeNames[ff[i]])
            ylabel('residual error')
        
        print(X[:,ff])
                  
        
    show()

def lambdaOptimalRegulation(X, y, attributeNames):
    N, M = X.shape

    # Add offset attribute
    X = np.concatenate((np.ones((X.shape[0],1)),X),1)
    attributeNames = [u'Offset']+attributeNames
    M = M+1

    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 5
    CV = model_selection.KFold(K)

    # Values of lambda
    lambdas = np.power(10.,range(-5,9))

    # Initialize variables
    #T = len(lambdas)
    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))
    Error_train_rlr = np.empty((K,1))
    Error_test_rlr = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    w_rlr = np.matrix(np.empty((M,K)))
    w_noreg = np.matrix(np.empty((M,K)))

    k = 0

    for train_index, test_index in CV.split(X,y):
    
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        internal_cross_validation = 10    
        
        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        
        # Compute mean squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

        # Estimate weights for the optimal value of lambda, on entire training set
        w_rlr[:,k] = np.linalg.lstsq(XtX+opt_lambda*np.eye(M),Xty)[0]
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum()/y_train.shape[0]
        Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum()/y_test.shape[0]

        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:,k] = np.linalg.lstsq(XtX,Xty)[0]
        # Compute mean squared error without regularization
        Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum()/y_train.shape[0]
        Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum()/y_test.shape[0]
        # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
        #m = lm.LinearRegression().fit(X_train, y_train)
        #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
        #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T,'.-')
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')    
        
        subplot(1,2,2)
        title('Optimal lambda = {0}'.format(opt_lambda))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        
        print('Cross validation fold {0}/{1}:'.format(k+1,K))
        print('Train indices: {0}'.format(train_index))
        print('Test indices: {0}\n'.format(test_index))

        k+=1

    # Display results
    print('\n')
    print('Linear regression without feature selection:\n')
    print('- Training error: {0}'.format(Error_train.mean()))
    print('- Test error:     {0}'.format(Error_test.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    print('Regularized Linear regression:')
    print('- Training error: {0}'.format(Error_train_rlr.mean()))
    print('- Test error:     {0}'.format(Error_test_rlr.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

    show()

