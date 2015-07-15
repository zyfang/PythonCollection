"""
Generalized Additive Model (GAM) estimator that can be used with sklearn, using GAM implementation in R from the mgcv library 

Note: this is basically a python/scikit-learn wrap around a GAM implementation in R, because I couldn't find a good python package for it.

Requires: R (3.1+)      
"""
__author__ = "Zhou Fang"
__email__ = "yfang@cs.uni-bremen.de"

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.metrics import r2_score
from sklearn.utils.validation import NotFittedError, check_is_fitted
from sklearn.utils import check_array, check_X_y
from rpy2.robjects import r
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri as rpyn
rpyn.activate()

# import R's utility package
utils = rpackages.importr('utils')
# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names we need
packnames = ('mgcv')

# Install necessary packages that haven't been installed yet
packnames_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(packnames_to_install) > 0:
    utils.install_packages(robjects.StrVector(packnames_to_install))

class GamR(BaseEstimator, ClassifierMixin):
    def __init__(self):
        r.library("mgcv")
        
    def fit(self, x, y):
        """
        Constructs GAM model(s) to predict y from X
        
        x: 1 or 2 dimensional array of predictor values with each row being one observation
        y: 1 or 2 dimensional array of predicted values (a GAM model is constructed for each output if y is 2 dimensional)
        """
        #Input validation for standard estimators using sklearn utils
        x, y = check_X_y(x, y, accept_sparse=['csr', 'csc', 'coo'], multi_output=True)
        #Convert to R matrices
        if x.ndim==1: #If we're only looking at 1 x at a time, shape[1] will give an error for one-dimensional arrays. Sklearn input validation doesn't change that.
            rX = r.matrix(x, nrow=x.shape[0], ncol=1)
        else:
            rX = r.matrix(x, nrow=x.shape[0], ncol=x.shape[1])
        if y.ndim==1: #If we're only looking at 1 y at a time, shape[1] will give an error for one-dimensional arrays
            rY = r.matrix(y, nrow=y.shape[0], ncol=1)
        else:
            rY = r.matrix(y, nrow=y.shape[0], ncol=y.shape[1])
        #Compute models (one for each column in y)
        self.gammodels = self.computeGAM(rX, rY)
        return self
    
    def computeGAM(self, rX, ry):
        """
        Put together R code iteratively for getting GAM models for each dependent variable in ry
        Returns list of R GAM models    
        """
        #For every dependent variable, construct model and put into list
        models = list()
        r.assign("xdata", rX) #Put data in R environment for the functions to use
        r.assign("ydata", ry) #Put data in R environment for the functions to use
        r("alldataframe<-data.frame(cbind(xdata,ydata))")
        self.xcolnames = np.asarray(r("colnames(alldataframe)[1:ncol(xdata)]"))
        self.ycolnames = np.asarray(r("colnames(alldataframe)[-(1:ncol(xdata))]")) #All the cols that are after xcol are ycol
        for ycolname in self.ycolnames:
            rcode="gam(%s~" % ycolname.replace("-", ".") #replace - by . because R doesn't take - in columnnames and converts them to . instead
            for xcolname in self.xcolnames: 
                rcode+="s(%s" % xcolname.replace("-", ".") #replace - by . because R doesn't take - in columnnames and converts them to . instead
                n_unique = r("length(unique(alldataframe$%s))" % xcolname)[0] #because R starts indexes at 1
                if n_unique<10: #If there aren't enough levels, need to set k lower
                    rcode+=",k=5"
                rcode+=")+"
            rcode=rcode[:-1]
            rcode+=",data=alldataframe)"
            gammodel = r(rcode)
            models.append(gammodel)
        return models
    
    def predict(self, x):
        """
        Get the predicted values for the given input values
        Returns an nxm np.array with the predicted y values corresponding to the given x, with m being the number of dependent variables and n the number of observations in x
        
        NOTE: assumes that the dimensions for the predicted y are the same as what was expected from the training, e.g. same amount of dependent variables
        """
        ## Check the estimator has been fit before calling this function
        check_is_fitted(self, "gammodels")
        
        #input is converted to an at least 2nd numpy array by sklearn util function, this is necessary for handling 1-dimensional x inputs etc. correctly (otherwise also doesn't convert to right amount of columns and rows)
        x = check_array(x, accept_sparse=['csr', 'csc', 'coo'])
        
        #Convert to R matrices
        if x.ndim==1: #If we're only looking at 1 x at a time, shape[1] will give an error for one-dimensional arrays. Sklearn input validation doesn't change that.
            rx = r.matrix(x, nrow=x.shape[0], ncol=1)
        else:
            rx = r.matrix(x, nrow=x.shape[0], ncol=x.shape[1])
        r.assign("newxdata", rx) #Put data in R environment for the functions to use
        r("newxdataframe<-data.frame(newxdata)")
        
        #Use gammodels list to predict each dependent variable and put together in R matrix
        for i, gammodel in enumerate(self.gammodels):
            r.assign("gmodel", gammodel)
            if i==0: #array is empty
                r("predmatrix<-predict(gmodel, newxdataframe)")
            else:
                r("predmatrix<-cbind(predmatrix,predict(gmodel,newxdataframe))")
        result = np.asarray(r['predmatrix'])
        return result
    
    def score(self, x, y):
        """
        Returns r2 score given the x values and real y values using the GAM models to get the predicted values. 
        Computes r2 using r2_score in sklearn package
        """
        y_pred = self.predict(x)
        return r2_score(y,y_pred)
        