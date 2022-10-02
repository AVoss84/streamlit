from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from numpy.random import multinomial, dirichlet
import matplotlib.pyplot as plt


class model_value:

    model_vs_dgp = pd.DataFrame()

    def __init__(self, 
                 reward : float = 1, penalty : float = 1, 
                 avg_tp_gain : float = 600, avg_fp_gain : float = 0, 
                 avg_tn_gain : float = 0, avg_fn_gain : float = -600, 
                 verbose : bool = True):

        self.reward = reward
        self.penalty = penalty
        self.verbose = verbose
        self.revenue = {'TP' : avg_tp_gain, 'TN': avg_tn_gain, 'FP': avg_fp_gain, 'FN': avg_fn_gain}

    def __repr__(self):
       return f'<<Model value simulator>>'

    def calc_avg_gain(self, y_pred: np.array, y_true: np.array, amount_variable: np.array = None)-> np.array:
        """
        Calculate average revenues/gains associated with confusion matrix cells. Use model predictions, 
        ground truth and amount (yield) variable (measured in currency) 
        """
        if (amount_variable is not None) and self.verbose : print('Updating average gains based on model input.\n')
        assert (len(y_train) == len(y_true)) & (len(y_pred) == len(y_true)), 'Input vectors do not have same length!'
        self.model_vs_dgp['Model'], self.model_vs_dgp['DGP'] = y_pred, y_true

        self.TN, self.FP, self.FN, self.TP = confusion_matrix(y_true, y_pred).ravel()

        # True Positives:
        tp_ind = (self.model_vs_dgp['Model'].values == 1) & (self.model_vs_dgp['DGP'].values == 1)

        # True Negatives:
        tn_ind = (self.model_vs_dgp['Model'].values == 0) & (self.model_vs_dgp['DGP'].values == 0)

        # False Negatives:
        fn_ind = (self.model_vs_dgp['Model'].values == 0) & (self.model_vs_dgp['DGP'].values == 1)

        # False Positives:
        fp_ind = (self.model_vs_dgp['Model'].values == 1) & (self.model_vs_dgp['DGP'].values == 0)

        avg_tp_gain = 0
        if (amount_variable is not None) and (np.sum(tp_ind) > 0):
          # True Positive: average associated money amount, e.g. subro amount, plus reward due to time saving  
          avg_tp_gain = np.nanmean(amount_variable[tp_ind])    # positive revenue
          if self.verbose : print(avg_tp_gain)
            
        avg_tp_gain += self.TP * self.reward
        if self.verbose : print(avg_tp_gain)
   
        avg_fp_gain = 0
        if (amount_variable is not None) and (np.sum(fp_ind) > 0):
          avg_fp_gain = np.nanmean(amount_variable[fp_ind])    # no gain
          if self.verbose : print(avg_fp_gain)

        avg_fp_gain -= self.FP * self.penalty   # false alarm
        if self.verbose : print(avg_fp_gain)

        avg_tn_gain = 0
        if (amount_variable is not None) and (np.sum(tn_ind) > 0):  
          avg_tn_gain = np.nanmean(amount_variable[tn_ind])   
          if self.verbose : print(avg_tn_gain)
          
        avg_tn_gain += self.TN * self.reward    # no gain, but less work due to straight through processing
        if self.verbose : print(avg_tn_gain)

        avg_fn_gain = 0
        if (amount_variable is not None) and (np.sum(fn_ind) > 0):    
          avg_fn_gain = -np.nanmean(amount_variable[fn_ind])    # missed opportunity / revenue
          if self.verbose : print(avg_fn_gain)
            
        avg_fn_gain -= self.FN * self.penalty
        if self.verbose : print(avg_fn_gain)
        self.revenue = {'TP' : avg_tp_gain, 'TN': avg_tn_gain, 'FP': avg_fp_gain, 'FN': avg_fn_gain}
        return self.revenue

    def posterior_inference(self, prior_para = np.array([2,2,2,2]), MCsim : int = 10**4, seed : int = 123)-> pd.DataFrame:
        """
        Draw from Dirichlet posterior density of confusion matrix cell probabilities  
        """                   
        if seed : np.random.seed(seed)
        suff_statistics = np.array([self.TN, self.FP, self.FN, self.TP])            # sufficient statistics        
        self.posterior_samples = pd.DataFrame(dirichlet(prior_para + suff_statistics, size = MCsim), columns=['TN', 'FP', 'FN', 'TP'])
        return self.posterior_samples
    
    def plot_posteriors(self):
      """
      Plot posterior of cell probablities of confusion matrix (only for binary classification)
      """
      assert hasattr(self, 'posterior_samples'), 'Call posterior_inference() first.'
      import matplotlib.pyplot as plt
      for col in self.posterior_samples.columns:
          ax = plt.hist(self.posterior_samples[col], bins=50, label=col, histtype='stepfilled') 
      plt.xlabel('probability') ; plt.ylabel('Density')
      plt.title('Posterior distributions of probabilties of TN/FP/FN/TP')
      plt.legend();
    
    def expected_revenue(self)-> np.array:
      """
      Expected Bayes (Model) Revenue, i.e. posterior weighted.
      """ 
      assert hasattr(self, 'posterior_samples'), 'Call posterior_inference() first.'
      self.expectation = (self.revenue['TP']*self.posterior_samples['TP'] + self.revenue['TN']*self.posterior_samples['TN'] + self.revenue['FP']*self.posterior_samples['FP'] + self.revenue['FN']*self.posterior_samples['FN']).values
      return self.expectation

    def plot_revenue(self):
      """
      Plot posterior of exptected revenue
      """
      if not hasattr(self, 'expectation'): self.expected_revenue()
      import matplotlib.pyplot as plt
      plt.hist(self.expectation, histtype='stepfilled', bins=50, label='Expected revenue')
      plt.xticks(rotation=35) ; plt.xlabel('Euro') ; plt.ylabel('Density')
      plt.title('Posterior distributions of the expected revenue')
      plt.legend();

