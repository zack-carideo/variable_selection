### Variable Selection E2E Pipeline For use in identifying variables for  Logistic Regression 

#### 2 Key Files 

1. variable_selection_pipeline.ipynb — End-to-end Jupyter notebook implementing:

- Synthetic fair lending data generation (credit scores, DTI, LTV, delinquencies, etc.)
- Assumption pre-screening (EPV, separation checks)
- Filter methods (Information Value, Mutual Information, correlation analysis)
- VIF-based multicollinearity elimination
- Nested CV with 7 ensemble methods (LASSO, Elastic Net, RFE, Forward Selection, MI, Tree Importance, IV)
- Assumption validation on final features (Box-Tidwell linearity test, VIF check)
- Final model diagnostics with coefficient summary

2. variable_selection_utils.py — Reusable utilities module containing:

- generate_fair_lending_data() — Creates realistic synthetic lending data
- NestedCVVariableSelector — Core nested CV framework with stability metrics
- Assumption validators (calculate_epv, calculate_vif, box_tidwell_test, check_separation)
- Filter methods (calculate_all_iv, calculate_mutual_information, correlation_filter)
- Selection methods (all 7 methods with consistent CV/random_state interfaces)
- Visualization utilities (plot_stability_across_folds, plot_cv_performance)



### Claude Starter Prompt 
```text 
Create a project plan to perform variable selection for a logistic regression. The goal is to perform variable selection across an ensemble of methods to identify optimal variables. Don’t forget to consider statistical  assumptions associated with a binary logistic regression that each selected variable must satisfy. We have 2 weeks to complete the variable selection process.

- account for variable types and how each  should be evaluated (continous vs categorical)
- when creating nested CV splits, each should contain data across all time periods (aka for each cv, from each year in data, allocate 80% to training 20% to validation)
- after checking basic data assumptions, all variables should be converted into WoE bins , with 

Using the code / process outlined in the first prompt Create a dummy dataset that represents fair lending data attributes, and create an e2e variable selection .ipynb that implements the full e2e variable selection process described above. make sure to utilize the nested CV structure. Test your code before returning results to ensure it runs without errors, and that the results are logically sound.   Make sure you think carefully about each step. return a working .ipynb that implements the variables selection process, and a .py file with core utilities leveraged throughout the variable selection process. the utilities should be imported in the .ipynb to keep the .ipynb focused on the e2e variable selection process. 
```