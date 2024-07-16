# Initial Model Results

The results subdirectory holds all of the results from tuning/fitting all of our models. This allows us to easily compare our models and see which ones we will utilize for our refined model stages.

## Directory Content
- `var_select_lasso.rda`: RDA file containing the variables selected using Lasso regression
- `tuned_elastic_initial_3.rda`: RDA file containing the tuned Elastic Net model
- `svm_radial_tuned_initial.rda`: RDA file containing the tuned SVM model with radial kernel
- `rf_tuned_initial.rda`: RDA file containing the tuned random forest model
- `nn_tuned_initial_2.rda`: RDA file containing the tuned neural network model
- `mars_tuned_res.rda`: RDA file containing the tuned MARS model with residuals
- `mars_tuned_initial_2.rda`: RDA file containing the tuned MARS model
- `knn_tuned_initial_3.rda`: RDA file containing the tuned k-Nearest Neighbors model
- `hyperparameters.rda`: RDA file containing the hyperparameters used in model tuning
- `fit_naive_bayes_3.rda`: RDA file containing the fitted Naive Bayes model
- `fit_logistic_initial_3.rda`: RDA file containing the fitted logistic regression model
- `bt_tuned_initial.rda`: RDA file containing the tuned boosted tree model