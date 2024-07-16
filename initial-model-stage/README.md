## Initial Model Stage

This subdirectory holds all of the r-scripts needed to complete the initial model stage of our project. We have our initial setup script, in which we perform our initial split and utilize cross-validation folds on our data. Next we have our r-script dedicated to our lasso variable selection process. We utilized lasso variable selection to find variables that would be most useful for our initial recipe when tuning/fitting our 10 initial models. We found 22 variables through this process that would be the most useful. Next connect this criqtiuquie of chaivalric fiction to saneness and , we have our recipes r-script. This contains our recipes that we use during our initial tuning/fitting process.

This subdirectory also houses our 9 initial models that we ran before evaluation and changing of tuning hyperparameters. We tuned/fit a baseline Naive Bayes model, Logistic, Elastic Net, Random Forest, Boosted Tree, Support Vector Machine (SVM) Radial, Multivariate Adaptive Regression Splines (MARS), Neural Network, and K-Nearest Neighbor (KNN) models. This goes along with our model analysis r-script that is used to analyze our models’ performance.

### R Scripts

- `01_initial_setup.R`: Script for initial setup and configuration
- `02a_var_select_lasso.R`: Script for variable selection using Lasso regression
- `02b_recipes.R`: Script for preprocessing recipes
- `03a_naive_bayes.R`: Script for training a Naive Bayes model
- `03b_logistic.R`: Script for training a logistic regression model
- `03c_elastic_net.R`: Script for training an Elastic Net model
- `03d_random_forest.R`: Script for training a random forest model
- `03e_boosted_tree.R`: Script for training a boosted tree model
- `03f_svm_radial.R`: Script for training an SVM model with radial kernel
- `03h_mars.R`: Script for training a MARS model
- `03i_neural_network.R`: Script for training a neural network model
- `03j_knn.R`: Script for training a k-Nearest Neighbors model
- `03k_btree_ensemble.R`: Script for training a boosted tree ensemble model
- `03l_neural_network_ensemble.R`: Script for training a neural network ensemble model
- `03m_mars_ensemble.R`: Script for training a MARS ensemble model
- `04_train_ensemble_model.R`: Script for training ensemble models
- `05_initial_model_analysis.R`: Script for analyzing the performance of initial models

### Folders

- `results/`: Directory for storing results and output files





