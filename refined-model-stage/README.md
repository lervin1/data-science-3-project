# Refined Models Stage

This subdirectory holds the models that we refined to try to increase our model’s performance. We changed our recipe slightly as well as redid our folds to 5 & 3. We tuned/fit additional Boosted Tree, Neural Network, and Random Forest models for model refinement. We assessed the models in the model analysis r-script and found that our boosted tree model was the best, but slightly worse than our initial boosted tree model.

### R Scripts

- `01_initial_setup.R`: Script for initial setup and configuration
- `02a_var_select_lasso.R`: Script for variable selection using Lasso regression
- `02b_recipes.R`: Script for preprocessing recipes
- `03a_boosted_tree.R`: Script for training a boosted tree model
- `03b_random_forest.R`: Script for training a random forest model
- `03c_neural_network.R`: Script for training a neural network model
- `04_model_analysis.R`: Script for analyzing the model performance

### Folders

- `results/`: Directory for storing results and output files


