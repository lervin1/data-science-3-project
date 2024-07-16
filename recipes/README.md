## Recipes

This subdirectory holds the recipes utilized to fit/tune all of our models. We have 4 recipes. The first two recipes are associated with our initial model stages, and are kitchen sink recipes. One recipe is dedicated to tree-based models and one recipe is not. The next 2 recipes are associated with our refined model stages. They contain 22 variables that were selected through lasso variable selection. Again, one recipe is dedicated to tree-based models and one recipe is not. 

## Directory Contents

- `recipe_refined_tree.rda`: RDA file containing the preprocessing recipe for the refined tree model
- `recipe_nb_3.rda`: RDA file containing the preprocessing recipe for the Naive Bayes model
- `recipe_ks_tree_3.rda`: RDA file containing the preprocessing recipe for all tree models
- `recipe_ks_3.rda`: RDA file containing the preprocessing recipe for all non tree models
