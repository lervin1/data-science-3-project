# Final Project 3 ---- Recipes

# Kitchen Sink Recipes

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

load(here('data-splits/baseball_split.rda'))
load(here('data-splits/baseball_train.rda'))
load(here('refined-model-stage/results/var_select_lasso_2.rda'))

baseball_lasso |> skimr::skim_without_charts()

# lasso variable selection tree based recipe
selected_lasso <- var_select_lasso_2 |>
  tidy() |>
  filter(estimate != 0) |> # get rid of unimportant variables
  pull(term) 


############################
# determine numeric vs categorical
numeric_vars <- baseball_train |>
  select(where(is.numeric)) |>
  colnames()

factor_vars <- baseball_train |>
  select(where(is.factor)) |>
  colnames()

# which numeric were chosen from lasso selection
imp_numeric <- selected_lasso[selected_lasso %in% numeric_vars]

# which categorical need to go in the recipe
num_true <- map(factor_vars,
                ~ startsWith(selected_lasso, prefix = .x) |>
                  sum())

# assign raw names from dataset
names(num_true) <- factor_vars

# imp_factors <- c("source", "destination", "product_id", "name")

# if at least one factor level was important, let's keep it!
imp_factors <- enframe(unlist(num_true)) |>
  filter(value != 0) |>
  pull(name)

var_keep <- c(imp_numeric, imp_factors)

var_remove <- setdiff(names(baseball_train),
                      c(var_keep, 'is_hit', 'pitch_number_appearance'))


# reduce dataframe
baseball_lasso <- baseball_train |>
  select(all_of(var_keep), is_hit, pitch_number_appearance)

recipe_refined_tree <- recipe(is_hit ~ ., data = baseball_lasso) |> 
  step_impute_mode(all_nominal_predictors()) |> 
  step_impute_mean(all_numeric_predictors())|> 
  step_mutate(pitch_number_appearance = ifelse(pitch_number_appearance > 60, "high", "low") |> factor()) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |> 
  step_novel(all_nominal_predictors()) |>
  step_other(all_nominal_predictors()) |>
  step_zv(all_predictors()) |> 
  step_corr(all_predictors(),threshold = 0.8) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_lincomb()

baked_4 <- prep(recipe_refined_tree) |> 
  bake(new_data = NULL)
save(recipe_refined_tree, file = here('recipes/recipe_refined_tree.rda'))
