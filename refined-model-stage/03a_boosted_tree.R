# Define and fit btree model on kitchen sink tree based recipe

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(hardhat)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# set seed
set.seed(572867)

# load training data (folds)
load(here("data-splits/baseball_folds_refined.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/recipe_refined_tree.rda"))

# use parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doMC::registerDoMC(cores = num_cores)
# using 12 cores

# set seed
set.seed(572867)

# model specifications ----
btree_mod <- boost_tree(
  trees = tune(),
  mtry = tune(),
  min_n = tune(),
  learn_rate = tune()) |>
  set_engine("xgboost") |>
  set_mode("classification")

# define workflows ----
btree_wflow <- workflow() |>
  add_model(btree_mod) |>
  add_recipe(recipe_refined_tree)

# hyperparameter tuning values ----
# check ranges for hyperparameters
hardhat::extract_parameter_set_dials(btree_mod)

# change hyperparameter ranges
btree_params <- parameters(btree_mod) |>
  update(mtry = mtry(c(11, 31)),
         trees = trees(c(1000, 2000)),
         min_n = min_n(c(2, 22)),
         learn_rate = learn_rate(c(-4, -2)))

btree_params
# build tuning grid
btree_grid <- grid_regular(btree_params, levels = 3)
btree_grid
# fit workflows/models ----
# set seed
set.seed(572867)

tic.clearlog() # clear log
tic("BTree Attempt Refined Model Stage")
btree_tuned_refined <- 
  btree_wflow |> 
  tune_grid(
    baseball_folds_refined, 
    grid = btree_grid, 
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(roc_auc)
  )
toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_bt_refined <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows) ----
save(btree_tuned_refined, tictoc_bt_refined, file = here("refined-model-stage/results/bt_tuned_refined.rda"))
