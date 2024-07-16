# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(hardhat)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# set seed
set.seed(20243012)

# load training data (folds)
load(here("data-splits/baseball_folds.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/recipe_ks_tree.rda"))

# use parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doMC::registerDoMC(cores = num_cores)
# using 12 cores

# set seed
set.seed(20243012)

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
  add_recipe(recipe_ks_tree)

# hyperparameter tuning values ----
# check ranges for hyperparameters
hardhat::extract_parameter_set_dials(btree_mod)

# change hyperparameter ranges
btree_params <- parameters(btree_mod) |>
  update(mtry = mtry(c(11,41)),
         trees = trees(c(1000, 2000)),
         min_n = min_n(c(2, 16)),
         learn_rate = learn_rate(c(-4, -1)))
      

btree_params
# build tuning grid
btree_grid <- grid_regular(btree_params, levels = c(4, 3, 2, 4))
btree_grid
# fit workflows/models ----
# set seed
set.seed(20243012)

tic.clearlog() # clear log
tic("BTree ensemble member")
btree_tuned_res <- 
  btree_wflow |> 
  tune_grid(
    baseball_folds, 
    grid = btree_grid, 
    control = stacks::control_stack_grid(),
    metrics = metric_set(roc_auc)
  )
toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_bt_res <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows) ----
save(btree_tuned_res, tictoc_bt_res, file = here("initial-model-stage/results/bt_tuned_res.rda"))
