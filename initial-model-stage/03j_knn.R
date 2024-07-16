# Define and fit k-nearest neighbors model with kitchen sink recipe

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

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
knn_mod <- nearest_neighbor(neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("classification")

# define workflows ----
knn_wflow <-  workflow() |>
  add_model(knn_mod) |>
  add_recipe(recipe_ks_tree)

# hyperparameter tuning values ----
# check ranges for hyperparameters
knn_params <- hardhat::extract_parameter_set_dials(knn_mod)

# build tuning grid
knn_grid <- grid_regular(knn_params, levels = 5)

# fit workflows/models ----
# set seed
set.seed(20243012)
tic.clearlog() # clear log
tic("KNN Attempt Initial Model Stage")
knn_tuned_initial <- 
  knn_wflow |> 
  tune_grid(
    baseball_folds, 
    grid = knn_grid, 
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(roc_auc)
  )
toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_knn_initial <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows) ----
save(knn_tuned_initial, tictoc_knn_initial, file = here("initial-model-stage/results/knn_tuned_initial.rda"))
