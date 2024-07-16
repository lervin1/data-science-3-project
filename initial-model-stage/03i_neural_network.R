# Single layer neural net tuning with kitchen sink recipe ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(here)

# Handle conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
doMC::registerDoMC(cores = num_cores)

# load resamples ----
load(here("data-splits/baseball_folds.rda"))

# load preprocessing/recipe ----
load(here("recipes/recipe_ks_tree.rda"))

set.seed(1234)
# model specifications ----
nn_model <- mlp(
  mode = "classification", 
  hidden_units = tune(),
  penalty = tune()) |>
  set_engine("nnet")

# define workflow ----
nn_wflow <- workflow() |>
  add_model(nn_model) |>
  add_recipe(recipe_ks_tree)


# hyperparameter tuning values ----
nn_param <- hardhat::extract_parameter_set_dials(nn_model)

nn_grid <- grid_regular(nn_param, levels = 5)


# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("NN Attempt Initial Model Stage") # start clock

# tuning code in here
nn_tuned_initial <- nn_wflow |>
  tune_grid(
    resamples = baseball_folds,
    grid = nn_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(roc_auc)
  )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_nn_initial <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(nn_tuned_initial, tictoc_nn_initial,
     file = here("initial-model-stage/results/nn_tuned_initial.rda"))