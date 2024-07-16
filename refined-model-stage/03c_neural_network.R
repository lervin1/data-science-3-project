# Single layer neural net tuning with kitchen sink recipe ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(here)
library(doParallel)

# Handle conflicts
tidymodels_prefer()

# parallel processing ----
# num_cores <- parallel::detectCores(logical = TRUE)
# doMC::registerDoMC(cores = num_cores)

num_cores <- parallel::detectCores(logical = TRUE)
cl <- makePSOCKcluster(num_cores)

# load resamples ----
load(here("data-splits/baseball_folds_refined.rda"))

# load preprocessing/recipe ----
load(here("recipes/recipe_refined_tree.rda"))

set.seed(572867)
# model specifications ----
nn_model <- mlp(
  mode = "classification", 
  hidden_units = tune(),
  penalty = tune()) |>
  set_engine("nnet")

# define workflow ----
nn_wflow <- workflow() |>
  add_model(nn_model) |>
  add_recipe(recipe_refined_tree)


# hyperparameter tuning values ----
nn_param <- hardhat::extract_parameter_set_dials(nn_model) |> 
  update(hidden_units = hidden_units(c(6, 10)))

nn_grid <- grid_regular(nn_param, levels = 5)


# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("NN Attempt Refined Model Stage") # start clock
# starting parallel processing
registerDoParallel(cl)

# tuning code in here
nn_tuned_refined <- nn_wflow |>
  tune_grid(
    resamples = baseball_folds_refined,
    grid = nn_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(roc_auc)
  )
# stopping cluster
stopCluster(cl)

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_nn_refined <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(nn_tuned_refined, tictoc_nn_refined,
     file = here("refined-model-stage/results/nn_tuned_refined.rda"))