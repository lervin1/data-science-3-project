# Define and fit radial with kitchen sink recipe

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(tictoc)
library(doParallel)

# Handle conflicts
tidymodels_prefer()

# set seed
set.seed(20243013)

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
doMC::registerDoMC(cores = num_cores)

# num_cores <- parallel::detectCores(logical = TRUE)
# cl <- makePSOCKcluster(num_cores - 1)

# load resamples ----
load(here("data-splits/baseball_folds.rda"))

# load preprocessing/recipe ----
load(here("recipes/recipe_ks.rda"))

# model specifications ----
svm_radial_model <- svm_rbf(
  mode = "classification", 
  cost = tune(),
  rbf_sigma = tune()
) |>
  set_engine("kernlab")

# define workflows ----
svm_radial_wflow <- workflow() |>
  add_model(svm_radial_model) |>
  add_recipe(recipe_ks)

# hyperparameter tuning values ----
svm_radial_param <- hardhat::extract_parameter_set_dials(svm_radial_model)

svm_radial_grid <- grid_regular(svm_radial_param, levels = 3)


# fit workflow/model ----

tic.clearlog()
tic("SVM RAD Initial Model Stage") # start clock
# starting parallel processing
# registerDoParallel(cl)

# tuning code in here
svm_radial_tuned_initial <- svm_radial_wflow |>
  tune_grid(
    resamples = baseball_folds,
    grid = svm_radial_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(roc_auc)
  )

# stopping cluster
# stopCluster(cl)

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_svm_radial_initial <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(svm_radial_tuned_initial, tictoc_svm_radial_initial,
     file = here("initial-model-stage/results/svm_radial_tuned_initial.rda"))

