# Define and fit rf model with kitchen sink tree

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(hardhat)
library(tictoc)
library(doParallel)

# handle common conflicts
tidymodels_prefer()

# set seed
set.seed(20243012)

# load training data (folds)
load(here("data-splits/baseball_folds.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/recipe_ks_tree.rda"))

# use parallel processing
# num_cores <- parallel::detectCores(logical = TRUE)
# doMC::registerDoMC(cores = num_cores)

num_cores <- parallel::detectCores(logical = TRUE)
cl <- makePSOCKcluster(num_cores - 1)

# set seed
set.seed(20243012)

# model specifications ----
rf_mod <-
  rand_forest(trees = tune(), 
              min_n = tune(),
              mtry = tune()) |>
  set_engine("ranger") |>
  set_mode("classification")

# define workflows ----
rf_wflow <- workflow() |>
  add_model(rf_mod) |>
  add_recipe(recipe_ks_tree)

# hyperparameter tuning values ----
rf_params <- hardhat::extract_parameter_set_dials(rf_mod) |>
  update(mtry = mtry(c(1, 30)),
         trees = trees(c(250, 750)))


# build tuning grid
rf_grid <- grid_regular(rf_params, levels = c(5, 2, 4))

# fit workflows/models ----
# set seed
set.seed(20243012)

tic.clearlog() # clear log
tic("RF Attempt Initial Model Stage")
# starting parallel processing
registerDoParallel(cl)

rf_tuned_initial <- 
  rf_wflow |>  
  tune_grid(
    baseball_folds, 
    grid = rf_grid, 
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(roc_auc)
  )
# stopping cluster
stopCluster(cl)

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_rf_initial<- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows) ----
save(rf_tuned_initial, tictoc_rf_initial, file = here("initial-model-stage/results/rf_tuned_initial.rda"))