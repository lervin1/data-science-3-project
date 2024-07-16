# Final Project ----
# Define and fit elastic net model


# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(tictoc)
library(doParallel)

# handle common conflicts
tidymodels_prefer()

# load training data (folds)
load(here("data-splits/baseball_folds.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/recipe_ks.rda"))

# use parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doMC::registerDoMC(cores = num_cores)

# num_cores <- parallel::detectCores(logical = TRUE)
# registerDoParallel(cores = num_cores - 1)

# model specifications ----
elastic_spec <- 
  logistic_reg(penalty = tune(),
               mixture = tune()) |> 
  set_engine("glmnet") |> 
  set_mode("classification") 

# define workflows ----
elastic_wflow <- workflow() |>
  add_model(elastic_spec) |>
  add_recipe(recipe_ks)

# hyperparameter tuning values ----
# check ranges for hyperparameters
hardhat::extract_parameter_set_dials(elastic_spec)

# no need to update default
elastic_params <- parameters(elastic_spec)

# build tuning grid
elastic_grid <- grid_regular(elastic_params, levels = 5)
elastic_grid
# fit workflows/models ----
# set seed
set.seed(20243012)

tic.clearlog() # clear log
tic("Elastic KS") # start clock

tuned_elastic_initial <- 
  elastic_wflow |> 
  tune_grid(
    baseball_folds, 
    grid = elastic_grid, 
    control = control_grid(save_workflow = TRUE)
  )
toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_elastic_initial <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)
# write out results (fitted/trained workflows) ----
save(tuned_elastic_initial, tictoc_elastic_initial, file = here("initial-model-stage/results/tuned_elastic_initial.rda"))