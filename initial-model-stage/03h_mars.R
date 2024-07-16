# Tuning for MARS model with Kitchen Sink recipe

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
load(here("recipes/recipe_ks.rda"))

set.seed(123)
# model specifications ----
mars_model <- mars(
  mode = "classification",
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")

# define workflow ----
mars_wflow <- workflow() |>
  add_model(mars_model) |>
  add_recipe(recipe_ks)

# hyperparameter tuning values ----
mars_param <- hardhat::extract_parameter_set_dials(mars_model)

mars_grid <- grid_regular(mars_param, levels = 5)
mars_grid
# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("MARS Attempt Initial Model Stage") # start clock

# tuning code in here
mars_tuned_initial <- mars_wflow |>
  tune_grid(
    resamples = baseball_folds,
    grid = mars_grid, 
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(roc_auc)
  )
toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_mars_initial <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(mars_tuned_initial, tictoc_mars_initial, file = here("initial-model-stage/results/mars_tuned_initial.rda"))
