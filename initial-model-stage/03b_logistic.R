# Final Project ----
# Define and fit logistic model


# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(tictoc)
#library(doParallel)

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
logistic_spec <- 
  logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification") 

# define workflows ----
logistic_wflow <- workflow() |>
  add_model(logistic_spec) |>
  add_recipe(recipe_ks)

# fit workflows/models ----
tic.clearlog() # clear log
tic("Logistic Initial Model Stage") # start clock

fit_logistic_initial <- logistic_wflow |>
  fit_resamples(resamples = baseball_folds,
                control = control_resamples(save_workflow = TRUE))
toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_logistic_initial <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)
# write out results (fitted/trained workflows) ----
save(fit_logistic_initial, tictoc_logistic_initial, file = here("initial-model-stage/results/fit_logistic_initial.rda"))