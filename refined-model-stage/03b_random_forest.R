# Define and fit rf model with refined tree rec

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
set.seed(572867)

# load training data (folds)
load(here("data-splits/baseball_folds_refined.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/recipe_refined_tree.rda"))

# use parallel processing
# num_cores <- parallel::detectCores(logical = TRUE)
# doMC::registerDoMC(cores = num_cores)

num_cores <- parallel::detectCores(logical = TRUE)
cl <- makePSOCKcluster(num_cores - 1)

# set seed
set.seed(572867)

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
  add_recipe(recipe_refined_tree)

# hyperparameter tuning values ----
rf_params <- hardhat::extract_parameter_set_dials(rf_mod) |>
  update(mtry = mtry(c(11, 31)),
         trees = trees(c(500, 800)),
         min_n = min_n(c(2, 22)))


# build tuning grid
rf_grid <- grid_regular(rf_params, levels = c(3, 2, 3))

# fit workflows/models ----
# set seed
set.seed(572867)

tic.clearlog() # clear log
tic("RF Attempt Refined Model Stage")
# starting parallel processing
registerDoParallel(cl)

rf_tuned_refined <- 
  rf_wflow |>  
  tune_grid(
    baseball_folds_refined, 
    grid = rf_grid, 
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(roc_auc)
  )
# stopping cluster
stopCluster(cl)

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_rf_refined<- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows) ----
save(rf_tuned_refined, tictoc_rf_refined, file = here("refined-model-stage/results/rf_tuned_refined.rda"))