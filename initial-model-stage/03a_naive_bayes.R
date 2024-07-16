# Final Project ----
# Fit baseline (naive bayes) Model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(discrim)
library(tictoc)
library(doParallel)

# handle common conflicts
tidymodels_prefer()

# load folds data
load(here("data-splits/baseball_folds.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/recipe_nb.rda"))

# use parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doMC::registerDoMC(cores = num_cores)

# num_cores <- parallel::detectCores(logical = TRUE)
# registerDoParallel(cores = num_cores - 1)

# model specifications ----
naive_bayes_spec <- 
  naive_Bayes() |> 
  set_engine("klaR") |> 
  set_mode("classification") 

# define workflows ----
naive_bayes_wflow <- workflow() |>
  add_model(naive_bayes_spec) |>
  add_recipe(recipe_nb)

# fit workflows/models ----
tic.clearlog() # clear log
tic("Naive Bayes Baseline") # start clock

fit_naive_bayes <- naive_bayes_wflow |>
  fit_resamples(resamples = baseball_folds,
                control = control_resamples(save_workflow = TRUE))
toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_naive_bayes <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)
# write out results (fitted/trained workflows) ----
save(fit_naive_bayes, tictoc_naive_bayes, file = here("initial-model-stage/results/fit_naive_bayes.rda"))

fit_naive_bayes |> 
  unnest(.metrics) |> 
  filter(.metric == "roc_auc") |> 
  ggplot(aes(.estimate)) +
  geom_density() +
  geom_rug()