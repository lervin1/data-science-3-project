# Final Project
# Train final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts
tidymodels_prefer()

# use parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doMC::registerDoMC(cores = num_cores)

# load training data and rf tuned
load(here("initial-model-stage/results/bt_tuned_initial.rda"))
load(here("data-splits/baseball_train.rda"))

# finalize workflow
final_wflow <- btree_tuned_initial |> 
  extract_workflow(btree_tuned_initial) |>  
  finalize_workflow(select_best(btree_tuned_initial, metric = "roc_auc"))

# train final model ----
# set seed
set.seed(20243012)
final_fit <- fit(final_wflow, baseball_train)

# save out final fit
save(final_fit, file = here("final-model/results/final_fit.rda"))
