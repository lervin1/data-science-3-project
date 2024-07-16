# Final Project 3 ----
# Refined data folding

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# reading in data ----
load(here("data/baseball_dataset.rda"))
load(here("data-splits/baseball_train.rda"))

# set seed
set.seed(537926)

# cross validation
baseball_folds_refined <- vfold_cv(baseball_train, v = 5, repeats = 3,
                        strata = is_hit)

# save out files ----
save(baseball_folds_refined, file = here("data-splits/baseball_folds_refined.rda"))
