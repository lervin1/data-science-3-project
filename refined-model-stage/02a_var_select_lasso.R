# Variable selection using lasso regression

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doParallel)

# handle common conflicts
tidymodels_prefer()

# parallel processing ----
# num_cores <- parallel::detectCores(logical = TRUE)
# doMC::registerDoMC(cores = num_cores)

num_cores <- parallel::detectCores(logical = TRUE)
cl <- makePSOCKcluster(num_cores)

registerDoParallel(cl)

# create resamples/folds ----
load(here('data-splits/baseball_train.rda'))

set.seed(383628)
lasso_folds_2 <- baseball_train |> 
  vfold_cv(v = 5, repeats = 1, strata = is_hit)

# basic recipe ----
recipe_lasso_2 <- recipe(is_hit ~., baseball_train) |> 
  step_rm(player_name, events, type, on_3b, on_2b, on_1b, inning, sz_top, sz_bot, bat_score, fld_score,
        role_key) |> 
  step_impute_mode(all_nominal_predictors()) |>
  step_impute_median(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |> 
  step_nzv(all_predictors()) |> 
  step_normalize(all_predictors())

prep_check <- prep(recipe_lasso_2) |> 
  bake(new_data = NULL)

# model specifications ----
lasso_mod <- logistic_reg(mixture = 1, penalty = tune()) |> 
  set_engine('glmnet') |> 
  set_mode('classification')

# define workflows ----
lasso_wflow <- workflow() |> 
  add_model(lasso_mod) |> 
  add_recipe(recipe_lasso_2)

# hyperparameter tuning values ----
lasso_params <- extract_parameter_set_dials(lasso_mod) |> 
  update(penalty = penalty(c(-3, 0)))

# build tuning grid
lasso_grid <- grid_regular(lasso_params, levels = 5)

# fit workflow/model ----
lasso_tune <- lasso_wflow |> 
  tune_grid(
    resamples = lasso_folds_2,
    grid = lasso_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(save_workflow = TRUE)
  )

# build and interpret autoplot ----
autoplot(lasso_tune)
# minimum point is 0.001, so we will find a better penalty by searching smaller values

# extract best model (optimal tuning parameters)
#using best lasso
optimal_wflow <- extract_workflow(lasso_tune) |> 
  finalize_workflow(select_best(lasso_tune, metric = 'roc_auc'))


# fit best model/results
var_select_lasso_2 <- fit(optimal_wflow, baseball_train)

# stop parallel processing
stopCluster(cl)

# write out variable selection results ----
save(var_select_lasso_2, file = here('refined-model-stage/results/var_select_lasso_2.rda'))
