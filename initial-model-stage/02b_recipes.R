# Final Project 3 ---- Recipes

# Kitchen Sink Recipes

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

load(here('data-splits/baseball_split.rda'))
load(here('data-splits/baseball_train.rda'))
load(here('initial-model-stage/results/var_select_lasso.rda'))

# Kitchen Sink Recipe for Parametric

# recipe_ks_2 <- recipe(is_hit ~ ., data = baseball_train) |> 
#   step_rm(player_name, events, type, on_3b, on_2b, on_1b, inning, sz_top, sz_bot, bat_score, fld_score,
#           role_key) |> 
#   step_impute_knn(all_nominal_predictors()) |> 
#   step_impute_linear(all_numeric_predictors(), impute_with = imp_vars(pitch_type, stand, p_throws, balls, strikes,
#                                                                       outs_when_up,
#                                                                       times_faced,release_speed, release_pos_x,
#                                                                       release_pos_z, zone, pfx_x, pfx_z, plate_x, plate_z,vx0, vy0, vz0, ax, ay, az, release_pos_y,
#                                                                       pitch_number,pitch_number_appearance,
#                                                                       pitcher_at_bat_number))|> 
#   step_dummy(all_nominal_predictors()) |> 
#   step_novel(all_nominal_predictors()) |>
#   step_other(all_nominal_predictors()) |>
#   step_zv(all_predictors()) |> 
#   step_normalize(all_numeric_predictors()) |> 
#   step_lincomb()
# 
# baked <- prep(recipe_ks_2) |> 
#   bake(new_data = NULL)
# 
# save(recipe_ks_2, file = here('recipes/recipe_ks_2.rda'))
# 
# 
# # Kitchen Sink Recipe for Tree
# 
# recipe_ks_tree_2 <- recipe(is_hit ~ ., data = baseball_train) |> 
#   step_rm(player_name, events, type, on_3b, on_2b, on_1b, inning, sz_top, sz_bot, bat_score, fld_score,
#           role_key) |> 
#   step_impute_knn(all_nominal_predictors()) |> 
#   step_impute_linear(all_numeric_predictors(), impute_with = imp_vars(pitch_type, stand, p_throws, balls, strikes,
#                                                                       outs_when_up,
#                                                                       times_faced,release_speed, release_pos_x,
#                                                                       release_pos_z, zone, pfx_x, pfx_z, plate_x, plate_z,vx0, vy0, vz0, ax, ay, az, release_pos_y,
#                                                                       pitch_number,pitch_number_appearance,
#                                                                       pitcher_at_bat_number))|> 
#   step_dummy(all_nominal_predictors(), one_hot = TRUE) |> 
#   step_novel(all_nominal_predictors()) |>
#   step_other(all_nominal_predictors()) |>
#   step_zv(all_predictors()) |> 
#   step_normalize(all_numeric_predictors()) |> 
#   step_lincomb()
# 
# baked2 <- prep(recipe_ks_tree_2) |> 
#   bake(new_data = NULL)
# save(recipe_ks_tree_2, file = here('recipes/recipe_ks_tree_2.rda'))
# 
# # Naive Bayes Baseline Recipe
# 
# recipe_nb_2 <- recipe(is_hit ~ ., data = baseball_train) |> 
#   step_rm(player_name, events, type, on_3b, on_2b, on_1b, inning, sz_top, sz_bot, bat_score, fld_score,
#           role_key) |> 
#   step_impute_knn(all_nominal_predictors()) |> 
#   step_impute_linear(all_numeric_predictors(), impute_with = imp_vars(pitch_type, stand, p_throws, balls, strikes,
#                                                                       outs_when_up,
#                                                                       times_faced,release_speed, release_pos_x,
#                                                                       release_pos_z, zone, pfx_x, pfx_z, plate_x, plate_z,vx0, vy0, vz0, ax, ay, az, release_pos_y,
#                                                                       pitch_number,pitch_number_appearance,
#                                                                       pitcher_at_bat_number))|> 
#   step_nzv(all_predictors()) |> 
#   step_normalize(all_numeric_predictors()) |> 
#   step_lincomb()
# 
# baked <- prep(recipe_nb_2) |> 
#   bake(new_data = NULL)
# 
# save(recipe_nb_2, file = here('recipes/recipe_nb_2.rda'))

# lasso variable selection tree based recipe
selected_lasso <- var_select_lasso |>
  tidy() |>
  filter(estimate != 0) |> # get rid of unimportant variables
  pull(term) 


############################
# determine numeric vs categorical
numeric_vars <- baseball_train |>
  select(where(is.numeric)) |>
  colnames()

factor_vars <- baseball_train |>
  select(where(is.factor)) |>
  colnames()

# which numeric were chosen from lasso selection
imp_numeric <- selected_lasso[selected_lasso %in% numeric_vars]

# which categorical need to go in the recipe
num_true <- map(factor_vars,
                ~ startsWith(selected_lasso, prefix = .x) |>
                  sum())

# assign raw names from dataset
names(num_true) <- factor_vars

# imp_factors <- c("source", "destination", "product_id", "name")

# if at least one factor level was important, let's keep it!
imp_factors <- enframe(unlist(num_true)) |>
  filter(value != 0) |>
  pull(name)

var_keep <- c(imp_numeric, imp_factors)

var_remove <- setdiff(names(baseball_train),
                      c(var_keep, 'is_hit'))

# reduce dataframe
baseball_lasso <- baseball_train |>
  select(all_of(var_keep), is_hit)

recipe_ks <- recipe(is_hit ~ ., data = baseball_lasso) |> 
  step_impute_mode(all_nominal_predictors()) |> 
  step_impute_mean(all_numeric_predictors())|> 
  step_dummy(all_nominal_predictors()) |> 
  step_novel(all_nominal_predictors()) |>
  step_other(all_nominal_predictors()) |>
  step_nzv(all_predictors()) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_lincomb()

baked_3 <- prep(recipe_ks) |> 
  bake(new_data = NULL)

save(recipe_ks, file = here('recipes/recipe_ks.rda'))

recipe_ks_tree <- recipe(is_hit ~ ., data = baseball_lasso) |> 
  step_impute_mode(all_nominal_predictors()) |> 
  step_impute_mean(all_numeric_predictors())|> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |> 
  step_novel(all_nominal_predictors()) |>
  step_other(all_nominal_predictors()) |>
  step_nzv(all_predictors()) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_lincomb()

baked_4 <- prep(recipe_ks_tree) |> 
  bake(new_data = NULL)
save(recipe_ks_tree, file = here('recipes/recipe_ks_tree.rda'))

recipe_nb <- recipe(is_hit ~ ., data = baseball_lasso) |> 
  step_impute_mode(all_nominal_predictors()) |> 
  step_impute_mean(all_numeric_predictors())|> 
  step_nzv(all_predictors()) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_lincomb()

baked <- prep(recipe_nb) |> 
  bake(new_data = NULL)

save(recipe_nb, file = here('recipes/recipe_nb.rda'))
