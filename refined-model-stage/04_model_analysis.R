# Final Project ----
# Refined Model Analysis

# Load package(s) & set seed ----
library(tidymodels)
library(tidyverse)
library(here)
library(knitr)

# Handle conflicts
tidymodels_prefer()

# load model tunes
load(here("refined-model-stage/results/rf_tuned_refined.rda"))
load(here("refined-model-stage/results/bt_tuned_refined.rda"))
load(here("refined-model-stage/results/nn_tuned_refined.rda"))

# extract runtime for all three models
rf_runtime <- tictoc_rf_refined |>
  select(runtime)
bt_runtime <- tictoc_bt_refined |>
  select(runtime)
nn_runtime <- tictoc_nn_refined |>
  select(runtime)

# find best for each model ----

# boosted tree model
# looking at top 5 roc_auc values
show_best(btree_tuned_refined, metric = "roc_auc") |>
  slice_head(n = 5, by = mean)
# looking at best roc_auc value for table
tbl_btree_ref <- show_best(btree_tuned_refined, metric = "roc_auc") |>
  slice_max(mean) |>
  select(mean, n, std_err) |>
  mutate(model = "Boosted Tree",
         recipe = "Feature Engineering Recipe",
         bt_runtime,
         metric = "ROC_AUC") 

# random forest model
# looking at top 5 roc_auc values
show_best(rf_tuned_refined, metric = "roc_auc") |>
  slice_head(n = 5, by = mean)
# looking at the best roc_auc value for table
tbl_rf_ref <- show_best(rf_tuned_refined, metric = "roc_auc") |>
  slice_max(mean) |>
  select(mean, n, std_err) |>
  mutate(model = "Random Forest",
         recipe = "Feauture Engineering Recipe",
         rf_runtime,
         metric = "ROC_AUC") 

# neural networks model
# looking at the top 5 roc_auc values
show_best(nn_tuned_refined, metric = "roc_auc") |>
  slice_head(n = 5, by = mean)
# looking at the best roc_auc value for table
tbl_nn_ref <- show_best(nn_tuned_refined, metric = "roc_auc") |>
  slice_max(mean) |>
  select(mean, n, std_err) |>
  mutate(model = "Neural Network",
         recipe = "Feature Engineering Recipe",
         nn_runtime,
         metric = "ROC_AUC") 

results_table_2 <- bind_rows(tbl_btree_ref, tbl_rf_ref, tbl_nn_ref)
save(results_table_2, file = here("refined-model-stage/results/results_table_2.rda"))
refine_results_table <- results_table_2 |>
  kable()