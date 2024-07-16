# Final Project --
# Initial Model Analysis

# Load package(s) & set seed ----
library(tidymodels)
library(tidyverse)
library(here)
library(yardstick)

# Handle conflicts
tidymodels_prefer()

# load model tunes
load(here("initial-model-stage/results/fit_logistic_initial.rda"))
load(here("initial-model-stage/results/fit_naive_bayes.rda"))
load(here("initial-model-stage/results/mars_tuned_initial.rda"))
load(here("initial-model-stage/results/rf_tuned_initial.rda"))
load(here("initial-model-stage/results/bt_tuned_initial.rda"))
load(here("initial-model-stage/results/nn_tuned_initial.rda"))
load(here("initial-model-stage/results/knn_tuned_initial.rda"))
load(here("initial-model-stage/results/svm_radial_tuned_initial.rda"))
load(here("initial-model-stage/results/tuned_elastic_initial.rda"))
load(here("initial-model-stage/results/baseball_model.rda"))
load(here("initial-model-stage/results/bt_tuned_res.rda"))
load(here("initial-model-stage/results/nn_tuned_res.rda"))
load(here("initial-model-stage/results/mars_tuned_res.rda"))
load(here("initial-model-stage/results/baseball_model.rda"))
load(here("data-splits/baseball_train.rda"))
# select_best to find best hyperparameters for elastic model
# select_best(tuned_elastic_1, metric = "roc_auc")
baseball_model
# extract runtime for all three models
elastic_ks_runtime <- tictoc_elastic_initial |>
  select(runtime)
mars_runtime <- tictoc_mars_initial |>
  select(runtime)
logistic_ks_runtime <- tictoc_logistic_initial |>
  select(runtime)
naive_bayes_runtime <- tictoc_naive_bayes |>
  select(runtime)
rf_runtime <- tictoc_rf_initial |>
  select(runtime)
bt_runtime <- tictoc_bt_initial |>
  select(runtime)
knn_runtime <- tictoc_knn_initial |>
  select(runtime)
nn_runtime <- tictoc_nn_initial |>
  select(runtime)
ensemble_runtime <- tictoc_ensemble |>
  select(runtime)
svm_radial_runtime <- tictoc_svm_radial_initial |>
  select(runtime)
# obtain naive bayes performance metrics
naive_bayes_metrics <- fit_naive_bayes |>
  collect_metrics() |>
  mutate(model = "Naive Bayes") |>
  filter(.metric == "roc_auc")
naive_bayes_metrics


# obtain logistic performance metrics for kitchen sink logistic
logistic_metrics <- fit_logistic_initial |>
  collect_metrics() |>
  mutate(model = "Logistic") |>
  filter(.metric == "roc_auc")
logistic_metrics

tbl_mars <- show_best(mars_tuned_initial, metric = "roc_auc") |>
  slice_max(mean) |>
  select(mean, n, std_err) |>
  mutate(model = "MARS",
         recipe = "Kitchen Sink Recipe",
         mars_runtime,
         metric = "ROC_AUC") 
tbl_svm_rad <- show_best(svm_radial_tuned_initial, metric = "roc_auc") |>
  slice_max(mean) |>
  select(mean, n, std_err) |>
  mutate(model = "SVM Radial",
         recipe = "Kitchen Sink Recipe",
         svm_radial_runtime,
         metric = "ROC_AUC") 
tbl_btree <- show_best(btree_tuned_initial, metric = "roc_auc") |>
  slice_max(mean) |>
  select(mean, n, std_err) |>
  mutate(model = "Boosted Tree",
         recipe = "Kitchen Sink Recipe",
         bt_runtime,
         metric = "ROC_AUC") 
tbl_btree
tbl_rf <- show_best(rf_tuned_initial, metric = "roc_auc") |>
  slice_max(mean) |>
  select(mean, n, std_err) |>
  mutate(model = "Random Forest",
         recipe = "Kitchen Sink Recipe",
         rf_runtime,
         metric = "ROC_AUC") 

tbl_knn <- show_best(knn_tuned_initial, metric = "roc_auc") |>
  slice_max(mean) |>
  select(mean, n, std_err) |>
  mutate(model = "K-Nearest Neighbors",
         recipe = "Kitchen Sink Recipe",
         knn_runtime,
         metric = "ROC_AUC") 

tbl_nn <- show_best(nn_tuned_initial, metric = "roc_auc") |>
  slice_max(mean) |>
  select(mean, n, std_err) |>
  mutate(model = "Neural Network",
         recipe = "Kitchen Sink Recipe",
         nn_runtime,
         metric = "ROC_AUC") 
# make table!!
tbl_elastic_ks <- show_best(tuned_elastic_initial, metric = "roc_auc") |>
  slice_max(mean) |>
  select(mean, n, std_err) |>
  mutate(model = "Elastic Net",
         recipe = "Kitchen Sink Recipe",
         elastic_ks_runtime,
         metric = "ROC_AUC") 
tbl_naive_bayes <- naive_bayes_metrics |>
  select(mean, n, std_err) |>
  mutate(model = "Naive Bayes",
         recipe = "Baseline Recipe",
         naive_bayes_runtime,
         metric = "ROC_AUC") 
tbl_log_ks <- logistic_metrics |>
  select(mean, n, std_err) |>
  mutate(model = "Logistic",
         recipe = "Kitchen Sink Recipe",
         logistic_ks_runtime,
         metric = "ROC_AUC")


# without ensemble results

btree_hp <- select_best(btree_tuned_initial, metric = "roc_auc") |>
  mutate(model = "Boosted Tree")
rf_hp <- select_best(rf_tuned_initial, metric = "roc_auc") |>
  mutate(model = "Random Forest")
nn_hp <- select_best(nn_tuned_initial, metric = "roc_auc") |>
  mutate(model = "Neural Net")

best_3_hyperparameters <- bind_rows(btree_hp, rf_hp, nn_hp)
hyperparameters <- best_3_hyperparameters 
save(hyperparameters, file = here("initial-model-stage/results/hyperparameters.rda"))
# ensemble results

# autoplots
fig_bt_autoplot <- autoplot(btree_tuned_initial) 
ggsave(filename = "figures/fig_bt_autoplot.png", fig_bt_autoplot, width = 6, height = 4)
fig_rf_autoplot <- autoplot(rf_tuned_initial)
ggsave(filename = "figures/fig_rf_autoplot.png", fig_rf_autoplot, width = 6, height = 4)
fig_nn_autoplot <- autoplot(nn_tuned_initial)
ggsave(filename = "figures/fig_nn_autoplot.png", fig_nn_autoplot, width = 6, height = 4)

# ensemble stuff
load(here("initial-model-stage/results/nn_tuned_res.rda"))
load(here("initial-model-stage/results/mars_tuned_res.rda"))
load(here("initial-model-stage/results/bt_tuned_res.rda"))
load(here("initial-model-stage/results/baseball_model.rda"))

btree_tuned_res
bt_res_runtime <- tictoc_bt_res |>
  select(runtime)
nn_res_runtime <- tictoc_nn_res |>
  select(runtime)
mars_res_runtime <- tictoc_mars_res |>
  select(runtime)

ensemble_runtime <- tictoc_ensemble |>
  select(runtime)

total_ensemble_time <- ensemble_runtime + bt_res_runtime + nn_res_runtime + mars_res_runtime
total_ensemble_time
# train ensemble on whole training data
baseball_en_pred <- baseball_train |>
  select(is_hit) |>
  bind_cols(predict(baseball_model, baseball_train))
baseball_en_pred
member_pred <- baseball_train |>
  select(is_hit) |>
  bind_cols(predict(baseball_model, baseball_train, members = TRUE))
member_pred

pred_class_ensem <- predict(baseball_model, baseball_train, type = "class")
pred_prob_ensem <- predict(baseball_model, baseball_train, type = "prob")

prob_result <- baseball_train |>
  select(is_hit) |>
  bind_cols(pred_class_ensem, pred_prob_ensem)



tbl_en <- roc_auc(prob_result, is_hit, .pred_0)|>
  mutate(model = "Ensemble Model",
         .metric = "ROC AUC",
         total_ensemble_time)
tbl_en
save(tbl_en, file = here("figures/ensemble_metric.rda"))
results_table <- bind_rows(tbl_naive_bayes, tbl_log_ks, tbl_elastic_ks, tbl_mars,
                           tbl_svm_rad, tbl_btree, tbl_rf, tbl_knn, tbl_nn)
results_table
save(results_table, file = here("figures/results_table.rda"))
