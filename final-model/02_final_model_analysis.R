# Final Project
# Assess final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load testing data and final fit
load(here("data-splits/baseball_test.rda"))
load(here("final-model/results/final_fit.rda"))

pred_class_bt <- predict(final_fit, baseball_test, type = "class")
pred_prob_bt <- predict(final_fit, baseball_test, type = "prob")

prob_result <- baseball_test |>
  select(is_hit) |>
  bind_cols(pred_class_bt, pred_prob_bt)

prob_result


# create ROC curve (task 11)
bt_roc_curve <- roc_curve(prob_result, is_hit, .pred_0)
autoplot(bt_roc_curve)
ggsave("figures/bt_roc_curve.png")

# calc area under curve with roc_auc
roc_auc_final <- roc_auc(prob_result, is_hit, .pred_0)|>
  mutate(.metric = "ROC AUC")

# use accuracy metric on rf
accuracy_final <- accuracy(prob_result, is_hit, .pred_class) |>
  mutate(.metric = "Accuracy")

# confusion matrix
conf_matrix <- conf_mat(prob_result, is_hit, .pred_class) |> 
  autoplot(type = "heatmap")

# save
save(conf_matrix, file = here('figures/conf_matrix.rda'))

# specificity
spec_final <- spec(prob_result, is_hit, .pred_class) |>
  mutate(.metric = "Specificity")
# sensitivity
sens_final <- sens(prob_result, is_hit, .pred_class) |>
  mutate(.metric = "Sensitivity")
# ppv
ppv_final <- ppv(prob_result, is_hit, .pred_class) |>
  mutate(.metric = "Positive Predictive Value")
# npv
npv_final <- npv(prob_result, is_hit, .pred_class) |>
  mutate(.metric = "Negative Predictive Value")

final_metrics <- bind_rows(roc_auc_final, accuracy_final, spec_final, sens_final, ppv_final, npv_final)
final_metrics
save(final_metrics, file = here("final-model/results/final_metrics.rda"))