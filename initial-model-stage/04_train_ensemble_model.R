# Train & explore ensemble model

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)
library(tictoc)

# Handle common conflicts
tidymodels_prefer()

# Load candidate model info ----
load(here("initial-model-stage/results/nn_tuned_res.rda"))
load(here("initial-model-stage/results/bt_tuned_res.rda"))
load(here("initial-model-stage/results/mars_tuned_res.rda"))

# Create data stack ----
baseball_data_stack <- stacks() |>
  add_candidates(nn_tuned_res) |> 
  add_candidates(btree_tuned_res) |> 
  add_candidates(mars_tuned_res) 

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)
# penalty decides the amount of regularization
# high penalty uses less models than low penalty

# Blend predictions (tuning step, set seed)
set.seed(1234)
baseball_stack_blend <- baseball_data_stack |>
  blend_predictions(penalty = blend_penalty)



# Save blended model stack for reproducibility & easy reference (for report)
save(baseball_stack_blend, file = here("initial-model-stage/results/baseball_stack_blend.rda"))

# Explore the blended model stack
autoplot(baseball_stack_blend) +
  theme_minimal()
# show how many members are from each model
autoplot_blend <- autoplot(baseball_stack_blend, type = 'weights') +
  theme_minimal()
ggsave(filename = "figures/autoplot_blend.png", autoplot_blend)
# show members and optimal parameters for them
# nn members
nn_members <- baseball_stack_blend |>
  collect_parameters("nn_tuned_res") |>
  filter(coef != 0)
# bt members
bt_members <- baseball_stack_blend |>
  collect_parameters("btree_tuned_res") |>
  filter(coef != 0)
# mars members
mars_members <- baseball_stack_blend |>
  collect_parameters("mars_tuned_res") |>
  filter(coef != 0)
bt_members
# create large table 
bind_rows(nn_members, bt_members, mars_members)
# fit to training set ----
tic.clearlog() # clear log
tic("Ensemble Model") # start clock
baseball_model <- baseball_stack_blend |>
  fit_members()
toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_ensemble <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)
# Save trained ensemble model for reproducibility & easy reference (for report)
save(baseball_model, tictoc_ensemble, file = here("initial-model-stage/results/baseball_model.rda"))
save(bt_members, file = here("initial-model-stage/results/ensemble_members.rda"))