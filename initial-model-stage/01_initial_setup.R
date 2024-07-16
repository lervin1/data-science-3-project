# Final Project 3 ----
# Initial data checks, data splitting, & data folding

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# reading in data ----
load(here("data/baseball_dataset_2.rda"))

# baseball_dataset <- baseball_dataset |>
#   select(!des & !game_type & !player_name & !description & !game_year & !inning_topbot & !fielder_2 &!sv_id
#          &!game_pk & !pitcher_1 & !fielder_2_1 & !fielder_3 & !fielder_4 & !fielder_5 & !fielder_6 & !fielder_7
#          & !fielder_8 & !fielder_9 & !woba_denom & !at_bat_number & !home_score & !away_score & !post_away_score & !post_home_score
#          & !if_fielding_alignment & !of_fielding_alignment & !sp_indicator & !rp_indicator & !year_2023  & !bb_type
#          & !game_date & !batter  & !pitcher  & !home_team & !away_team &!delta_home_win_exp &!delta_run_exp &!post_bat_score
#          &!post_fld_score &!delta_home_win_exp &!estimated_ba_using_speedangle &!estimated_woba_using_speedangle &!woba_value
#          &!babip_value &!iso_value)
# 
# baseball_dataset <- baseball_dataset |>
#   mutate(pitch_type = factor(pitch_type),
#          events = factor(events),
#          stand = factor(stand),
#          p_throws = factor(p_throws),
#          type = factor(type),
#          balls = factor(balls, ordered = TRUE),
#          strikes = factor(strikes, ordered = TRUE),
#          outs_when_up = factor(outs_when_up, ordered = TRUE),
#          inning = factor(inning, ordered = TRUE),
#          role_key = factor(role_key),
#          times_faced = factor(times_faced, ordered = TRUE)) |>
#   filter(!is.na(release_speed)) |>
#   filter(!is.na(az)) |>
#   janitor::clean_names()
# 
# skimr::skim(baseball_dataset)
# 
# save(baseball_dataset, file = here('data/baseball_dataset_2.rda'))
# 

# setting a seed ----

set.seed(1234)

# initial split of data
baseball_split <- baseball_dataset |>
  initial_split(prop = 0.8, strata = is_hit)

# verify each resulting dataframe has the correct number of columns and rows
baseball_train <- baseball_split |> training()
baseball_test <- baseball_split |> testing()

# cross validation
baseball_folds <- vfold_cv(baseball_train, v = 8, repeats = 3,
                        strata = is_hit)

# save out files ----
save(baseball_split, file = here("data-splits/baseball_split.rda"))
save(baseball_train, file = here("data-splits/baseball_train.rda"))
save(baseball_test, file = here("data-splits/baseball_test.rda"))
save(baseball_folds, file = here("data-splits/baseball_folds.rda"))
