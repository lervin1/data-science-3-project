# initial read in of the data

library(tidyverse)
library(here)
library(tidymodels)
library(naniar)

full_data <- read_csv(here("data/savant_pitch_level.csv")) |>
 janitor::clean_names()
  
inplay_data <- full_data |> 
  mutate(year_2023 = lubridate::year(game_date) == 2023) |> 
  filter(year_2023) |> 
  filter(events != 'NA') |> 
  filter(events != 'walk' & events != 'strikeout')

baseball_data <- initial_split(inplay_data, prop = 0.33)
baseball_dataset <- training(baseball_data)

save(baseball_dataset, file = here('data/baseball_dataset.rda'))


load(here('data/baseball_dataset.rda'))

baseball_dataset <- baseball_dataset |> 
  filter(events != 'catcher_interf') |> 
  filter(events != 'caught_stealing_2b') |> 
  filter(events != 'caught_stealing_3b') |> 
  filter(events != 'caught_stealing_home') |> 
  mutate(is_hit = ifelse(events == 'single' | events == 'double' | events == 'triple'
                         | events == 'home_run', 1, 0),
        across(where(is.character), as.factor),
        is_hit = factor(is_hit))

save(baseball_dataset, file = here('data/baseball_dataset.rda'))





