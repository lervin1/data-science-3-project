# Final Project ----
# Storing figures for report

## load packages ----

library(tidyverse)
library(tidymodels)
library(here)
library(naniar)
library(knitr)
library(kableExtra)

# load necessary data
load(here("data/baseball_dataset.rda"))

# figure 1 ----

fig_1 <- baseball_dataset |> 
  ggplot(aes(x = is_hit)) +
  geom_bar(color = "black", fill = "slategray2") +
  theme_bw() +
  labs(x = "Hit or Not (0 = Not Hit, 1 = Hit)",
       y = "Count",
       title = "Comparison of Hit vs Non Hit Pitches") +
  theme(text = element_text(size = 10))

# ggsave
ggsave(filename = "figures/fig_1.png", fig_1, width = 6, height = 4)

# missingness figure
missingness <- baseball_dataset |> 
  miss_var_summary() 

missingness_filtered <- missingness |> 
  filter(n_miss > 0) 

missingness_table <- kable(missingness_filtered, "html") |> 
  kable_styling(position = "center") |> 
  scroll_box(width = "100%", height = "400px")

# save table
save(missingness_table, file = here('figures/missingness_table.rda'))

fig_miss <- gg_miss_var(baseball_dataset)

# ggsave
ggsave(filename = "figures/fig_miss.png", fig_miss, width = 6, height = 4)
