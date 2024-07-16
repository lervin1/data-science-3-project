# Data / Variable Exploration

# load packages & data ---
library(tidyverse)
library(here)
library(tidymodels)
library(naniar)
library(knitr)
library(kableExtra)

load(here("data/baseball_dataset.rda"))

# Checking for missingne

# skim data
skimr::skim(baseball_dataset)

missingness <- baseball_dataset |> 
  miss_var_summary() 

view(missingness)

missingness_filtered <- missingness |> 
  filter(n_miss > 0) 


ggplot(missingness_filtered, aes(x = variable, y = pct_miss)) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.5) +
  labs(title = "Missingness Summary", x = "Variable", y = "Percentage of Missing Values") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Target Variable = is_hit

# inspecting target variable

is_hit_summary <- baseball_dataset |> summarize(
  Count = n(),
  Mean = mean(is_hit, na.rm = TRUE),
  Q1 = quantile(is_hit, 0.25, na.rm = TRUE),
  Median = median(is_hit, na.rm = TRUE),
  Q3 = quantile(is_hit, 0.75, na.rm = TRUE))



# looking at hit vs not hit
baseball_dataset |> 
  count(is_hit)

# for qmd
kable(is_hit_summary, "html") |> 
  kable_styling(position = "center") |> 
  kable_classic(html_font = "Georgia") 

 baseball_dataset |> 
  ggplot(aes(x = is_hit)) +
  geom_bar() +
  theme_bw() +
  labs(x = "Hit or Not (0 = Not Hit, 1 = Hit)", y = "Count", title = "Comparison of Hit vs Non Hit Pitches") +
  theme(text=element_text(size=10,  family="Georgia"))

 # Looking at correlations between is_hit 
 
 # determining correlations
 target_column <- "is_hit"
 
 # selecting only numeric variables for correlation table
baseball_dataset_correlations <- baseball_dataset |> 
   select_if(is.numeric) 
 
correlation_matrix <- baseball_dataset_correlations |> 
  select(all_of(target_column), everything()) %>%
  cor()
correlation_matrix
correlation_long <- as.data.frame(as.table(correlation_matrix))

correlation_long

# Filter the long-format correlation table to show correlations only with "is_hit" & No NA as resulting correlation
is_hit_correlations <- correlation_long[correlation_long$Var1 == "is_hit" ,] 
is_hit_correlations <- is_hit_correlations[complete.cases(is_hit_correlations), ]

# Making table arranged from largest correlation to smallest
correlation_table <- is_hit_correlations |> 
  arrange(desc(Freq))

# Print the filtered correlation table
print(correlation_table)
 
# graph of correlations
ggplot(correlation_table, aes(Var1, Var2, fill = Freq)) +
   geom_tile(color = "white") +
   scale_fill_gradient2(low = "red", high = "darkgreen", mid = "white", midpoint = 0) +
   theme_bw() +
   theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
   theme(text=element_text(size=10,  family="Georgia")) +
   labs(title = "Correlation Matrix",
        x = "Variable 1",
        y = "Variable 2")

