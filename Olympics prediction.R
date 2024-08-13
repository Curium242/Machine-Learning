library(dplyr)

athletes <- read.csv("C:\\Users\\Astatine\\Documents\\athlete_events.csv", stringsAsFactors = FALSE)
noc_regions <- read.csv("C:\\Users\\Astatine\\Documents\\noc_regions.csv", stringsAsFactors = FALSE)

merged_data <- merge(athletes, noc_regions, by = "NOC")

merged_data <- rename(merged_data, Countries = region)
gdp_data <- read.csv("C:\\Users\\Astatine\\Documents\\GDP by Country 1999-2022.csv", stringsAsFactors = FALSE)

write.csv(merged_data, "C:\\Users\\Astatine\\Documents\\updated_athletes.csv", row.names = FALSE)
filtered_data <- merged_data %>%
  filter(Year >= 2000, !is.na(Medal))

write.csv(filtered_data, "C:\\Users\\Astatine\\Documents\\filtered_athletes.csv", row.names = FALSE)

install.packages(c("randomForest", "caret", "nnet","tidyverse","broom","zoo"))
install.packages("zoo")
library(tidyverse)
library(caret)
library(randomForest)
library(broom)
filtered_data <- read.csv("C:\\Users\\Astatine\\Documents\\filtered_athletes_2.csv", stringsAsFactors = FALSE)

yearly_medal_tally <- filtered_data %>%
  mutate(Medal_Value = case_when(
    Medal == "Gold" ~ 3,
    Medal == "Silver" ~ 2,
    Medal == "Bronze" ~ 1,
    TRUE ~ 0)) %>%
  group_by(Countries, Year) %>%
  summarise(Total_Medals = sum(Medal_Value), .groups = 'drop') %>%
  arrange(desc(Total_Medals))
years <- seq(min(yearly_medal_tally$Year), max(yearly_medal_tally$Year), by = 4)
countries <- unique(yearly_medal_tally$Countries)

all_combinations <- expand.grid(Year = years, Countries = countries)

complete_data <- left_join(all_combinations, yearly_medal_tally, by = c("Countries", "Year"))
complete_data$Total_Medals[is.na(complete_data$Total_Medals)] <- 0
write.csv(complete_data, "C:\\Users\\Astatine\\Documents\\complete_athletes.csv", row.names = FALSE)

training_data <- yearly_medal_tally

  mutate(Year = parse_number(Year)) 
write.csv(gdp_long, "C:\\Users\\Astatine\\Documents\\gdp.csv", row.names = FALSE)
gdp_long <- read.csv("C:\\Users\\Astatine\\Documents\\gdp.csv", stringsAsFactors = FALSE)
gdp_long1 <- gdp_data %>%
  pivot_longer(
    cols = -Country, 
    names_to = "Year", 
    values_to = "GDP"
  ) %>%
  mutate(Year = as.numeric(Year))
merged_data <- inner_join(complete_data, gdp_long, by = c("Countries" = "Country", "Year"))
write.csv(merged_data, "C:\\Users\\Astatine\\Documents\\merged.csv", row.names = FALSE)

library(randomForest)
library(caret)


data <- read_csv("C:\\Users\\Astatine\\Documents\\combined_data_with_gdp.csv")

data$GDP <- as.numeric(gsub(",", "", data$GDP))

growth_data <- data %>%
  group_by(Country) %>%
  arrange(Country, Year) %>%
  mutate(Growth_Rate = (GDP - lag(GDP)) / lag(GDP)) %>% \
  na.omit() %>% 
  summarise(Average_Growth = mean(Growth_Rate, na.rm = TRUE)) 
last_known_year <- max(data$Year)
years_diff <- 2024 - last_known_year

data <- data %>%
  filter(Year == last_known_year) %>%
  left_join(growth_data, by = "Country") %>%
  mutate(Projected_GDP_2024 = ifelse(is.na(Average_Growth), GDP, GDP * ((1 + Average_Growth) ^ years_diff))) # Project GDP for 2024

future_data <- data.frame(Country = data$Country, Year = rep(2024, nrow(data)), GDP = data$Projected_GDP_2024)

set.seed(123)  
training_indices <- sample(1:nrow(data), 0.8 * nrow(data))
training_data <- data[training_indices, ]
testing_data <- data[-training_indices, ]

model <- lm(Total_Medals ~ GDP + Year, data = training_data)

predicted_medals_2024 <- predict(model, newdata = future_data)

predicted_medals_2024[predicted_medals_2024 < 0] <- 0

results_2024 <- data.frame(Country = future_data$Country, Predicted_Medals = predicted_medals_2024)

print(results_2024)
top_countries <- results_2024 %>% 
  arrange(desc(Predicted_Medals)) %>% 
  head(3)
print(top_countries)
results_2024 <- data.frame(
  Country = future_data$Country, 
  GDP = future_data$GDP, 
  Predicted_Medals = predicted_medals_2024
)

head(results_2024)

prediction_intervals <- predict(model, newdata = filter(future_data, Country %in% top_countries$Country), interval = "confidence")
top_countries_intervals <- cbind(top_countries, prediction_intervals)
print(top_countries_intervals)

summary(model)
ggplot(results_2024, aes(x = GDP, y = Predicted_Medals)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Scatter Plot of GDP vs. Predicted Medals",
       x = "GDP (in billions)",
       y = "Predicted Medal Count")
results_2024 <- data.frame(Country = future_data$Country, GDP = future_data$GDP, Predicted_Medals = predicted_medals_2024)
ggplot(results_2024, aes(x = GDP, y = Predicted_Medals)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Scatter Plot of GDP vs. Predicted Medals",
       x = "GDP (in billions)",
       y = "Predicted Medal Count")
ggplot(results_2024, aes(x = GDP, y = Predicted_Medals)) +
  geom_point() +
  scale_x_log10(labels = scales::comma) +  # Converts x-axis to a logarithmic scale and formats labels
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Scatter Plot of Log GDP vs. Predicted Medals",
       x = "GDP (log scale, in billions)",
       y = "Predicted Medal Count") +
  theme_minimal()

top_countries <- results_2024 %>%
  arrange(desc(Predicted_Medals)) %>%
  head(10)

ggplot(top_countries, aes(x = reorder(Country, -Predicted_Medals), y = Predicted_Medals, fill = Country)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top Countries by Predicted Medals",
       x = "Country",
       y = "Predicted Medal Count")

residuals <- resid(model)
fitted_values <- fitted(model)

ggplot(data.frame(Fitted = fitted_values, Residuals = residuals), aes(x = Fitted, y = Residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs. Fitted Values",
       x = "Fitted Values (Predicted Medals)",
       y = "Residuals") +
  theme_minimal()


library(car)


model_leverage <- hatvalues(model)
model_residuals <- rstandard(model)

plot(model_leverage, model_residuals,
     xlab = "Leverage",
     ylab = "Standardized Residuals",
     main = "Leverage vs. Standardized Residuals",
     pch = 20, cex = 1.2,
     col = ifelse(abs(model_residuals) > 2 | model_leverage > 0.2, "red", "black")) # Color points based on condition

abline(h = 0, col = "red", lwd = 2)

avg_leverage <- 2 * mean(model_leverage)
abline(v = avg_leverage, col = "blue", lwd = 2)

with(data.frame(model_leverage, model_residuals, Row = row.names(training_data)),
     text(model_leverage, model_residuals, labels = Row, cex = 0.8, pos = 4, col = "blue"))
