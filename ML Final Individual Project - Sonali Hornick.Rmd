---
title: "ML Individual Project - Sonali Hornick"
output:
  pdf_document: default
  word_document: default
  html_document: default
date: "2024-08-02"
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r prep}
# Load libraries
library(dplyr)
library(readr)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(gbm)
library(BART)
library(tm)
library(text2vec)
# Load datasets
austinhouses_data <- read_csv('C:/Users/sonal/Downloads/austinhouses_data.csv')
austinhouses_holdout <- read_csv('C:/Users/sonal/Downloads/austinhouses_holdout.csv')
# Initial data exploration and verification
str(austinhouses_data)
summary(austinhouses_data)
```

```{r ftengineering}
# Handling missing values
austinhouses_data <- austinhouses_data %>%
  mutate(garageSpaces = ifelse(is.na(garageSpaces), 0, garageSpaces))
# Create new features
austinhouses_data <- austinhouses_data %>%
  mutate(price_per_sqft = latestPrice / livingAreaSqFt,
         age = 2024 - yearBuilt)
summary(austinhouses_data)
# Create neighborhood-level features by aggregating data by zipcode
neighborhood_data <- austinhouses_data %>%
  group_by(zipcode) %>%
  summarise(
    avg_price = mean(latestPrice, na.rm = TRUE),
    median_price = median(latestPrice, na.rm = TRUE),
    avg_livingAreaSqFt = mean(livingAreaSqFt, na.rm = TRUE),
    median_livingAreaSqFt = median(livingAreaSqFt, na.rm = TRUE),
    avg_numOfBathrooms = mean(numOfBathrooms, na.rm = TRUE),
    avg_numOfBedrooms = mean(numOfBedrooms, na.rm = TRUE),
    num_houses = n()
  )
# Inspect the aggregated neighborhood data
head(neighborhood_data)
# Merge the neighborhood data back into the original dataset
austinhouses_data <- left_join(austinhouses_data, neighborhood_data, by = 'zipcode')
# Verify the merged dataset
summary(austinhouses_data)
# Save the neighborhood data to a CSV file
write_csv(neighborhood_data, 'neighborhood_data.csv')
```

```{r modeltraining}
# Select relevant columns and ensure all variables are correctly formatted
austinhouses <- austinhouses_data %>%
  select(latestPrice, latitude, longitude, hasAssociation, livingAreaSqFt, numOfBathrooms, numOfBedrooms, zipcode) %>%
  mutate(
    log_latestPrice = log(latestPrice),
    hasAssociation = as.factor(hasAssociation)
  )
# Split the data into training and testing sets
set.seed(200)
train_indices <- createDataPartition(austinhouses$log_latestPrice, p = 0.8, list = FALSE)
train_data <- austinhouses[train_indices, ]
test_data <- austinhouses[-train_indices, ]
# Initialize an empty data frame to store the MSE values
mse_values <- data.frame(
  Model = character(),
  MSE = numeric(),
  stringsAsFactors = FALSE
)
# Regression Tree
tree_austinhouses <- rpart(log_latestPrice ~ latitude + longitude + hasAssociation + livingAreaSqFt + numOfBathrooms + numOfBedrooms, data = train_data)
predictions_tree <- predict(tree_austinhouses, test_data)
predictions_tree_price <- exp(predictions_tree)
tree_mse <- mean((test_data$latestPrice - predictions_tree_price)^2)
mse_values <- rbind(mse_values, data.frame(Model = "Regression Tree", MSE = tree_mse))
print(paste("Regression Tree MSE:", tree_mse))
plot(tree_austinhouses)
text(tree_austinhouses, pretty = 0)
# Bagging
bag.austinhouses <- randomForest(log_latestPrice ~ latitude + longitude + hasAssociation + livingAreaSqFt + numOfBathrooms + numOfBedrooms, data = train_data, mtry = 6, importance = TRUE)
predictions_bag <- predict(bag.austinhouses, newdata = test_data)
predictions_bag_price <- exp(predictions_bag)
bagging_mse <- mean((predictions_bag_price - test_data$latestPrice)^2)
mse_values <- rbind(mse_values, data.frame(Model = "Bagging", MSE = bagging_mse))
print(paste("Bagging MSE:", bagging_mse))
# Random Forest
rf_austinhouses <- randomForest(log_latestPrice ~ latitude + longitude + hasAssociation + livingAreaSqFt + numOfBathrooms + numOfBedrooms, data = train_data, mtry = 2, importance = TRUE)
predictions_rf <- predict(rf_austinhouses, newdata = test_data)
predictions_rf_price <- exp(predictions_rf)
rf_mse <- mean((predictions_rf_price - test_data$latestPrice)^2)
mse_values <- rbind(mse_values, data.frame(Model = "Random Forest", MSE = rf_mse))
print(paste("Random Forest MSE:", rf_mse))
varImpPlot(rf_austinhouses)
# BART
x_train <- train_data %>%
  select(latitude, longitude, hasAssociation, livingAreaSqFt, numOfBathrooms, numOfBedrooms) %>%
  mutate(hasAssociation = as.numeric(as.factor(hasAssociation)))
x_test <- test_data %>%
  select(latitude, longitude, hasAssociation, livingAreaSqFt, numOfBathrooms, numOfBedrooms) %>%
  mutate(hasAssociation = as.numeric(as.factor(hasAssociation)))
y_train <- train_data$log_latestPrice
y_test <- test_data$log_latestPrice
# Convert data frames to matrices
x_train <- as.matrix(x_train)
x_test <- as.matrix(x_test)
bartfit <- gbart(x.train = x_train, y.train = y_train, x.test = x_test)
yhat_bart <- bartfit$yhat.test.mean
bart_mse <- mean((exp(y_test) - exp(yhat_bart))^2)
mse_values <- rbind(mse_values, data.frame(Model = "BART", MSE = bart_mse))
print(paste("BART MSE:", bart_mse))

# COMPARISON OF MODELS
# Plot the MSE values for better visualization
ggplot(mse_values, aes(x = Model, y = MSE)) +
  geom_bar(stat = "identity", fill = "hotpink") +
  theme_minimal() +
  labs(title = "Model Comparison", y = "MSE", x = "Model Name")
# Print and compare the MSE values
print(mse_values)
```

```{r holdoutpredictions}
# Generate predictions on the holdout dataset using the Bagging model
predictions_holdout_bagging <- predict(bag.austinhouses, data = austinhouses_holdout)
predictions_holdout_bagging_price <- exp(predictions_holdout_bagging)
# Create a CSV file with the predictions
submission_bagging <- data.frame(PredictedPrice = predictions_holdout_bagging_price)
write.csv(submission_bagging, file = 'C:/Users/sonal/Downloads/bagging_predictions.csv', row.names = FALSE)
print("Bagging predictions are attached in submission on canvas and have been saved to 'bagging_predictions.csv'.")
```

