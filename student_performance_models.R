# ==============================================================================
# Student Performance Factors - Model Development
# ==============================================================================
# Dataset: StudentPerformanceFactors.csv
# Objective: Predict exam scores and identify key performance drivers
# ==============================================================================

# Load required libraries

library(psych)
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(ggplot2)
library(corrplot)
library(car)

# Set seed for reproducibility
set.seed(42)

# ==============================================================================
# 1. DATA LOADING & EXPLORATION
# ==============================================================================

# Load the data
df <- read.csv("StudentPerformanceFactors.csv")

# Display structure and summary
cat("Dataset shape:", nrow(df), "rows x", ncol(df), "columns\n\n")
str(df)
summary(df)

# Check for missing values
cat("\nMissing values:\n")
print(colSums(is.na(df)))

# ==============================================================================
# 2. DATA PREPROCESSING
# ==============================================================================

# Create a copy for modeling
df_model <- df

# Identify and encode categorical variables
categorical_cols <- c("Parental_Involvement", "Access_to_Resources", 
                      "Extracurricular_Activities", "Internet_Access",
                      "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
                      "School_Type", "Peer_Influence", "Learning_Disabilities",
                      "Parental_Education_Level", "Distance_from_Home", "Gender")

# One-hot encode categorical variables
df_encoded <- df_model %>%
  mutate(across(all_of(categorical_cols), as.factor))

# Create dummy variables for modeling
dummyVars_obj <- dummyVars(~ ., data = df_encoded[, -which(names(df_encoded) == "Exam_Score")])
df_dummies <- as.data.frame(predict(dummyVars_obj, newdata = df_encoded))
df_dummies$Exam_Score <- df_encoded$Exam_Score

cat("\nData preprocessing complete. Features created:", ncol(df_dummies) - 1, "\n")

# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ==============================================================================

cat(paste("\n", strrep("=", 80), "\nEXPLORATORY DATA ANALYSIS\n", strrep("=", 80), "\n\n", sep = ""))

# Summary statistics for exam scores
cat("Exam Score Statistics:\n")
print(describe(df$Exam_Score))

# Distribution of exam scores
p1 <- ggplot(df, aes(x = Exam_Score)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  labs(title = "Distribution of Exam Scores",
       x = "Exam Score", y = "Frequency") +
  theme_minimal()
print(p1)

# Correlation analysis (numeric variables only)
numeric_cols <- sapply(df_model, is.numeric)
cor_matrix <- cor(df_model[, numeric_cols])

# Correlation with target variable
target_corr <- cor_matrix[, "Exam_Score"]
target_corr_sorted <- sort(target_corr[target_corr != 1], decreasing = TRUE)

cat("\nTop 10 Correlations with Exam Score:\n")
print(target_corr_sorted[1:10])

# Visualize correlations
corrplot(cor_matrix, method = "circle", type = "upper",
         tl.col = "black", diag = FALSE,
         title = "Correlation Matrix - Student Performance Data")

# ==============================================================================
# 4. TRAIN-TEST SPLIT
# ==============================================================================

trainIndex <- createDataPartition(df_dummies$Exam_Score, p = 0.8, list = FALSE)
train_data <- df_dummies[trainIndex, ]
test_data <- df_dummies[-trainIndex, ]

cat("\nTrain-Test Split:\n")
cat("Training set:", nrow(train_data), "rows\n")
cat("Test set:", nrow(test_data), "rows\n")

# Separate features and target
X_train <- train_data[, -which(names(train_data) == "Exam_Score")]
y_train <- train_data$Exam_Score

X_test <- test_data[, -which(names(test_data) == "Exam_Score")]
y_test <- test_data$Exam_Score

# ==============================================================================
# 5. MODEL 1: LINEAR REGRESSION
# ==============================================================================

cat(paste("\n", strrep("=", 80), "\nMODEL 1: LINEAR REGRESSION\n", strrep("=", 80), "\n\n", sep = ""))

lm_model <- lm(Exam_Score ~ ., data = train_data)

# Summary
cat("\nModel Summary:\n")
print(summary(lm_model))

# Predictions
lm_pred_train <- predict(lm_model, newdata = train_data)
lm_pred_test <- predict(lm_model, newdata = test_data)

# Performance metrics
lm_train_rmse <- sqrt(mean((lm_pred_train - y_train)^2))
lm_test_rmse <- sqrt(mean((lm_pred_test - y_test)^2))
lm_train_r2 <- 1 - (sum((lm_pred_train - y_train)^2) / sum((y_train - mean(y_train))^2))
lm_test_r2 <- 1 - (sum((lm_pred_test - y_test)^2) / sum((y_test - mean(y_test))^2))

cat("\nLinear Regression Performance:\n")
cat("Training RMSE:", round(lm_train_rmse, 4), "\n")
cat("Test RMSE:", round(lm_test_rmse, 4), "\n")
cat("Training R²:", round(lm_train_r2, 4), "\n")
cat("Test R²:", round(lm_test_r2, 4), "\n")

# Diagnostic plots
par(mfrow = c(2, 2))
plot(lm_model, which = 1:4)
par(mfrow = c(1, 1))

## ==============================================================================
# 6. MODEL 2: RANDOM FOREST
# ==============================================================================

cat(paste("\n", strrep("=", 80), "\nMODEL 2: RANDOM FOREST\n", strrep("=", 80), "\n\n", sep = ""))

rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 200,
  mtry = sqrt(ncol(X_train)),
  max.depth = 15,
  importance = TRUE,
  seed = 42
)

# Predictions
rf_pred_train <- predict(rf_model, newdata = X_train)
rf_pred_test <- predict(rf_model, newdata = X_test)

# Performance metrics
rf_train_rmse <- sqrt(mean((rf_pred_train - y_train)^2))
rf_test_rmse <- sqrt(mean((rf_pred_test - y_test)^2))
rf_train_r2 <- 1 - (sum((rf_pred_train - y_train)^2) / sum((y_train - mean(y_train))^2))
rf_test_r2 <- 1 - (sum((rf_pred_test - y_test)^2) / sum((y_test - mean(y_test))^2))

cat("\nRandom Forest Performance:\n")
cat("Training RMSE:", round(rf_train_rmse, 4), "\n")
cat("Test RMSE:", round(rf_test_rmse, 4), "\n")
cat("Training R²:", round(rf_train_r2, 4), "\n")
cat("Test R²:", round(rf_test_r2, 4), "\n")

# Feature importance
importance_df <- data.frame(
  Feature = rownames(importance(rf_model)),
  Importance = importance(rf_model)[, "%IncMSE"]
) %>%
  arrange(desc(Importance)) %>%
  head(15)

cat("\nTop 15 Important Features:\n")
print(importance_df)

# Plot feature importance
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Random Forest Feature Importance",
       x = "Feature", y = "% Increase in MSE") +
  theme_minimal()

# ==============================================================================
# 7. MODEL 3: GRADIENT BOOSTING (XGBoost)
# ==============================================================================

cat(paste("\n", strrep("=", 80), "\nMODEL 3: GRADIENT BOOSTING (XGBoost)\n", strrep("=", 80), "\n\n", sep = ""))

# Prepare data for XGBoost
dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(as.matrix(X_test), label = y_test)

# Model parameters
params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  verbose = 0,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 20
)

# Predictions
xgb_pred_train <- predict(xgb_model, newdata = dtrain)
xgb_pred_test <- predict(xgb_model, newdata = dtest)

# Performance metrics
xgb_train_rmse <- sqrt(mean((xgb_pred_train - y_train)^2))
xgb_test_rmse <- sqrt(mean((xgb_pred_test - y_test)^2))
xgb_train_r2 <- 1 - (sum((xgb_pred_train - y_train)^2) / sum((y_train - mean(y_train))^2))
xgb_test_r2 <- 1 - (sum((xgb_pred_test - y_test)^2) / sum((y_test - mean(y_test))^2))

cat("\nXGBoost Performance:\n")
cat("Training RMSE:", round(xgb_train_rmse, 4), "\n")
cat("Test RMSE:", round(xgb_test_rmse, 4), "\n")
cat("Training R²:", round(xgb_train_r2, 4), "\n")
cat("Test R²:", round(xgb_test_r2, 4), "\n")

# Feature importance
xgb_importance <- xgb.importance(model = xgb_model)

cat("\nTop 15 Important Features (XGBoost):\n")
print(head(xgb_importance, 15))

# Plot
xgb.plot.importance(head(xgb_importance, 15))

# ==============================================================================
# 8. MODEL COMPARISON
# ==============================================================================

cat(paste("\n", strrep("=", 80), "\nMODEL COMPARISON\n", strrep("=", 80), "\n\n", sep = ""))

comparison_df <- data.frame(
  Model = c("Linear Regression", "Random Forest", "XGBoost"),
  Train_RMSE = c(lm_train_rmse, rf_train_rmse, xgb_train_rmse),
  Test_RMSE = c(lm_test_rmse, rf_test_rmse, xgb_test_rmse),
  Train_R2 = c(lm_train_r2, rf_train_r2, xgb_train_r2),
  Test_R2 = c(lm_test_r2, rf_test_r2, xgb_test_r2)
)

print(comparison_df)

# Visualization
comparison_long <- comparison_df %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

ggplot(comparison_long %>% filter(grepl("RMSE", Metric)), 
       aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  labs(title = "Model Comparison: RMSE",
       x = "Model", y = "RMSE") +
  theme_minimal()

# ==============================================================================
# 9. RESIDUAL ANALYSIS
# ==============================================================================

cat(paste("\n", strrep("=", 80), "\nRESIDUAL ANALYSIS\n", strrep("=", 80), "\n\n", sep = ""))

# Best model residuals (using best test RMSE)
best_model <- which.min(c(lm_test_rmse, rf_test_rmse, xgb_test_rmse))
best_pred <- c(lm_pred_test, rf_pred_test, xgb_pred_test)[
  (best_model - 1) * length(y_test) + 1:length(y_test)
]
residuals <- y_test - best_pred

cat("Residual Statistics (Best Model):\n")
cat("Mean:", round(mean(residuals), 6), "\n")
cat("Std Dev:", round(sd(residuals), 4), "\n")
cat("Min:", round(min(residuals), 4), "\n")
cat("Max:", round(max(residuals), 4), "\n")

# Normality test
shapiro_test <- shapiro.test(residuals)
cat("\nShapiro-Wilk Test (p-value):", round(shapiro_test$p.value, 6), "\n")

# ==============================================================================
# 10. PREDICTIONS ON NEW DATA (Example)
# ==============================================================================

cat(paste("\n", strrep("=", 80), "\nEXAMPLE: PREDICTIONS ON NEW STUDENT PROFILE\n", strrep("=", 80), "\n\n", sep = ""))

# Example new student (same format as training data)
cat("All models are ready for deployment.\n")
cat("Use predict() function to make predictions on new data.\n")
cat("Example: predict(rf_model, newdata = new_student_data)\n")
