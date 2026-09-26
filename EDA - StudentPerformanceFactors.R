getwd()

install.packages("tidyverse")
install.packages("skimr")
install.packages("janitor")
install.packages("corrplot")
install.packages("naniar")


library(tidyverse)
library(skimr)
library(janitor)
library(corrplot)
library(naniar)
 
## ---- 1. Import data
## Update the path below to wherever you saved the CSV locally.
df <- read_csv("StudentPerformanceFactors.csv") %>%
  clean_names()  # standardises column names to snake_case (safety net;
                 # this file's names are already clean, but harmless to run)
 
## ---- 2. First look at the data 
dim(df)          # rows x columns -> should be 6607 x 20
glimpse(df)      # column names + data types + a preview of values
head(df, 10)     # first 10 rows, sanity check that the import looks right
sum(duplicated(df))  # count of exact duplicate rows (expect 0, but always check)
 
## ---- 3. Classify variables by type 
## This is for the "names and types of variables" part of the proposal.
## Continuous (interval/ratio, numeric, meaningful magnitude):
continuous_vars <- c("hours_studied", "attendance", "sleep_hours",
                      "previous_scores", "tutoring_sessions",
                      "physical_activity", "exam_score")
 
## Ordinal (categorical, but with a natural Low < Medium < High / Near < Far order):
ordinal_vars <- c("parental_involvement", "access_to_resources",
                   "motivation_level", "family_income",
                   "teacher_quality", "distance_from_home",
                   "parental_education_level")
 
## Nominal (categorical, no inherent order):
nominal_vars <- c("extracurricular_activities", "internet_access",
                   "school_type", "peer_influence",
                   "learning_disabilities", "gender")
 
## Recode ordinal variables as ordered factors so summaries/plots respect
## the natural ordering instead of sorting alphabetically (e.g. "High" before "Low").
df <- df %>%
  mutate(
    parental_involvement     = factor(parental_involvement, levels = c("Low","Medium","High"), ordered = TRUE),
    access_to_resources      = factor(access_to_resources, levels = c("Low","Medium","High"), ordered = TRUE),
    motivation_level         = factor(motivation_level, levels = c("Low","Medium","High"), ordered = TRUE),
    family_income            = factor(family_income, levels = c("Low","Medium","High"), ordered = TRUE),
    teacher_quality          = factor(teacher_quality, levels = c("Low","Medium","High"), ordered = TRUE),
    distance_from_home       = factor(distance_from_home, levels = c("Near","Moderate","Far"), ordered = TRUE),
    parental_education_level = factor(parental_education_level,
                                       levels = c("High School","College","Postgraduate"),
                                       ordered = TRUE),
    across(all_of(nominal_vars), as.factor)  # nominal vars just become unordered factors
  )
 
## ---- 4. Missing value analysis ----------------------------------
## Quick numeric summary: how many NAs per column, and what % of rows that is.
colSums(is.na(df))
df %>%
  summarise(across(everything(), ~ mean(is.na(.)) * 100)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "pct_missing") %>%
  filter(pct_missing > 0) %>%
  arrange(desc(pct_missing))
 
## Visual: bar chart of missingness by variable — makes it obvious at a glance
## which columns have gaps (expect teacher_quality, parental_education_level,
## distance_from_home to show up here, each roughly 1-1.5% missing).
gg_miss_var(df, show_pct = TRUE)
 
## Check whether missingness clusters in the same rows (i.e. do students
## missing teacher_quality also tend to be missing distance_from_home?).
## Useful for deciding whether missingness is random or systematic.
vis_miss(df, warn_large_data = FALSE)
 
## ---- 5. Summary statistics: continuous variables -------------------
## skim() gives mean, sd, quartiles, a mini histogram, and missing count
## for every numeric variable in one table — good for pasting into the report.
df %>% select(all_of(continuous_vars)) %>% skim()
 
## Base R equivalent if you just want a plain table:
summary(df %>% select(all_of(continuous_vars)))
 
## ---- 6. Summary statistics: categorical variables -------------------
## Frequency + proportion tables for every ordinal/nominal variable.
categorical_vars <- c(ordinal_vars, nominal_vars)
for (v in categorical_vars) {
  cat("\n---", v, "---\n")
  print(df %>% count(.data[[v]], name = "n") %>% mutate(pct = round(100 * n / sum(n), 1)))
}
 
## ---- 7. Univariate distributions: continuous variables ----------------
## Histograms show shape (skew, multimodality) for each continuous variable.
df %>%
  select(all_of(continuous_vars)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill = "steelblue", colour = "white") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distributions of continuous variables")
 
## Boxplots double up as an outlier-detection tool: points beyond the
## whiskers are flagged by ggplot as potential outliers.
df %>%
  select(all_of(continuous_vars)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(fill = "lightgreen") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Boxplots for outlier detection")
 
## Explicit IQR-based outlier count per continuous variable (Tukey's rule:
## anything beyond 1.5 x IQR past Q1/Q3 counts as an outlier).
outlier_summary <- df %>%
  select(all_of(continuous_vars)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  group_by(variable) %>%
  summarise(
    q1 = quantile(value, 0.25, na.rm = TRUE),
    q3 = quantile(value, 0.75, na.rm = TRUE),
    iqr = q3 - q1,
    lower = q1 - 1.5 * iqr,
    upper = q3 + 1.5 * iqr,
    n_outliers = sum(value < lower | value > upper, na.rm = TRUE)
  )
outlier_summary
 
## Known anomaly worth flagging in the proposal: Exam_Score should logically
## be capped at 100, but check for and inspect any values above that.
df %>% filter(exam_score > 100)
 
## ---- 8. Univariate distributions: categorical variables -----------------
for (v in categorical_vars) {
  p <- ggplot(df, aes(x = .data[[v]])) +
    geom_bar(fill = "coral") +
    theme_minimal() +
    labs(title = paste("Distribution of", v), x = v, y = "Count")
  print(p)
}
 
## ---- 9. Bivariate: Exam_Score vs continuous predictors ------------------
## Scatterplots with a fitted trend line show whether the relationship
## with the outcome looks linear, and how strong/noisy it is.
predictors_num <- setdiff(continuous_vars, "exam_score")
df %>%
  select(all_of(c(predictors_num, "exam_score"))) %>%
  pivot_longer(all_of(predictors_num), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value, y = exam_score)) +
  geom_point(alpha = 0.15) +
  geom_smooth(method = "lm", se = FALSE, colour = "red") +
  facet_wrap(~ variable, scales = "free_x") +
  theme_minimal() +
  labs(title = "Exam_Score vs. continuous predictors")
 
## ---- 10. Bivariate: Exam_Score vs categorical predictors ------------------
## Boxplots of the outcome split by each categorical group — quick way to
## see which categorical factors seem associated with higher/lower scores.
for (v in categorical_vars) {
  p <- ggplot(df, aes(x = .data[[v]], y = exam_score, fill = .data[[v]])) +
    geom_boxplot(show.legend = FALSE) +
    theme_minimal() +
    labs(title = paste("Exam_Score by", v), x = v, y = "Exam Score")
  print(p)
}
 
## ---- 11. Correlation matrix among continuous variables ------------------
## Pearson correlations flag which numeric predictors move together with
## Exam_Score (and with each other, which matters later for multicollinearity).
corr_matrix <- df %>%
  select(all_of(continuous_vars)) %>%
  cor(use = "pairwise.complete.obs")
 
corr_matrix  # printed table
corrplot(corr_matrix, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "Correlation matrix: continuous variables", mar = c(0,0,2,0))
 

