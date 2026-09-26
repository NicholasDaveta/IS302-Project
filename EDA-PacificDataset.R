# Pacific Employment Data - Exploratory Data Analysis
# EDA Script for Employed Population by Education, Occupation and Economic Sector

setwd("~/2026/Semester 2/IS302/Project/Proposal - 2")

library(readr)
library(dplyr)

# Load the dataset
pacific <- read_csv("Employed population by education, occupation and economic sector data.csv")

# Display first few rows
View(pacific)

# 1. Dataset Overview
cat("\n===== DATASET OVERVIEW =====\n")
cat("Dimensions:", nrow(pacific), "rows x", ncol(pacific), "columns\n")
cat("Column names and types:\n")
str(pacific)

# 2. Basic Statistics
cat("\n===== SUMMARY STATISTICS =====\n")
summary(pacific)

# 3. Missing Values
cat("\n===== MISSING VALUES =====\n")
missing_summary <- colSums(is.na(pacific))
print(missing_summary)
cat("\nPercentage missing by column:\n")
print(round(colSums(is.na(pacific)) / nrow(pacific) * 100, 2))

# 4. Target Variable (Observation value / OBS_VALUE) Distribution
cat("\n===== OBSERVATION VALUE STATISTICS =====\n")
cat("Mean:", mean(pacific$`OBS_VALUE`, na.rm = TRUE), "\n")
cat("Median:", median(pacific$`OBS_VALUE`, na.rm = TRUE), "\n")
cat("Std Dev:", sd(pacific$`OBS_VALUE`, na.rm = TRUE), "\n")
cat("Min:", min(pacific$`OBS_VALUE`, na.rm = TRUE), "\n")
cat("Max:", max(pacific$`OBS_VALUE`, na.rm = TRUE), "\n")
cat("Total employed:", sum(pacific$`OBS_VALUE`, na.rm = TRUE), "\n")

# 5. Geographic Distribution (Pacific Island Countries)
cat("\n===== GEOGRAPHIC DISTRIBUTION =====\n")
cat("Countries/Territories represented:\n")
geo_count <- table(pacific$`Pacific Island Countries and territories`)
print(sort(geo_count, decreasing = TRUE))
cat("\nTotal records by country:\n")
print(tapply(pacific$`OBS_VALUE`, pacific$`Pacific Island Countries and territories`, 
             function(x) sum(x, na.rm = TRUE)))

# 6. Temporal Distribution
cat("\n===== TIME PERIOD DISTRIBUTION =====\n")
cat("Years covered:\n")
time_count <- table(pacific$Time)
print(sort(time_count, decreasing = TRUE))
cat("\nTotal employed by year:\n")
print(tapply(pacific$`OBS_VALUE`, pacific$Time, function(x) sum(x, na.rm = TRUE)))

# 7. Education Level Distribution
cat("\n===== EDUCATION LEVEL DISTRIBUTION =====\n")
cat("Frequency of each education level:\n")
print(table(pacific$`Education level`))
cat("\nAverage employment by education level:\n")
education_stats <- tapply(pacific$`OBS_VALUE`, pacific$`Education level`, 
                          function(x) c(Mean = mean(x, na.rm = TRUE), 
                                        Total = sum(x, na.rm = TRUE), 
                                        N = sum(!is.na(x))))
print(education_stats)

# 8. Occupation Distribution
cat("\n===== OCCUPATION DISTRIBUTION =====\n")
cat("Frequency of each occupation type:\n")
print(table(pacific$Occupation))
cat("\nAverage employment by occupation:\n")
occupation_stats <- tapply(pacific$`OBS_VALUE`, pacific$Occupation, 
                           function(x) c(Mean = mean(x, na.rm = TRUE), 
                                         Total = sum(x, na.rm = TRUE), 
                                         N = sum(!is.na(x))))
print(occupation_stats)

# 9. Economic Sector Distribution
cat("\n===== ECONOMIC SECTOR DISTRIBUTION =====\n")
cat("Frequency of each economic sector:\n")
print(table(pacific$`Economic sector`))
cat("\nAverage employment by economic sector:\n")
sector_stats <- tapply(pacific$`OBS_VALUE`, pacific$`Economic sector`, 
                       function(x) c(Mean = mean(x, na.rm = TRUE), 
                                     Total = sum(x, na.rm = TRUE), 
                                     N = sum(!is.na(x))))
print(sector_stats)

# 10. Disability Status Distribution
cat("\n===== DISABILITY STATUS DISTRIBUTION =====\n")
cat("Frequency by disability status:\n")
print(table(pacific$Disability))
cat("\nEmployment by disability status:\n")
print(tapply(pacific$`OBS_VALUE`, pacific$Disability, 
             function(x) c(Mean = mean(x, na.rm = TRUE), 
                           Total = sum(x, na.rm = TRUE))))

# 11. Cross-tabulations
cat("\n===== EDUCATION BY OCCUPATION =====\n")
edu_occ <- xtabs(OBS_VALUE ~ `Education level` + Occupation, data = pacific, na.action = na.pass)
print(edu_occ)

cat("\n===== EDUCATION BY ECONOMIC SECTOR =====\n")
edu_sec <- xtabs(OBS_VALUE ~ `Education level` + `Economic sector`, data = pacific, na.action = na.pass)
print(edu_sec)

cat("\n===== OCCUPATION BY ECONOMIC SECTOR =====\n")
occ_sec <- xtabs(OBS_VALUE ~ Occupation + `Economic sector`, data = pacific, na.action = na.pass)
print(occ_sec)

# 12. Visualizations
cat("\n===== GENERATING VISUALIZATIONS =====\n")

# Histogram of Observation Values
hist(pacific$`OBS_VALUE`, main = "Distribution of Employment Numbers", 
     xlab = "Number of Employed Persons", ylab = "Frequency", 
     col = "steelblue", breaks = 50, na.rm = TRUE)


# Boxplot: Employment by Education Level
boxplot(`OBS_VALUE` ~ `Education level`, data = pacific, 
        main = "Employment Distribution by Education Level", 
        ylab = "Number of Employed Persons", 
        col = "lightblue", las = 2)


# Boxplot: Employment by Occupation
boxplot(`OBS_VALUE` ~ Occupation, data = pacific, 
        main = "Employment Distribution by Occupation", 
        ylab = "Number of Employed Persons", 
        col = "lightcoral", las = 2)


# Boxplot: Employment by Economic Sector
boxplot(`OBS_VALUE` ~ `Economic sector`, data = pacific, 
        main = "Employment Distribution by Economic Sector", 
        ylab = "Number of Employed Persons", 
        col = "lightgreen", las = 2)


# Barplot: Employment by Country
country_totals <- tapply(pacific$`OBS_VALUE`, pacific$`Pacific Island Countries and territories`, 
                         function(x) sum(x, na.rm = TRUE))
barplot(sort(country_totals, decreasing = TRUE), 
        main = "Total Employment by Pacific Island Country", 
        ylab = "Number of Employed Persons", 
        col = "skyblue", las = 2)


# Barplot: Employment by Year
year_totals <- tapply(pacific$`OBS_VALUE`, pacific$Time, function(x) sum(x, na.rm = TRUE))
barplot(year_totals, 
        main = "Total Employment Over Time", 
        ylab = "Number of Employed Persons", 
        xlab = "Year",
        col = "lightcoral", las = 2)


# Barplot: Employment by Disability Status
disability_totals <- tapply(pacific$`OBS_VALUE`, pacific$Disability, 
                            function(x) sum(x, na.rm = TRUE))
barplot(disability_totals, 
        main = "Employment by Disability Status", 
        ylab = "Number of Employed Persons", 
        col = c("lightgreen", "lightyellow", "lightcoral"), las = 2)


# Top Education-Occupation combinations
top_combinations <- pacific %>%
  filter(!is.na(`OBS_VALUE`)) %>%
  group_by(`Education level`, Occupation) %>%
  summarise(Total = sum(`OBS_VALUE`, na.rm = TRUE), .groups = 'drop') %>%
  arrange(desc(Total)) %>%
  head(15)

combo_labels <- paste(top_combinations$`Education level`, "\n", 
                      top_combinations$Occupation, sep = "")
barplot(top_combinations$Total, names.arg = combo_labels,
        main = "Top 15 Education-Occupation Combinations", 
        ylab = "Total Employment", 
        col = "steelblue", las = 2, cex.names = 0.8)


# Top Education-Sector combinations
top_sector_combos <- pacific %>%
  filter(!is.na(`OBS_VALUE`)) %>%
  group_by(`Education level`, `Economic sector`) %>%
  summarise(Total = sum(`OBS_VALUE`, na.rm = TRUE), .groups = 'drop') %>%
  arrange(desc(Total)) %>%
  head(15)

sector_labels <- paste(top_sector_combos$`Education level`, "\n", 
                       top_sector_combos$`Economic sector`, sep = "")
barplot(top_sector_combos$Total, names.arg = sector_labels,
        main = "Top 15 Education-Sector Combinations", 
        ylab = "Total Employment", 
        col = "lightcoral", las = 2, cex.names = 0.8)

