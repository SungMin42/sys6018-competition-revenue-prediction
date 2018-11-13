library(tidyverse)
library(dplyr)
library(magrittr)
library(Amelia)
library(jsonlite)

# Reading in the data
# we take note that the total dataset is approximately 25 GB
# we will work with a 1 GB subset of this data for now
setwd("../")
df.train = read_csv("data/train_old.csv")
train.device = paste("[", paste(df.train$device, collapse = ","), "]") %>% fromJSON(flatten = T)
train.geoNetwork = paste("[", paste(df.train$geoNetwork, collapse = ","), "]") %>% fromJSON(flatten = T)
train.totals = paste("[", paste(df.train$totals, collapse = ","), "]") %>% fromJSON(flatten = T)
train.trafficSource = paste("[", paste(df.train$trafficSource, collapse = ","), "]") %>% fromJSON(flatten = T)
df.train = cbind(df.train, train.device, train.geoNetwork, train.totals, train.trafficSource)
df.train[, c('device', 'geoNetwork', 'totals', 'trafficSource')] = NULL
df.train$transactionRevenue[is.na(df.train$transactionRevenue)] = 0
df.train1 = df.train

# Create artificial train test dataset to make sure that dataset has all factors contained in 
# test
# Read in test dataset
df.test = read_csv("data/test_new.csv")
test.device = paste("[", paste(df.test$device, collapse = ","), "]") %>% fromJSON(flatten = T)
test.geoNetwork = paste("[", paste(df.test$geoNetwork, collapse = ","), "]") %>% fromJSON(flatten = T)
test.totals = paste("[", paste(df.test$totals, collapse = ","), "]") %>% fromJSON(flatten = T)
test.trafficSource = paste("[", paste(df.test$trafficSource, collapse = ","), "]") %>% fromJSON(flatten = T)
df.test = cbind(df.test, test.device, test.geoNetwork, test.totals, test.trafficSource)
df.test[, c('device', 'geoNetwork', 'totals', 'trafficSource')] = NULL
df.test1 = df.test

str(df.test1)
index = -which(colnames(df.train1) %in% colnames(df.test1))
index2 = -which(colnames(df.test1) %in% colnames(df.train1))
colnames(df.train1)[index]
colnames(df.test1)[index2]

# colnames(df.train1)[order(colnames(df.train1))] ==  colnames(df.test1)[order(colnames(df.test1))]

# so the campaigncode variable is not present in the test
# we will comment this out
df.train1$campaignCode = NULL
df.train1$sessionId = NULL
# likewise [1] "customDimensions"        "timeOnSite"              "sessionQualityDim"      
# "transactions"            "totalTransactionRevenue" are in test but not in train so NULL these
df.test1[, colnames(df.test1)[index2]] = NULL
df.test1$transactionRevenue = NULL
df.test1$hits = NULL

#   Artificially created target variable so ncol matches
df.test1 = add_column(df.test1, transactionRevenue = rep(-1, nrow(df.test1)))

#   Indicator variable for both train and test
df.train1 = add_column(df.train1, TrainInd = rep(1, nrow(df.train1)))
df.test1 = add_column(df.test1, TrainInd = rep(0, nrow(df.test1)))

#   creating train test table
df.train.test = rbind(df.train1, df.test1)

# Finding variables with missing values
missmap(df.train.test, main = "Missing Map")
str(df.train.test)

# setting NA equivalents to NA
na_vals <- c('unknown.unknown', '(not set)', 'not available in demo dataset', 
             '(not provided)', '(none)', '<NA>')

for(col in 1:ncol(df.train.test)) {
  df.train.test[which(df.train.test[, col] %in% na_vals), col] = NA
}

head(df.train.test, n = 5)

# Checking if there are missing values
missing = colMeans(is.na(df.train.test))

# we will now subset to columns that do not have missing variables
missing.columns.index = which(missing == 0)
df.train.test = df.train.test[, missing.columns.index]

# setting variables without useful information to NULL
df.train.test$date = NULL
df.train.test$visits = NULL
df.train.test$sessionId = NULL
df.train.test$isMobile = NULL
df.train.test$hits = NULL
df.train.test$socialEngagementType = NULL

# setting transactionRevenue and hits variables to numeric
df.train.test$transactionRevenue = as.numeric(df.train.test$transactionRevenue)

# Getting list of character variables
str(df.train.test)

factor.variables = sapply(df.train.test, class)
factor.variables = factor.variables[factor.variables == "character"]

# Class change for factor variables
df.train.test.dummy = df.train.test[, names(factor.variables)]
df.train.test[, names(factor.variables)] = lapply(df.train.test.dummy, factor)
str(df.train.test)

# Rechecking if there are any missing values
table(is.na(df.train.test))
# So there are no missing variables

# Now that schema and features are consistent, split back into train and test
df.train1 = df.train.test %>% filter(TrainInd == 1)
df.test1 = df.train.test %>% filter(TrainInd == 0)

# Dropping Survived and Train Ind from df.test1
df.test1 = df.test1 %>% select(-TrainInd, -transactionRevenue)

# Dropping TrainInd from df.train and df.test1
df.train1 = df.train1 %>% select(-TrainInd)

# performing log transformation on response
df.train1$transactionRevenue = exp(df.train1$transactionRevenue)
str(df.train1)
