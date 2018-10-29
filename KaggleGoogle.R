# Google Prediction Kaggle

library(lubridate)
library(magrittr)
library(jsonlite)
library(data.table)
library(ggplot2)
library(fastDummies)
library(glmnet)

# Import
trainread <- read.csv('train.csv')
testread <- read.csv('test.csv')

train <- trainread
test <- testread

# Format dates/times
train$date <- as.Date(as.character(train$date), format='%Y%m%d')
train$visitStartTime <- as_datetime(train$visitStartTime)

test$date <- as.Date(as.character(test$date), format='%Y%m%d')
test$visitStartTime <- as_datetime(test$visitStartTime)

# Parse JSON
tr_device <- paste("[", paste(train$device, collapse = ","), "]") %>% fromJSON(flatten = T)
tr_geoNetwork <- paste("[", paste(train$geoNetwork, collapse = ","), "]") %>% fromJSON(flatten = T)
tr_totals <- paste("[", paste(train$totals, collapse = ","), "]") %>% fromJSON(flatten = T)
tr_trafficSource <- paste("[", paste(train$trafficSource, collapse = ","), "]") %>% fromJSON(flatten = T)
train <- cbind(train, tr_device, tr_geoNetwork, tr_totals, tr_trafficSource) %>% as.data.table()

te_device <- paste("[", paste(test$device, collapse = ","), "]") %>% fromJSON(flatten = T)
te_geoNetwork <- paste("[", paste(test$geoNetwork, collapse = ","), "]") %>% fromJSON(flatten = T)
te_totals <- paste("[", paste(test$totals, collapse = ","), "]") %>% fromJSON(flatten = T)
te_trafficSource <- paste("[", paste(test$trafficSource, collapse = ","), "]") %>% fromJSON(flatten = T)
test <- cbind(test, te_device, te_geoNetwork, te_totals, te_trafficSource) %>% as.data.table()

# Drop old JSON columns
train[, c('device', 'geoNetwork', 'totals', 'trafficSource') := NULL]

test[, c('device', 'geoNetwork', 'totals', 'trafficSource') := NULL]

# Identify and remove various NA types
na_vals <- c('unknown.unknown', '(not set)', 'not available in demo dataset', 
             '(not provided)', '(none)', '<NA>')
for(col in names(train)) {
  set(train, i=which(train[[col]] %in% na_vals), j=col, value=NA)
}
for(col in names(test)) {
  set(test, i=which(train[[col]] %in% na_vals), j=col, value=NA)
}

#Remove varaibles where all values are the same
unique <- sapply(train, function(x) { length(unique(x[!is.na(x)])) })
one_val <- names(unique[unique <= 1])
one_val = setdiff(one_val, c('bounces', 'newVisits'))
train[, (one_val) := NULL]

unique <- sapply(test, function(x) { length(unique(x[!is.na(x)])) })
one_val <- names(unique[unique <= 1])
one_val = setdiff(one_val, c('bounces', 'newVisits'))
test[, (one_val) := NULL]

#Define numeric columns, set as type
num_cols1 <- c('hits', 'pageviews', 'bounces', 'newVisits',
              'transactionRevenue')
num_cols2 <- c('hits', 'pageviews', 'bounces', 'newVisits')
     
train[, (num_cols1) := lapply(.SD, as.numeric), .SDcols=num_cols1]
test[, (num_cols2) := lapply(.SD, as.numeric), .SDcols=num_cols2]

#--------------------------------------
#Plots of data

train %>% 
  ggplot(aes(x=log(transactionRevenue), y=..density..)) + 
  geom_histogram(fill='steelblue', na.rm=TRUE, bins=40) + 
  geom_density(aes(x=log(transactionRevenue)), fill='orange', color='orange', alpha=0.3, na.rm=TRUE) + 
  labs(
    title = 'Distribution of transaction revenue',
    x = 'Natural log of transaction revenue'
  )

data.table(
  pmiss = sapply(train, function(x) { (sum(is.na(x)) / length(x)) }),
  column = names(train)
) %>%
  ggplot(aes(x = reorder(column, -pmiss), y = pmiss)) +
  geom_bar(stat = 'identity', fill = 'steelblue') + 
  scale_y_continuous(labels = scales::percent) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  labs(
    title='Missing data by feature',
    x='Feature',
    y='% missing')

#-----------------------------------

# Define variables to use for regression
vars <- c('transactionRevenue', 'channelGrouping', 'visitNumber', 'browser', 'operatingSystem', 'deviceCategory', 'subContinent', 'hits', 'pageviews', 'bounces', 'newVisits')
vars2 <- c('channelGrouping', 'visitNumber', 'browser', 'operatingSystem', 'deviceCategory', 'subContinent', 'hits', 'pageviews', 'bounces', 'newVisits')

# Solve NA values for certain columns
trainmod <- train[,vars,with=FALSE]
trainmod[is.na(get('bounces')), ('bounces'):=0]
trainmod[is.na(get('newVisits')), ('newVisits'):=0]
trainmod[is.na(get('pageviews')), ('pageviews'):=1]

testmod <- test[,vars2,with=FALSE]
testmod[is.na(get('bounces')), ('bounces'):=0]
testmod[is.na(get('newVisits')), ('newVisits'):=0]
testmod[is.na(get('pageviews')), ('pageviews'):=1]

# Create dummy variables, apply log function to all Y vars where a transaction existed. Set all other transaction revenues to 0
traindummy <- fastDummies::dummy_cols(trainmod, select_columns = c('operatingSystem','subContinent','browser','channelGrouping','deviceCategory'))
cat <- c('operatingSystem', 'subContinent', 'browser','channelGrouping','deviceCategory')
traindummyclean <- traindummy[, -cat,with=FALSE]
traindummyclean$transactionRevenue <- log(traindummyclean$transactionRevenue)
traindummyclean[is.na(get('transactionRevenue')), ('transactionRevenue'):=0]

testdummy <- fastDummies::dummy_cols(testmod, select_columns = c('operatingSystem','subContinent','browser','channelGrouping','deviceCategory'))
cat <- c('operatingSystem', 'subContinent', 'browser','channelGrouping','deviceCategory')
testdummyclean <- testdummy[, -cat,with=FALSE]

traindummyclean <- data.frame(traindummyclean)
testdummyclean <- data.frame(testdummyclean)

# Solve problem of categorical variable types appearing in test but not train set and vice versa.
for (col in names(traindummyclean)){
  if (!(col %in% names(testdummyclean))){
    print(col)
    testdummyclean[col] <- 0
  }
}

for (col in names(testdummyclean)){
  if (!(col %in% names(traindummyclean))){
    print(col)
    traindummyclean[col] <- 0
  }
}
#------------------------------------
# Set Regression Model

setDT(traindummyclean)
setDT(testdummyclean)

testdummyclean <- testdummyclean[,-c('transactionRevenue')]

#Use random subset of train set (1/8) because computer memory too small
trainsample <- traindummyclean[sample(nrow(traindummyclean),nrow(traindummyclean)/8)]

#Set X, Y vars
X = as.matrix(trainsample[,-c('transactionRevenue'),with=FALSE])
Y = as.matrix(trainsample$transactionRevenue)

#Run Lasso 
lasso <- cv.glmnet(X, Y)
plot(lasso)
lasso$lambda.min
#1700 MSE
coef(lasso, s = 'lambda.min')


#Prep test set for prediction - not done yet - errors with predict function and memory
newX <- model.matrix(~.,data=testdummyclean[0:200000,])
preds <- predict(lasso, newx = newX, s = 'lambda.min')

#Histogram of training set transaction revenue for reference to compare to predictions
#1.29% purchase rate (also for later reference to compare to predictions)
hist(subset(traindummyclean$transactionRevenue,traindummyclean$transactionRevenue > 0))


