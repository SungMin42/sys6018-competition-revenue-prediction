# loading required libraries
library(tidyverse)
library(magrittr)
library(randomForest)
library(caret)
library(Matrix)
library(xgboost)

# subsetting train and test based on non.zero.coef
df.train1$fullVisitorId = NULL

fullVisitorId = as.character(df.test1$fullVisitorId)
df.test1$fullVisitorId = NULL

# fitting a model
subset= sample(nrow(df.train1), nrow(df.train1)/10)

rf = randomForest(transactionRevenue~., data = df.train1, subset = subset, mtry = round(sqrt(ncol(df.train1))), n.tree = 30)
importance(rf)
str(df.train1)


# refitting sparse matrices
str(df.train1)

x.sparse.train = sparse.model.matrix(~., data = df.train1)
y.train = df.train1$transactionRevenue

x.sparse.test = sparse.model.matrix(~., data = df.test1)
str(df.train1)

# xgboost modelling with cross validation
bst.cv <- xgb.cv(data = x.sparse.train, label = y.train, nfold = 5,
                 max_depth = 2, 
                 nrounds = 1000, 
                 objective = "reg:linear",
                 early_stopping_rounds = 50)

# So the best model seems to be at iteration 382

# training this model
bst2 = xgboost(data = x.sparse.train, label = y.train,
               max_depth = 2, 
               nrounds = 382, 
               objective = "reg:linear")

predictions = predict(bst2, df.test1)
write.csv(cbind(fullVisitorId, predictions), "prediction rf old.csv")
