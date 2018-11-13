library(tree)
library(ISLR)

# attaching dataframe
attach(Carseats)
High = ifelse(Sales <= 8, "No", "Yes")
Carseats = data.frame(Carseats, High)
tree.carseats = tree(High~.-Sales, Carseats)
summary(tree.carseats)
# Classification tree:
#   tree(formula = High ~ . - Sales, data = Carseats)
# Variables actually used in tree construction:
#   [1] "High.1"
# Number of terminal nodes:  2 
# Residual mean deviance:  0 = 0 / 398 
# Misclassification error rate: 0 = 0 / 400 

# plotting the tree
plot(tree.carseats)
text(tree.carseats, pretty = 0)

# printing branches and split criteria
tree.carseats

# train test split
set.seed(2)
train = sample(1:nrow(Carseats), 200)

Carseats.test = Carseats[-train, ]
High.test = High[-train]

tree.carseats = tree(High ~ . - Sales, Carseats, subset = train)
tree.pred =predict(tree.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
mean(tree.pred == High.test)

# pruning the tree using cv
set.seed(3)
cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)
cv.carseats

# plotting the cross validation
par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")

# now applying pruning
prune.carseats = prune.misclass(tree.carseats, best = 9)
plot(prune.carseats)
text(prune.carseats, pretty = 0)

tree.pred = predict(prune.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
mean(tree.pred == High.test)

# playing with the bst number to obtain larger tree
prune.carseats = prune.misclass(tree.carseats, best = 15)
plot(prune.carseats)
text(prune.carseats, pretty = 0)
tree.pred = predict(prune.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
mean(tree.pred == High.test)
# so the larger tree has lwoer prediction accuracy


# Regression Trees --------------------------------------------------------

library(MASS)
Boston = Boston
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston = tree(medv ~ ., Boston, subset = train)
summary(tree.boston)

# plotting the tree
plot(tree.boston)
text(tree.boston, pretty = 0)

# now doing cross validation
cv.boston = cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type =  'b')

# pruning a tree
prune.boston = prune.tree(tree.boston, best = 5)
plot(prune.boston)
text(prune.boston, pretty = 0)

# using unpruned tree to make predictions
yhat = predict(tree.boston, newdata = Boston[-train, ])
boston.test = Boston[-train, 'medv']
plot(yhat, boston.test)
abline(0, 1)
mean((yhat - boston.test)^2)


# Random Forest -----------------------------------------------------------

library(caret)
set.seed(1)
bag.boston = randomForest(medv~., data = Boston, subset = train, mtry = 13, importance = TRUE)
bag.boston

yhat.bag = predict(bag.boston, newdata = Boston[-train, ])
plot(yhat.bag, boston.test)
mean((yhat.bag - boston.test)^2)

# now building the random forest
set.seed(1)
rf.boston = randomForest(medv~., data = Boston, subset = train, mtry = 6, importance = TRUE)
yhat.rf = predict(rf.boston, newdata = Boston[-train, ])
mean((yhat.rf - boston.test)^2)

# now getting the importance
importance(rf.boston)

# plotting this
varImpPlot(rf.boston)


# Boosting ----------------------------------------------------------------

library(gbm)
set.seed(1)
boost.boston = gbm(medv~., data = Boston[train, ], distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
summary(boost.boston)

# now looking at the partial dependence to see how the underlying variable is linked to the response
par(mfrow = c(1, 2))
plot(boost.boston, i = 'rm')
plot(boost.boston, i = 'lstat')

# changing the shrinkage parameter from default 0.001
boost.boston = gbm(medv~., data = Boston[train, ], distribution = "gaussian", n.trees =5000, 
                   interaction.depth = 4, shrinkage = 0.2, verbose =F)
yhat.boost = predict(boost.boston, newdata = Boston[-train, ], n.trees = 5000)
mean((yhat.boost - boston.test)^2)

