library(randomForest)
set.seed(35)

train <- read.csv("../input/train.csv")
test <- read.csv("../input/test.csv")

# Use only numeric cols
train <- train[, sapply(train, is.numeric)]
test <- test[, sapply(test, is.numeric)]

train <- na.roughfix(train)
test <- na.roughfix(test)

rf <- randomForest(as.factor(target) ~ ., data = train, ntree = 200)
yhat <- predict(rf, test, type="prob")[,2]

write.csv(data.frame(ID = test$ID, PredictedProb = yhat), "random_forest_benchmark.csv", row.names = F)