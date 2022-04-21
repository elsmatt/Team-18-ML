## BNP Paribas competition - initial script
## Author: Aleksandr Beloushkin, Voronezh, Russian Federation. 


## Read Data
library(readr)
train_data <- read_csv("../input/train.csv")
test_data  <- read_csv("../input/test.csv")

## Putting NA's to target column in test dataset
test_data$target <- rep(NA,nrow(test_data))

## Binding train and test datasets into one for further preporcessing. 

## first store target column
train_target <- train_data$target
## store test Id column and remove it from the train and test data
test_Id <- test_data$ID
train_data$ID <- test_data$ID <- NULL
## marking train and test data with additional variable and binding them
train_data$isTrain <- rep(TRUE,nrow(train_data))
test_data$isTrain <- rep(FALSE,nrow(test_data))
train_test <- rbind(train_data, test_data)

## making factors and strings being integer values
feature.names <- names(train_test)[names(train_test) != "target"]
for (f in feature.names) {
    if (class(train_test[[f]])=="character") {
        levels <- unique(c(train_test[[f]]))
        train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
    }
}

## NA imputing, first make table showing how mucn NAs we have
na_count <-sapply(train_test, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count$name <- rownames(na_count)

## dividing between 2 classes - where are a lot of NA's and where are not many...
bigna_names <- na_count[na_count$na_count > 15000,]$name
bigna_names <- bigna_names[bigna_names != "target"]
lowna_names <- na_count[(na_count$na_count > 0) & (na_count$na_count <= 15000),]$name

## using maximum values to impute into columns containing too many NA's 
for (i in bigna_names) {
    maxval = max(train_test[, i], na.rm = TRUE);
    train_test[, i][is.na(train_test[, i])] <- -maxval
}

## using median values for imputing into other cols containing NA's 

library(caret)
pre<-preProcess(train_test[,lowna_names], method="medianImpute",na.remove = TRUE)
train_test[,lowna_names] <-predict(pre, train_test[,lowna_names])
unloadNamespace("caret")

## return to divided train and test sets 

cols_to_drop = c("isTrain")
train <- train_test[train_test$isTrain == TRUE,!names(train_test) %in% cols_to_drop]
test <- train_test[train_test$isTrain == FALSE,!names(train_test) %in% cols_to_drop]
feature.names <- feature.names[feature.names != "isTrain"]


## Making a model
library(xgboost)

train_target <- train$target
train_eval   <- train[,feature.names]

## Making a small validation set to analyze progress
h <-sample(nrow(train),2000)

dval   <-xgb.DMatrix(data=data.matrix(train_eval[h,]),label=train$target[h])
dtrain <-xgb.DMatrix(data=data.matrix(train_eval[-h,]),label=train$target[-h])

cat("start training a model \n")
set.seed(3322)
xgb_watchlist <-list(val=dval,train=dtrain)
xgb_params <- list(  objective           = "binary:logistic", 
                     booster = "gbtree",
                     eval_metric = "auc",
                     eta                 = 0.018, 
                     max_depth           = 6, #changed from default of 8
                     subsample           = 0.48, # 0.7
                     colsample_bytree    = 0.59 # 0.7
)

xgb_model <- xgb.train(
    params              = xgb_params, 
    data                = dtrain, 
    nrounds             = 401, # change to 1500 to run outside of kaggle
    verbose             = 1,  #0 if full training set and no watchlist provided
    watchlist           = xgb_watchlist,
    print.every.n       = 100,
    maximize            = FALSE
)


predictions <- predict(xgb_model, data.matrix(test[,feature.names]))
submission  <- data.frame(ID=test_Id, PredictedProb=round(predictions,digits = 6))
cat("saving the submission file\n")
write_csv(submission, "paribas_submission.csv")