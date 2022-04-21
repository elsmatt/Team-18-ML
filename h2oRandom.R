rm(list=ls())

setwd("C:/Users/rober/Desktop/ML2 - Project")

## Read Data
train_data <- read.csv("train.csv")
## REMOVE ID column
original = train_data[-c(1)]
data = train_data[-c(1)]

#hist(data$)

## TURN BLANKS INTO NA
data[data == ""] = NA

## Make target A FACTOR
data$target <- factor(data$target)

# IDENTIFY FEATURES AND TARGET
features = colnames(data)[colnames(data) != "target"]
target = colnames(data)[colnames(data) == "target"]

## TURNING CHARACTER VARIABLES INTO FACTORS
for (i in features) {
    if (class(data[[i]])=="character") {
        levels <- unique(c(data[[i]]))
        # assigning the unique levels
        data[[i]] <- as.integer(factor(data[[i]], levels=levels))
        # converting back from integer to factor
        data[[i]] = factor(data[[i]])
    }
}

str(data)

## NA IDENTIFICATION - make table with variable name, type, and NA count
na_count <- sapply(data, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
## add row names and type for each
na_count$name <- rownames(na_count)
na_count$type <- sapply(data,class)

## FACTOR VARIABLES WITH LARGE QUANTITIES OF NA's
# REMOVE v30  --- 60110 NA's
# REMOVE v113 --- 55304 NA's
# NEXT   v56  --- 6882  NA's

## REMOVE v30 and v113 from the data and na_count data.frame
data = data[, -which(names(data) %in% c("v30","v113"))]
na_count = na_count[!grepl('v30|v113', na_count$name),]

## IDENTIFY ALL FACTORS
factors = na_count$name[na_count$type == "factor"]
## Drop target from factors
factors = factors[-1]
## IDENTIFY ALL NUMERICS
numerics = na_count$name[na_count$type == "numeric"]

## ROUGH.FIX FACTOR VARIABLES
library(randomForest)
data[,factors] = na.roughfix(data[,factors])

## SCALE NUMERIC DATA
data[,numerics] = scale(data[,numerics],center=TRUE,scale=TRUE)

## IMPUTE MISSING VALUES USING PREDICTIVE MEAN MATCHING WITH mice PACKAGE
library(mice)
imputed_numerics <- mice(data[,numerics], m=1, maxit = 1, method = 'pmm', seed = 500)
# densityplot(imputed_numerics)
completeData <- complete(imputed_numerics,1)
data[,numerics]<-completeData
dim(data)

#na_count <- sapply(data, function(y) sum(length(which(is.na(y)))))
#na_count <- data.frame(na_count)
#na_count$name <- rownames(na_count)
#na_count$type <- sapply(data,class)

## LOAD & INITIALIZE h2o PACKAGE
library(h2o)
h2o.init()

## MAKE h2o DATA
data.hex = as.h2o(data)
y = "target"
x = setdiff(names(data),y)

## SPLITTING TO TRAIN AND TEST
splits = h2o.splitFrame(data = data.hex,
                        ratios = c(0.8),
                        seed = 1)

## SPLITTING AND PRINTING THE LENGTH
train = splits[[1]]   # 80%
test = splits[[2]]    # 20%
h2o.nrow(train)
h2o.nrow(test)

## DEFINING randomForest MODEL
rf = h2o.randomForest(x = x, y = y,
                      ntrees = 100,                     # Number of trees
                      max_depth = 12,                   # Maximum tree depth
                      min_rows = 10,                    # Number of observations for a leaf
                      calibrate_model = TRUE,           # Use Platt scaling to calculate calibrated class probabilities
                      calibration_frame = test,         # Frame to be used for Platt scaling
                      binomial_double_trees = TRUE,     # Build twice as many trees (one per class)
                      training_frame = train,           # Training Data
                      validation_frame = test,          # Validation Data
                      col_sample_rate_per_tree = 0.1)   # Column sample rate per tree

# BEST MODEL PERFORMANCE
# Evaluate the model performance on a test set so we get an honest estimate of top model performance
perf = h2o.performance(rf, newdata = test)
print(perf)

# LOG LOSS
h2o.logloss(perf)

## CONFUSION MATRIX
h2o.confusionMatrix(perf)

## ROC CURVE
plot(perf, type = "roc")