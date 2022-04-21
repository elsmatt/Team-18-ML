rm(list=ls())

## Start time
start_time = Sys.time()

setwd("C:/Users/rober/Desktop/ML2 - Project")

## Read Data
train_data <- read.csv("train.csv")
## REMOVE ID column
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

table(na_count$type)
## FACTOR VARIABLES WITH LARGE QUANTITIES OF NA's
# REMOVE v30  --- 60110 NA's
# REMOVE v113 --- 55304 NA's

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

## DEFINE HYPER PARAMETERS
gbm_params = list(learn_rate = c(0.05),             # Learning rate  (0-1), same as shrinkage, penalizing importance of iterations
                  max_depth = c(12),                # maximum tree depth
                  sample_rate = c(0.8),             # Row sample rate per tree  (0-1)
                  col_sample_rate = c(0.1))         # Column sample rate   (0-1)

## TRAIN & VALIDATE GBM
gbm_final_grid = h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_final_grid",
                      training_frame = train,
                      validation_frame = test,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params)

## IDENTIFY MODEL PARAMETERS & RESULTS
gbm_gridperf = h2o.getGrid(grid_id = "gbm_final_grid",
                           sort_by = "auc",
                           decreasing = TRUE)

## GRID PERFORMANCE SORTED BY VALIDATION AUC
print(gbm_gridperf)

## IDENTIFY BEST MODEL
best_gbm = h2o.getModel(gbm_gridperf@model_ids[[1]])

## BEST MODEL PERFORMANCE
## Evaluate the model performance on a test set so we get an honest estimate of top model performance
best_gbm_perf = h2o.performance(model = best_gbm, newdata=test)
print(best_gbm_perf)

## LOG LOSS
h2o.logloss(best_gbm_perf)

## CONFUSION MATRIX
h2o.confusionMatrix(best_gbm_perf)

## ROC CURVE
plot(best_gbm_perf, type = "roc")

## VARIABLE IMPORTANCE
varimp = h2o.varimp(best_gbm)
print(varimp)
h2o.varimp_plot(best_gbm)

## LEARNING CURVE
plot(best_gbm, timestep = "number_of_trees", metric = "auc")
h2o.varimp(best_gbm)

## End time and Duration
end_time = Sys.time()
duration = end_time - start_time