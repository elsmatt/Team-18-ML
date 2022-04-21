## Specify Hyper Parameters
gbm_params = list(learn_rate = c(0.01,0.02,0.03,0.05),
                  max_depth = c(5,10,12,15,20),
                  sample_rate = c(0.2,0.3,0.5,0.7,0.8),
                  col_sample_rate = c(0.1,0.2,0.25,0.3,0.5))

## Train and validate a grid of GBM's
gbm_grid_demo = h2o.grid("gbm", x = x, y = y,
                          grid_id = "gbm_grid_demo",
                          training_frame = train,
                          validation_frame = test,
                          ntrees = 100,
                          seed = 1,
                          hyper_params = gbm_params)

## Identify Model Parameters and Results
gbm_gridperf = h2o.getGrid(grid_id = "gbm_grid_demo",
                           sort_by = "auc",
                           decreasing = TRUE)

## GRID PERFORMANCE SORTED BY VALIDATION AUC
print(gbm_gridperf)

## IDENTIFY BEST MODEL
best_gbm = h2o.getModel(gbm_gridperf@model_ids[[1]])