#9.7 In this problem, you will use support vector approaches in order to
# predict whether a given car gets high or low gas mileage based on the
# Auto data set.
rm(list=ls())

# a)Create a binary variable that takes on a 1 for cars with gas
# mileage above the median, and a 0 for cars with gas mileage
# below the median.
library(ISLR)# for Auto
library(dplyr)
library(e1071)
set.seed(1)
data(Auto)

y<-ifelse(Auto$mpg > median(Auto$mpg), 1, 0)#binary variable
newdf <- data.frame(Auto, y = as.factor(y))
newdf <- newdf[,-c(1,9)]

# b)Fit a support vector classifier to the data with various values
# of cost, in order to predict whether a car gets high or low gas
# mileage. Report the cross-validation errors associated with different 
# values of this parameter. 
# Comment on your results. 
# Note you will need to fit the classifier without the gas mileage variable
# to produce sensible results.

set.seed(1)
tune.out <- tune(svm, y ~., data = newdf, kernel = "linear",
                    ranges = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
summary(tune.out)
tune.out$best.parameters
#cost=1
tune.out$best.performance
#0.08435897
# cross-validation error was minimized when cost equals 1


# c)Now repeat (b), this time using SVMs with radial and 
# polynomial basis kernels, with different values of gamma and degree and
# cost. Comment on your results.

#using SVMs with radial basis kernels
set.seed(1)
radial_tune <- tune(svm, y~., data=newdf, kernel='radial',
                    ranges = list(cost = c(.001, .01, .1, 1, 5, 10, 100,1000),
                    gamma = c(0.5, 1, 2, 3, 4)))
radial_tune$best.parameters
#      cost gamma
# 12    1     1
radial_tune$best.performance
# 0.06634615
# As we can see from the output, the training CV error is minimized for a 
# radial model at cost=1 and gamma=1. 


#using SVMs with polynomial basis kernels
set.seed(1)
poly_tune<-tune(svm, y~., data=newdf, kernel='polynomial',
                 ranges = list(cost = c(.001, .01, .1, 1, 5, 10, 100,1000),
                               degree = c(1,2,3,4,5)))
poly_tune$best.parameters
#    cost degree
# 23  100      3
poly_tune$best.performance
#[1] 0.08423077

# As we can see from the output, the training CV error is minimized for 
# a polynomial model at cost=100 and degree=3. In addition, the training CV error 
# is better than that of the linear kernel model but worse than that of
# the radial kernel model, which suggested the true 
# decision boundary is non-linear.


# d)Make some plots to back up your assertions in (b) and (c).

# Hint: In the lab, we used the plot() function for svm objects
# only in cases with p = 2. 
# When p > 2, you can use the plot() function to create plots displaying pairs of variables at a time.
# Essentially, instead of typing

##plot (svmfit , dat)
# where svmfit contains your fitted model and dat is a data frame
# containing your data, you can type

##plot (svmfit , dat , x1 ~ x4)
# in order to plot just the first and fourth variables. However, you
# must replace x1 and x4 with the correct variable names. To find
# out more, type ?plot.svm.


svmfit_linear <- svm(y~., data=newdf, kernel="linear", cost=1)
svmfit_radial <- svm(y~., data=newdf, kernel="radial", cost=1, gamma=1)
svmfit_polynomial <- svm(y~., data=newdf, kernel="polynomial", cost=100, degree=3)
names_list <- names(newdf);names_list 

#linear plot
plot(svmfit_linear, newdf, displacement~weight)

#radial plot
plot(svmfit_radial, newdf, displacement~weight)

#polynomial plot
plot(svmfit_polynomial, newdf, displacement~weight)

