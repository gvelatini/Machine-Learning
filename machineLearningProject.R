# Background
# 
# Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large
# amount of data about personal activity relatively inexpensively. These type of devices are part of the
# quantified self movement - a group of enthusiasts who take measurements about themselves regularly 
# to improve their health, to find patterns in their behavior, or because they are tech geeks. 
# One thing that people regularly do is quantify how much of a particular activity they do, 
# but they rarely quantify how well they do it. In this project, your goal will be to use data from 
# accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform 
# barbell lifts correctly and incorrectly in 5 different ways. More information is available from the 
# website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 
# 
# Data
# 
# The training data for this project are available here:
#   
#   https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
# 
# The test data are available here:
#   
#   https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

library(AppliedPredictiveModeling)
library(caret)
library(pgmm)
require(ggplot2)
require(dplyr)


setwd("C:/Users/Gr/Documents/coursera/Machine Learning/week4")

# read in training and test data

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "training.csv", method = "auto", quiet=FALSE)
inTrain <- read.csv("training.csv", na.strings = c("NA", ""))

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "testing.csv", method = "auto", quiet=FALSE)
inTest <- read.csv("testing.csv", na.strings = c("NA", ""))


View(inTrain)

inTrain <- inTrain[,7:160]
inTest <- inTest[,7:160]

#  remove mostly empty variables

sum(is.na(inTrain$kurtosis_roll_belt))

goodCols <- apply(is.na(inTrain),2,sum) < 19200

inTrain <- inTrain[, goodCols]
ncol(inTrain)

inTest <- inTest[, goodCols]
ncol(inTest)

#  partition into smaller data set

partTrain <- createDataPartition(y=inTrain$classe,p=0.3,list=FALSE)
inTrainSmall <- inTrain[partTrain,] 

valDat <- inTrain[-partTrain,]

# create random forest model

modelRF <- train(classe~.,data=inTrainSmall, method="rf", trControl=trainControl(method="cv",number=5), prox=TRUE,allowParallel=TRUE)
print(modelRF)

print(modelRF$finalModel)

# estimate test set error rate using validation data

valPred <- predict(modelRF,valDat)
confusionMatrix(valPred,valDat$classe)



