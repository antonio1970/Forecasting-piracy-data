library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(corrplot)
library(ROCR)
library(tree)
library(randomForest)
library(rstanarm)
library(pROC)
library(VIM)
library(dvmisc)
library(rcompanion)

data_piracy<-data2
View(data_piracy)
anyNA(data_piracy)
table(is.na(data_piracy))
prop.table(table(is.na(data_piracy))) * 100



install.packages("mice")
library(mice)  
md.pattern(data_piracy)

mice_plot <- aggr(data_piracy, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(data_piracy), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))


mydta<-kNN(data_piracy,k=5)

anyNA(mydta)
table(is.na(mydta))
prop.table(table(is.na(mydta))) * 100

mice_plot <- aggr(mydta, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(mydta), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))




mydta1<-data.frame(mydta$cbus,mydta$employ,mydta$educ,mydta$infla,mydta$taxret,mydta$trade,mydta$corrupt,mydta$voice,mydta$rule,mydta$piracy)

library(dummies)

# example data
df1 <- data.frame(id = mydta$id, year = mydta$time)
df1 <- cbind(df1, dummy(df1$year, sep = "_"))


mydta2<-data.frame(mydta1,df1)
write.dta(mydta2,"piracy.dta")


set.seed(100)  # setting seed to reproduce results of random sampling
trainingRowIndex <- sample(1:nrow(piracy), 0.8*nrow(piracy))  # row indices for training data
trainingData <- piracy[trainingRowIndex, ]  # model training data
testData  <- piracy[-trainingRowIndex, ]   # test data


library(Matrix)
cat(rankMatrix(trainingData), "\n")    
cat(rankMatrix(testData), "\n") 





lmMod <- lm(trainingData$mydta_piracy ~., data=trainingData)  # build the model
summary(lmMod)
print(lmMod)
lmPred <- predict(lmMod, testData)
plot(lmMod)



actuals_preds <- data.frame(cbind(actuals=testData$mydta_piracy, predicteds=lmPred))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)  
head(actuals_preds)
min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))  
print(min_max_accuracy)
summary(min_max_accuracy)
mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)  
print(mape)
DMwR::regr.eval(actuals_preds$actuals, actuals_preds$predicteds)
RMSE(lmPred,testData$mydta_piracy)




trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3444)

#SVM Radial
svm_Radial <- train(mydta_piracy~., data = trainingData, method = "svmRadial",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
svm_Radial
plot(svm_Radial)
test_pred1 <- predict(svm_Radial, newdata = testData, type='raw')
actuals_preds1 <- data.frame(cbind(actuals=testData$mydta_piracy, predicteds=test_pred1))  # make actuals_predicteds dataframe.
correlation_accuracy1 <- cor(actuals_preds1)  
head(actuals_preds1)
min_max_accuracy1 <- mean(apply(actuals_preds1, 1, min) / apply(actuals_preds1, 1, max))  
mape1 <- mean(abs((actuals_preds1$predicteds - actuals_preds1$actuals))/actuals_preds1$actuals)  
DMwR::regr.eval(actuals_preds1$actuals, actuals_preds1$predicteds)
RMSE(test_pred1,testData$mydta_piracy)

#SVM Linear

svm_Linear <- train(mydta_piracy~., data = trainingData, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
svm_Linear
plot(svm_Linear,trainingData$mydta_piracy)
test_pred2 <- predict(svm_Linear, newdata = testData)

actuals_preds2 <- data.frame(cbind(actuals=testData$mydta_piracy, predicteds=test_pred2))  # make actuals_predicteds dataframe.
correlation_accuracy2 <- cor(actuals_preds2)  
head(actuals_preds2)
min_max_accuracy2 <- mean(apply(actuals_preds2, 1, min) / apply(actuals_preds2, 1, max))  
mape2 <- mean(abs((actuals_preds2$predicteds - actuals_preds2$actuals))/actuals_preds2$actuals)  
DMwR::regr.eval(actuals_preds2$actuals, actuals_preds2$predicteds)
RMSE(test_pred2,testData$mydta_piracy)





