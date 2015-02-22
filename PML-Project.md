# Practical Machine Learning - Course Project
Sai krishnan  
Saturday, February 21, 2015  
##Background  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. In this project, data from accelerometers on the belt, forearm, arm, and dumbell of the 6 participants was collected. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Our goal is to build a model based on sensor data to predict whether the lift was performed correctly or not. 

Data Sources  

Training dataset: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
Test dataset: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  

Outcome variable: "classe"  
- lift exactly according to the specification (category 'A')  
- throwing the elbows to the front            (category 'B')  
- lifting the dumbbell only halfway           (category 'C')  
- lowering the dumbbell only halfway          (category 'D')  
- throwing the hips to the front              (category 'E')  

## Data Cleaning & Exploratory Analysis  
  

```r
# Load the training and test datasets from working directory; replace missing values with 'NA'
harTrain <- read.csv("C:/Users/Admin/Documents/Practical Machine Learning/Project/pml-training.csv",header=TRUE,na.strings=c("NA","")) 
harTest <- read.csv("C:/Users/Admin/Documents/Practical Machine Learning/Project/pml-testing.csv",header=TRUE,na.strings=c("NA",""))

# Delete columns with missing values
harTrain <- harTrain[,colSums(is.na(harTrain)) == 0]
harTest  <-  harTest[,colSums(is.na(harTest))  == 0]

# Since we want to predict the type of lift using only activity monitor data, 
# the other variables (in columns 1 though 7) are eliminated from the datasets  
harTrain <- harTrain[,-c(1:7)]
harTest  <- harTest [,-c(1:7)]

# Compare the sizes of the training and testing datasets  
dim(harTrain); dim(harTest)
```

```
## [1] 19622    53
```

```
## [1] 20 53
```

```r
# Summarize the values in the outcome variable (classe) of the training dataset
summary(harTrain$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
We see that instances of correct lift (classe = 'A') are most (at 5580) as compared to the occurrences of each of the incorrect lifts.  

## Data Preparation  
The training dataset is fairly large with 19622 rows and 53 columns. We partition the training dataset using random sampling without replacement into the following 2 datasets to allow cross validation:  
- harTrng (80% of training data)    
- harVal  (20% of training data)  


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(12345) # For reproducibility
inTrain <- createDataPartition(y=harTrain$classe, p=0.80, list=FALSE)
harTrng <- harTrain[inTrain,]
harVal  <- harTrain[-inTrain,]
```
The core training dataset contains 15699 rows and the validation dataset contains 3923 rows.  Comparing the the values of the training an validation sets:- 

```r
par(mfrow=c(1,2))
plot(harTrng$classe, col="blue", xlab="classe levels in Training dataset", ylab="Frequency")
plot(harVal$classe, col="blue", xlab="classe levels in validation dataset", ylab="Frequency")
```

![](./PML-Project_files/figure-html/unnamed-chunk-3-1.png) 

```r
#Write to PNG file
dev.copy(png, file = "Plot1.png")
```

```
## png 
##   3
```

```r
dev.off()
```

```
## pdf 
##   2
```
We observe that the distribution of outcomes is similar in both the training and validation datasets.  

##Model Development for predicting exercise correctness  

```r
#Load required packages
library(MASS)
library(rpart)
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.1.2
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.1.2
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```
Using Decision Trees algorithm  

```r
dtModel      <- rpart(classe ~ ., data=harTrng, method="class")
dtPrediction <- predict(dtModel, harVal, type = "class")
fancyRpartPlot(dtModel)
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![](./PML-Project_files/figure-html/unnamed-chunk-5-1.png) 

```r
#Write to PNG file
dev.copy(png, file = "Plot2.png")
```

```
## png 
##   3
```

```r
dev.off()
```

```
## pdf 
##   2
```

```r
#Plot confusion matrix
confusionMatrix(dtPrediction, harVal$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1003  165    9   77   12
##          B   21  390   72   23   50
##          C   36   64  542  101   92
##          D   33   58   45  408   36
##          E   23   82   16   34  531
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7326          
##                  95% CI : (0.7185, 0.7464)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6604          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8987  0.51383   0.7924   0.6345   0.7365
## Specificity            0.9063  0.94753   0.9095   0.9476   0.9516
## Pos Pred Value         0.7923  0.70144   0.6491   0.7034   0.7741
## Neg Pred Value         0.9575  0.89041   0.9540   0.9297   0.9413
## Prevalence             0.2845  0.19347   0.1744   0.1639   0.1838
## Detection Rate         0.2557  0.09941   0.1382   0.1040   0.1354
## Detection Prevalence   0.3227  0.14173   0.2128   0.1478   0.1749
## Balanced Accuracy      0.9025  0.73068   0.8510   0.7910   0.8440
```
We observe that the accuracy of the decision trees algorithm is 73.26% with an out of sample error at 26.74%.  

Using Naive Bayes algorithm  

```r
library(e1071) #Package required to run Naive Bayes algorithm
```

```
## Warning: package 'e1071' was built under R version 3.1.2
```

```r
nbModel      <- naiveBayes(classe ~ ., data=harTrng)
nbPrediction <- predict(nbModel, newdata=harVal)
#Plot confusion matrix
confusionMatrix(nbPrediction, harVal$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 327  40  15   0  11
##          B 103 456  57  27 155
##          C 512 159 494 235  92
##          D 141  49  86 307 108
##          E  33  55  32  74 355
## 
## Overall Statistics
##                                         
##                Accuracy : 0.4943        
##                  95% CI : (0.4785, 0.51)
##     No Information Rate : 0.2845        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.3766        
##  Mcnemar's Test P-Value : < 2.2e-16     
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity           0.29301   0.6008   0.7222  0.47745  0.49237
## Specificity           0.97649   0.8919   0.6919  0.88293  0.93941
## Pos Pred Value        0.83206   0.5714   0.3311  0.44428  0.64663
## Neg Pred Value        0.77649   0.9030   0.9218  0.89604  0.89152
## Prevalence            0.28448   0.1935   0.1744  0.16391  0.18379
## Detection Rate        0.08335   0.1162   0.1259  0.07826  0.09049
## Detection Prevalence  0.10018   0.2034   0.3803  0.17614  0.13994
## Balanced Accuracy     0.63475   0.7463   0.7071  0.68019  0.71589
```
The model accuracy drops further to nearly 50% using the Naive Bayes algorithm. The out of sample error is very high in this case at nearly 50%.  

Using Random Forest algorithm  

```r
library(randomForest)  #Package required to run random forest algorithm
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
set.seed(12345)
rfModel      <- randomForest(classe ~. , data=harTrng, method="class")
rfPrediction <- predict(rfModel, harVal, type = "class")
#Plot confusion matrix
confusionMatrix(rfPrediction, harVal$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    4    0    0    0
##          B    1  755   10    0    0
##          C    0    0  674   10    0
##          D    0    0    0  633    1
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9934          
##                  95% CI : (0.9903, 0.9957)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9916          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9947   0.9854   0.9844   0.9986
## Specificity            0.9986   0.9965   0.9969   0.9997   1.0000
## Pos Pred Value         0.9964   0.9856   0.9854   0.9984   1.0000
## Neg Pred Value         0.9996   0.9987   0.9969   0.9970   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1925   0.1718   0.1614   0.1835
## Detection Prevalence   0.2852   0.1953   0.1744   0.1616   0.1835
## Balanced Accuracy      0.9988   0.9956   0.9911   0.9921   0.9993
```
We observe that the Random Forest algorithm gives the highest prediction accuracy amongst the three models. The model accuracy is 99.34% and the out of sample error is estimated at 0.66% which is the lowest amongst the three models we have tried.  

##Prediction  

We choose the random forest model for prediction as follows:-  

```r
prediction <- predict(rfModel, harTest, type="class")
#Display the 20 predicted values of classe for the test dataset 
prediction
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
#Function to create files for loading data
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
#Create 20 files with prediction for each test case
pml_write_files(prediction)
```
#References  
URL: http://groupware.les.inf.puc-rio.br/har  
