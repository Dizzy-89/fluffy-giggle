# Clearing the Global Environment
rm(list = ls())

# ===== Setting the Environment =====
set.seed(1)
options(scipen = 999, digits = 1)

# Checking for required Libraries
library(forecast)
library(caret)
library(e1071)
library(jtools)
library(neuralnet)
library(NeuralNetTools)
library(rpart)
library(rpart.plot)
library(rattle)
library(gmodels)
library(randomForest)
library(class)
library(uplift)

# Loading Data set
loan_df <- read.csv("df1_loan.csv", fileEncoding="UTF-8-BOM")

# Exploring the variables
summary(loan_df)

distribution <- table(loan_df$Gender)
barplot(distribution)

distribution <- table(loan_df$Married)
barplot(distribution)

distribution <- table(loan_df$Education)
barplot(distribution)

distribution <- table(loan_df$Self_Employed)
barplot(distribution)

distribution <- table(loan_df$Property_Area)
barplot(distribution)

distribution <- table(loan_df$Loan_Status)
barplot(distribution)

hist(loan_df$LoanAmount, main="Loan Amount",
     xlab="Dollar Amount", ylab="Frequency" )
hist(loan_df$ApplicantIncome)

# ===== Configuring the Data =====
# Sub-setting: Removing unneeded Categories
loan_df <- subset(loan_df, select=-c(X, Loan_ID))

# Converting Categorical Variables into Dummy Variables:
# 1 = Male; 0 = Female
loan_df$Gender <- ifelse(loan_df$Gender == "Male", 1, 0)

# 1 = Married; 0 = Single
loan_df$Married <- ifelse(loan_df$Married == "Yes", 1, 0)

# 1 = Not Graduate; 0 = Graduate
loan_df$Education <- ifelse(loan_df$Education >= "Not Graduate", 1, 0)

# 1 = Self-Employed; 0 = Not Self-Employed
loan_df$Self_Employed <- ifelse(loan_df$Self_Employed >= "Yes", 1, 0)

# 1 = Urban; 0 = Rural
loan_df$Property_Area <- ifelse(loan_df$Property_Area >= "Urban", 1, 0)

# 1 = Approved; 0 = Denied
loan_df$Loan_Status <- ifelse(loan_df$Loan_Status >= "Y", 1, 0)

# Changing Variables to Numeric
loan_df$Dependents <- as.numeric(loan_df$Dependents)
loan_df$Dependents[is.na(loan_df$Dependents)] <- 0
loan_df$Total_Income <- as.numeric(gsub('[$,]', '', loan_df$Total_Income))

# Editing NA values
loan_df <- loan_df[!is.na(loan_df$Credit_History), ]
loan_df <- loan_df[!is.na(loan_df$LoanAmount), ]
loan_df <- loan_df[!is.na(loan_df$Loan_Amount_Term), ]

# Checking to see if all Missing Values have been removed
summary(loan_df)

boxplot(LoanAmount ~ Loan_Status,data=loan_df, main="Loan Amount",
        xlab="1 For Loan Approval", ylab="Loan Amount")
boxplot(ApplicantIncome ~ Loan_Status,data=loan_df, main="Loan Amount",
        xlab="1 For Loan Approval", ylab="Loan Amount")
boxplot(Total_Income ~ Loan_Status,data=loan_df, main="Loan Amount",
        xlab="1 For Loan Approval", ylab="Loan Amount")

# ===== Principal Component Analysis (PCA) =====
cov(na.omit(loan_df))
cor(na.omit(loan_df))

## PCA on all variables
pca <- prcomp(na.omit(loan_df)) # Omit the NA Set
pca$rot
summary(pca)

# Plot the Proportion of Variance
pov <- pca$sdev^2 / sum(pca$sdev^2)
barplot(pov, xlab = "Principal Components", ylab = "Proportion of Variance Explained")

## PCA on all 13 variables with Normalization
pca.cor <- prcomp(na.omit(loan_df), scale. = T)
pca.cor$rot
summary(pca.cor)

# Plot proportion of Variance
pov.cor <- pca.cor$sdev^2 / sum(pca.cor$sdev^2)
barplot(pov.cor, xlab = "Principal Components", ylab = "Proportion of Variance Explained")

# ===== Normalization ====
# Adjusting to Normalization
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Apply Normalization to the data set
loan_df_adj <- as.data.frame(lapply(loan_df, min_max_norm))

# Normalized Data Visualizations
hist(loan_df_adj$LoanAmount, main="Loan Amount",
     xlab="Dollar Amount", ylab="Frequency" )

# ===== Partitioning the Data =====
train.index <- sample(c(1:dim(loan_df_adj)[1]), dim(loan_df_adj)[1] * 0.60)

train.df <- loan_df_adj[train.index, ]
valid.df <- loan_df_adj[-train.index, ]

# ===== Logistic Regression =====
# Fit Model
lm.fit <- glm(Loan_Status ~ ., data = train.df, family='binomial')

# Report Model
summ(lm.fit)

# Evaluate the Model - Training Set
lm.pred.pro.train <- predict(lm.fit, train.df, type='response')
lm.pred.train <- factor(ifelse(lm.pred.pro.train > 0.71, 1, 0))
confusionMatrix(lm.pred.train, factor(train.df$Loan_Status), positive='1')

# Evaluating the Model - Validation Set
lm.pred.pro.valid <- predict(lm.fit, valid.df, type='response')
lm.pred.valid <- factor(ifelse(lm.pred.pro.valid > 0.71, 1, 0))
confusionMatrix(lm.pred.valid, factor(valid.df$Loan_Status), positive='1')

# ===== Neural Network =====
# Fit Model
nn <- neuralnet(Loan_Status ~ ., data = train.df, hidden = 3)

# report model
plotnet(nn)
plot(nn)

# evaluate model (training set)
nn.pred.pro <- compute(nn, train.df)
nn.pred <- factor(ifelse(nn.pred.pro$net.result > 0.62, 1, 0))
confusionMatrix(as.factor(nn.pred), as.factor(train.df$Loan_Status), positive = '1')

# evaluate model (validation set)
nn.pred.pro.valid <- compute(nn, valid.df)
nn.pred.valid <- factor(ifelse(nn.pred.pro.valid$net.result > 0.62, 1, 0))
confusionMatrix(nn.pred.valid, factor(valid.df$Loan_Status), positive = '1')

# Create de-normalization function
# Extra Material: Talk on the Paper regarding De-Normalization v. Normalized Results
de_normalize = function(x, original){
  return (x * (max(original) - min(original)) + min(original))
}

nn.pred = de_normalize(nn.pred.pro.valid$net.result, valid.df$Loan_Status)
nn.pred = factor(ifelse(nn.pred > 0.62, "1", "0"), levels = c('1','0'))
nn.actual = factor(valid.df$Loan_Status, levels = c('1','0'))

#Confusion matrix for nn test set
confusionMatrix(nn.pred, nn.actual)

# ===== Classification Tree =====
class.tree <- rpart(Loan_Status ~ ., data = train.df, method="class")
summary(class.tree)

# Plot
prp(class.tree, type=4, extra=101, box.palette="GnYlRd",
    fallen.leaves=TRUE, branch=0.3, split.font=1, varlen=-10, under=TRUE)

# Fancy Rpart Plot
fancyRpartPlot(class.tree)

# Decision Rules
rpart.rules(class.tree, extra=4, cover=TRUE)

# Evaluate model w/ train set
pred.loan.train <- predict(class.tree, train.df, type='class')

confusionMatrix(as.factor(pred.loan.train), as.factor(train.df$Loan_Status), positive = '1')

# Evaluate model w/ test set
pred.loan.test <- predict(class.tree, valid.df, type='class')

confusionMatrix(as.factor(pred.loan.test), as.factor(valid.df$Loan_Status), positive = '1')

# ===== Random Forest =====
#Fit random forest model
rf <- randomForest(as.factor(Loan_Status) ~ ., data = train.df, ntree = 1000, 
                   mtry = 2, nodesize = 5, importance = TRUE)

#Plot Mean Decrease Accuracy
varImpPlot(rf, type = 1)

#predict Test Data
rf.pred.train <- predict(rf, train.df)

#Generate confusion matrix
confusionMatrix(as.factor(rf.pred.train), as.factor(train.df$Loan_Status), positive = '1')

#Predict validation data
rf.pred.test <- predict(rf, valid.df)
#Generate confusion matrix
confusionMatrix(as.factor(rf.pred.test), as.factor(valid.df$Loan_Status), positive = '1')

# use upliftRF to apply a Random Forest.
up.fit <- upliftRF(Loan_Status ~ Total_Income + Married + Dependents + Education 
                   + Self_Employed + ApplicantIncome + CoapplicantIncome + LoanAmount 
                   + Loan_Amount_Term + Credit_History + Property_Area + trt(Gender),
                   data = train.df, mtry = 3, ntree = 100, split_method = "KL",
                   minsplit = 200, verbose = TRUE)

levels(train.df$Loan_Status) <- levels(train.df$Loan_Status)
pred <- predict(up.fit, newdata = valid.df)

# first colunm: p(y | treatment) 
# second colunm: p(y | control) 

head(data.frame(pred, "uplift" = pred[,1] - pred[,2]))

# ===== Ensembles =====
# Prediction: Random Forest
rf.pred <- predict(rf, valid.df, type='response')
rf.pred.prob <- predict(rf, valid.df,type="prob")
confusionMatrix(as.factor(rf.pred), as.factor(valid.df$Loan_Status), positive = '1')

# Prediction: Logistic Regression
lr.pred.prob <- predict(lm.fit, valid.df, type ="response")
lr.pred <-ifelse(lr.pred.prob > 0.5, 1, 0)
confusionMatrix(as.factor(lr.pred), as.factor(valid.df$Loan_Status), positive = '1')

# Prediction: Neural Network
nn.pred.results <- compute(nn, valid.df)
nn.pred.prob<-nn.pred.results$net.result
nn.pred <-ifelse(nn.pred.prob > 0.5, 1, 0)
confusionMatrix(as.factor(nn.pred), as.factor(valid.df$Loan_Status), positive = '1')

# Ensemble using Averaging
valid.df$pred_avg<-(rf.pred.prob[,2]+lr.pred.prob+nn.pred.prob)/3
valid.df$pred_avg_class<-ifelse(valid.df$pred_avg>0.5,1,0)
confusionMatrix(as.factor(valid.df$Loan_Status),as.factor(valid.df$pred_avg_class))

# Ensemble using Majority Voting
valid.df$pred_vote_class<-ifelse(rf.pred==1 & nn.pred==1,1,
                                 ifelse(rf.pred==1 & lr.pred==1,1,
                                        ifelse(lr.pred==1 & nn.pred==1,1,0)))
confusionMatrix(as.factor(valid.df$Loan_Status),as.factor(valid.df$pred_vote_class))

# Ensemble using Weighted Averaging
confusionMatrix(as.factor(valid.df$Loan_Status),as.factor(rf.pred))
confusionMatrix(as.factor(valid.df$Loan_Status),as.factor(lr.pred))
confusionMatrix(as.factor(valid.df$Loan_Status),as.factor(nn.pred))

valid.df$pred_weighted<-(rf.pred.prob[,2]*0.25)+
  (lr.pred.prob*0.5)+(nn.pred*0.5)
valid.df$pred_weighted_class<-ifelse(valid.df$pred_weighted>0.5,1,0)
confusionMatrix(as.factor(valid.df$Loan_Status),as.factor(valid.df$pred_weighted_class))

# ===== K-Nearest Neighbor =====
# run kNN with k=16
nn5 <- knn(train.df, valid.df, cl=as.factor(train.df$Loan_Status), k=16)
confusionMatrix(as.factor(nn5), as.factor(valid.df$Loan_Status), positive = '1')

# Find optimal K (from 1 to 15) in terms of accuracy
accuracy.df <- data.frame(k=seq(1, 15, 1), accuracy=0)

for(i in 1:15){
  knn.pred <- knn(train.df, valid.df, cl=as.factor(train.df$Loan_Status), k=i)
  accuracy.df[i, 'accuracy'] <- confusionMatrix(knn.pred, 
                                                as.factor(valid.df$Loan_Status))$overall[1]
}

View(accuracy.df)

# run kNN with k=5
nn5 <- knn(train.df, valid.df, cl=as.factor(train.df$Loan_Status), k=5)
confusionMatrix(as.factor(nn5), as.factor(valid.df$Loan_Status), positive = '1')