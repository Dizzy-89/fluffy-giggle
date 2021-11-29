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

# Loading Data set
loan_df <- read.csv("df1_loan.csv", fileEncoding="UTF-8-BOM")

# Exploring the variables
summary(loan_df)

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

# Checking to see if all Missing Values has been removed
summary(loan_df)

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

# Adjusting to Normalization
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Apply Normalization to the data set
loan_df_adj <- as.data.frame(lapply(loan_df, min_max_norm))

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
class.tree <- rpart(Loan_Status ~ ., data = loan_df_adj, method="class")
summary(class.tree)

# Plot
prp(class.tree, type=4, extra=101, box.palette="GnYlRd",
    fallen.leaves=TRUE, branch=0.3, split.font=1, varlen=-10, under=TRUE)

# Fancy Rpart Plot
fancyRpartPlot(class.tree)

# Decision Rules
rpart.rules(class.tree, extra=4, cover=TRUE)

# Evaluate the Model
pred.loan <- predict(class.tree, loan_df_adj, type='class')
pred.loan

confusionMatrix(as.factor(pred.loan), as.factor(loan_df_adj$Loan_Status), positive = '1')

# ===== Random Forest =====
#Fit random forest model
rf <- randomForest(as.factor(Loan_Status) ~ ., data = train.df, ntree = 1000, 
                   mtry = 2, nodesize = 5, importance = TRUE)

#Plot Mean Decrease Accuracy
varImpPlot(rf, type = 1)

#Predict validation data
rf.pred <- predict(rf, valid.df)

#Generate confusion matrix
confusionMatrix(as.factor(rf.pred), as.factor(valid.df$Loan_Status), positive = '1')

# ===== K-Nearest Neighbor =====
# run kNN with k=5
nn5 <- knn(train.df, valid.df, cl=as.factor(train.df$Loan_Status), k=5)
confusionMatrix(as.factor(nn5), as.factor(valid.df$Loan_Status), positive = '1')

# Find optimal K (from 1 to 15) in terms of accuracy
accuracy.df <- data.frame(k=seq(1, 15, 1), accuracy=0)

for(i in 1:15){
  knn.pred <- knn(train.df, valid.df, cl=as.factor(train.df$Loan_Status), k=i)
  accuracy.df[i, 'accuracy'] <- confusionMatrix(knn.pred, 
                                                as.factor(valid.df$Loan_Status))$overall[1]
}

View(accuracy.df)
