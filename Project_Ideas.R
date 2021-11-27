# Clearing Global Environment
rm(list = ls())

# ===== Setting the Environment =====
set.seed(1)
options(scipen = 999, digits = 1)
library(forecast)
library(caret)
library(e1071)

# ===== Configuring the Data =====
# Loading Data set
loan_df <- read.csv("loan_eligibility_copy.csv", fileEncoding="UTF-8-BOM")

# Sub-setting: Removing unneeded Categories
loan_df <- subset(loan_df, select=-c(Loan.ID, Customer.ID, Purpose, Home.Ownership))

# Converting Categorical Variables into Conditional Statements:
# 1 = True; 0 = False
loan_df$Loan.Status <- ifelse(loan_df$Loan.Status == "Fully Paid", 1, 0)

# 1 = Short-Term; 0 = Long-Term
loan_df$Term <- ifelse(loan_df$Term == "Short Term", 1, 0)

# Keeps only the number of years individuals have worked in their current job.
loan_df$Years.in.current.job <- as.numeric(gsub("([0-9]+).*$", "\\1", loan_df$Years.in.current.job))
loan_df$Years.in.current.job <- ifelse(loan_df$Years.in.current.job >= 5, 1, 0)
# Changing <1 year Experience to 0:
loan_df$Years.in.current.job[is.na(loan_df$Years.in.current.job)] <- 0

# Changing Months since Last Delinquent
loan_df$Months.since.last.delinquent <- ifelse(loan_df$Months.since.last.delinquent >= 1, 1, 0)
# Change NA values to 0
loan_df$Months.since.last.delinquent[is.na(loan_df$Months.since.last.delinquent)] <- 0

# Editing NA values from Credit Score and Annual Income
loan_df <- loan_df[!is.na(loan_df$Credit.Score), ]

# Removing loans with Credit Scores of over 850
loan_df <- loan_df[!loan_df$Credit.Score > 850, ]

# Removing loans with Current Amount of $99999999
loan_df <- loan_df[!loan_df$Current.Loan.Amount == 99999999, ]

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

# Adjusting to PCA Normalization
loan_df_adj <- subset(loan_df, select=-c(Maximum.Open.Credit, Bankruptcies, Tax.Liens))

# ===== Partitioning the Data =====
train.index <- sample(c(1:dim(loan_df_adj)[1]), dim(loan_df_adj)[1] * 0.6)

train.df <- loan_df_adj[train.index, ]
valid.df <- loan_df_adj[-train.index, ]

# ===== Logistic Regression =====
# Fit Model
lm.fit <- glm(Term ~ ., data = train.df, family='binomial')

# Report Model
library(jtools)
summ(lm.fit)

# Evaluate the Model - Training Set
lm.pred.pro.train <- predict(lm.fit, train.df, type='response')
lm.pred.train <- factor(ifelse(lm.pred.pro.train > 0.5, 1, 0))
confusionMatrix(lm.pred.train, factor(train.df$Term), positive='1')

# Evaluating the Model - Validation Set
lm.pred.pro.valid <- predict(lm.fit, valid.df, type='response')
lm.pred.valid <- factor(ifelse(lm.pred.pro.valid > 0.5, 1, 0))
confusionMatrix(lm.pred.valid, factor(valid.df$Term), positive='1')

# ===== Neural Network (STILL Work-In-Progress) =====
library(neuralnet)
# Fit Model
nn <- neuralnet(Loan.Status ~ ., data = train.df, hidden = 3)

# report model
library(NeuralNetTools)
plotnet(nn)
plot(nn)

# evaluate model (training set)
nn.pred.pro <- compute(nn, train.df)
nn.pred <- factor(ifelse(nn.pred.pro$net.result > 0.5, 1, 0))
confusionMatrix(nn.pred, factor(train.df$Loan.Status))

# evaluate model (validation set)
nn.pred.pro.valid <- compute(nn, valid.df)
nn.pred.valid <- factor(ifelse(nn.pred.pro.valid$net.result > 0.5, 1, 0))
confusionMatrix(nn.pred.valid, factor(valid.df$status))


# ===== Classification Tree =====
library(rpart)
library(rpart.plot)
class.tree <- rpart(Term ~ ., data = loan_df_adj, method="class")
summary(class.tree)

# Plot
prp(class.tree, type=4, extra=101, box.palette="GnYlRd",
    fallen.leaves=TRUE, branch=0.3, split.font=1, varlen=-10, under=TRUE)

library(rattle)
fancyRpartPlot(class.tree)

# Decision Rules
rpart.rules(class.tree, extra=4, cover=TRUE)

# Evaluate the Model
library(gmodels)
pred.loan <- predict(class.tree, loan_df_adj, type='class')
pred.loan

confusionMatrix(as.factor(pred.loan), as.factor(loan_df_adj$Term))

# ===== K-Nearest Neighbor (STILL Work-In-Progress) =====
library(class)
# run kNN with k=5
nn5 <- knn(train.df, valid.df, cl=as.factor(train.df$Term), k=5)
confusionMatrix(as.factor(nn5), as.factor(valid.df$Term))

# Find optimal K (from 1 to 15) in terms of accuracy
accuracy.df <- data.frame(k=seq(1, 15, 1), accuracy=0)
for(i in 1:15){
  knn.pred <- knn(train.df, valid.df, cl=as.factor(train.df$Term), k=i)
  accuracy.df[i, 'accuracy'] <- confusionMatrix(knn.pred, 
                                                as.factor(valid.df$Term))$overall[1]
}
View(accuracy.df)
# ===== Random Forest =====
library(randomForest)
## random forest
rf <- randomForest(as.factor() ~ ., data = train.df, ntree = 500, 
                   mtry = 4, nodesize = 5, importance = TRUE)

varImpPlot(rf, type = 1)
summary(rf)
rf$votes

rf.pred <- predict(rf, valid.df)
confusionMatrix(as.factor(rf.pred), as.factor(valid.df$Personal.Loan))
