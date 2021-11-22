# Clearing Global Environment
rm(list = ls())

# ===== Setting the Environment =====
set.seed(1)
options(scipen = 999, digits = 1)
library(forecast)

# ===== Configuring the Data =====
# Loading Data set

loan_df <- read.csv("loan_eligibility_copy.csv",fileEncoding="UTF-8-BOM")
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

# Editing NA values from Credit Score and Annual Income
loan_df <- loan_df[!is.na(loan_df$Credit.Score), ]

# Removing loans with Credit Scores of over 850
loan_df <- loan_df[!loan_df$Credit.Score > 850, ]

# Removing loans with Current Amount of $99999999
loan_df <- loan_df[!loan_df$Current.Loan.Amount == 99999999, ]

# ===== Partitioning the Data =====
train.index <- sample(c(1:dim(loan_df)[1]), dim(loan_df)[1] * 0.6)

train.df <- loan_df[train.index, ]
valid.df <- loan_df[-train.index, ]

# ===== Using Linear Regression =====
loan.lm <- lm(Credit.Score ~ ., data = train.df)

# Showing Regression Results
ModelSummary <- summary(loan.lm)
ModelSummary
