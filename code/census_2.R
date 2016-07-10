library(caret)
library(e1071)
library(rpart)
library(party)

getwd()
path="/Users/ctrabuco/Desktop/us_census_full/code"
setwd(path)

# Input data files are available in the "../input/" directory.

cat("reading the train and test data\n")
train.raw  <- read.csv("../input/census_income_learn.csv",head=FALSE, na.strings= c("?"," ?"))
test.raw  <- read.csv("../input/census_income_test.csv",head=FALSE, na.strings = c("?"," ?"))

columns_data <- c("age",
                  "class_of_worker",
                  "detailed_industry_recode",
                  "detailed_occupation_recode",
                  "education",
                  "wage_per_hour",
                  "enroll_in_edu_inst_last_wk",
                  "marital_stat",
                  "major_industry_code",
                  "major_occupation_code",
                  "race",
                  "hispanic_origin",
                  "sex",
                  "member_of_a_labor_union",
                  "reason_for_unemployment",
                  "full_or_part_time_employment_stat",
                  "capital_gains",
                  "capital_losses",
                  "dividends_from_stocks",
                  "tax_filer_stat",
                  "region_of_previous_residence",
                  "state_of_previous_residence",
                  "detailed_household_and_family_stat",
                  "detailed_household_summary_in_hous",
                  "instance_weight", # ignore (?)
                  "migration_code-change_in_msa",
                  "migration_code-change_in_reg",
                  "migration_code-move_within_reg",
                  "live_in_this_house_1_year_ago",
                  "migration_prev_res_in_sunbelt",
                  "num_persons_worked_for_employer",
                  "family_members_under_18",
                  "country_of_birth_father",
                  "country_of_birth_mother",
                  "country_of_birth_self",
                  "citizenship",
                  "own_business_or_self_employed",
                  "fill_inc_questionnaire_for_veteran",
                  "veterans_benefits",
                  "weeks_worked_in_year",
                  "year",
                  "target")

numeric_columns_data <- c("age",
                  "wage_per_hour",
                  "capital_gains",
                  "capital_losses",
                  "dividends_from_stocks",
                  "num_persons_worked_for_employer",
                  "weeks_worked_in_year")

# columns that contain any NA, 
# (after analysis in previous phase, I found that they are the same in train and test)
columns_with_na = c("state_of_previous_residence",
                    "migration_code-change_in_msa",
                    "migration_code-change_in_reg",
                    "migration_code-move_within_reg",
                    "migration_prev_res_in_sunbelt",
                    "country_of_birth_father",
                    "country_of_birth_mother",
                    "country_of_birth_self")



colnames(train.raw) <- columns_data
colnames(test.raw) <- columns_data

###################################################
# MODEL 1 : Logistic Regression
###################################################

train <- train.raw
#extract instance_weight (ignore column as specified in metadata file)
columns_to_drop = c("instance_weight")

train = train[,-which(names(train) %in% columns_to_drop)]

# create get_mode function
get_mode <- function(x){
  uniq <- na.omit(unique(x)) 
  uniq[which.max(tabulate(match(x, uniq)))]
}

#replace the NA values for the mode of the column
for(c in columns_with_na){
  if (!(c %in% columns_to_drop)){
    train[is.na(train[,c]), c] <- get_mode(train[[c]])  
  }
}


train.names <- names(train)[1:ncol(train) ]
cat("convert categorical variables from string to integer\n")
for (f in train.names) {
  if (!(f %in% numeric_columns_data)){
    train[[f]] <- as.integer(train[[f]])
  }
}

train$target = train$target - 1
#library(Metrics)

n_folds = 10
folds_i = sample(rep(1:n_folds, length.out = nrow(train)))
cv_tmp <- matrix(NA,nrow = n_folds, ncol = length(train))
global_misClasificError = 0

for(k in 1:n_folds){
  test_i <- which(folds_i == k)
  train_ <- train[-test_i,]
  test_ <- train[test_i,]
  
  # train model
  model <- glm(train_$target ~.,family=binomial(link='logit'),data=train_)
  # make predictions with the model
  prediction = predict(model, test_, type="response")
  
  fitted.results <- ifelse(prediction > 0.5,1,0)
  misClasificError <- mean(fitted.results != test_$target)
  print(paste('Accuracy fold',k,'-' ,1-misClasificError))
  global_misClasificError = global_misClasificError + (1-misClasificError)
  # TODO: there should be a more elegant way to concatenate dataframes
  # for the moment this work (not my proudest code, but it makes the job)
  if( k==1 ) ideal1 <- test_$target
  if( k==2 ) ideal2 <- test_$target
  if( k==3 ) ideal3 <- test_$target
  if( k==4 ) ideal4 <- test_$target
  if( k==5 ) ideal5 <- test_$target
  if( k==6 ) ideal6 <- test_$target
  if( k==7 ) ideal7 <- test_$target
  if( k==8 ) ideal8 <- test_$target
  if( k==9 ) ideal9 <- test_$target
  if( k==10 ) ideal10 <- test_$target
  if( k==1 ) prediction1 <- fitted.results
  if( k==2 ) prediction2 <- fitted.results
  if( k==3 ) prediction3 <- fitted.results
  if( k==4 ) prediction4 <- fitted.results
  if( k==5 ) prediction5 <- fitted.results
  if( k==6 ) prediction6 <- fitted.results
  if( k==7 ) prediction7 <- fitted.results
  if( k==8 ) prediction8 <- fitted.results
  if( k==9 ) prediction9 <- fitted.results
  if( k==10 ) prediction10 <- fitted.results
}
ideal <- rbind(ideal1, ideal2, ideal3, ideal4, ideal5, ideal6, ideal7, ideal8, ideal9, ideal10 )
prediction <- rbind(prediction1, prediction2, prediction3, prediction4, prediction5, 
                    prediction6, prediction7, prediction8, prediction9, prediction10 )

confusionMatrix(prediction, ideal)

#Confusion Matrix and Statistics
#
#Reference
#Prediction      0      1
#0 185808   9076
#1   1340   3306
#
#Accuracy : 0.9478          
#95% CI : (0.9468, 0.9488)
#No Information Rate : 0.9379          
#P-Value [Acc > NIR] : < 2.2e-16       
#
#Kappa : 0.3669          
#Mcnemar's Test P-Value : < 2.2e-16       
#
#Sensitivity : 0.9928          
#Specificity : 0.2670          
#Pos Pred Value : 0.9534          
#Neg Pred Value : 0.7116          
#Prevalence : 0.9379          
#Detection Rate : 0.9312          
#Detection Prevalence : 0.9767          
#Balanced Accuracy : 0.6299          
#
#'Positive' Class : 0

###################################################
# MODEL 2 : Logistic Regression droping NA columns
###################################################

 
train <- train.raw

#extract instance_weight and the features with excess of NA
columns_to_drop = c("migration_code-change_in_msa",
                    "migration_code-change_in_reg",
                    "migration_code-move_within_reg",
                    "migration_prev_res_in_sunbelt",
                    "instance_weight")

train = train[,-which(names(train) %in% columns_to_drop)]

# create get_mode function
get_mode <- function(x){
  uniq <- na.omit(unique(x)) 
  uniq[which.max(tabulate(match(x, uniq)))]
}

#replace the NA values for the mode of the column
for(c in columns_with_na){
  if (!(c %in% columns_to_drop)){
    train[is.na(train[,c]), c] <- get_mode(train[[c]])  
  }
}


train.names <- names(train)[1:ncol(train) ]
cat("convert categorical variables from string to integer\n")
for (f in train.names) {
  if (!(f %in% numeric_columns_data)){
    train[[f]] <- as.integer(train[[f]])
  }
}

train$target = train$target - 1
#library(Metrics)
#library(caret)
n_folds = 10
folds_i = sample(rep(1:n_folds, length.out = nrow(train)))
cv_tmp <- matrix(NA,nrow = n_folds, ncol = length(train))
global_misClasificError = 0

for(k in 1:n_folds){
  test_i <- which(folds_i == k)
  train_ <- train[-test_i,]
  test_ <- train[test_i,]
  
  model <- glm(train_$target ~.,family=binomial(link='logit'),data=train_)
  
  prediction = predict(model, test_, type="response")
  
  fitted.results <- ifelse(prediction > 0.5,1,0)
  misClasificError <- mean(fitted.results != test_$target)
  print(paste('Accuracy fold',k,'-' ,1-misClasificError))
  global_misClasificError = global_misClasificError + (1-misClasificError)
  # TODO: there should be a more elegant way to concatenate dataframes
  # for the moment this work (not my proudest code, but it makes the job)
  if( k==1 ) ideal1 <- test_$target
  if( k==2 ) ideal2 <- test_$target
  if( k==3 ) ideal3 <- test_$target
  if( k==4 ) ideal4 <- test_$target
  if( k==5 ) ideal5 <- test_$target
  if( k==6 ) ideal6 <- test_$target
  if( k==7 ) ideal7 <- test_$target
  if( k==8 ) ideal8 <- test_$target
  if( k==9 ) ideal9 <- test_$target
  if( k==10 ) ideal10 <- test_$target
  if( k==1 ) prediction1 <- fitted.results
  if( k==2 ) prediction2 <- fitted.results
  if( k==3 ) prediction3 <- fitted.results
  if( k==4 ) prediction4 <- fitted.results
  if( k==5 ) prediction5 <- fitted.results
  if( k==6 ) prediction6 <- fitted.results
  if( k==7 ) prediction7 <- fitted.results
  if( k==8 ) prediction8 <- fitted.results
  if( k==9 ) prediction9 <- fitted.results
  if( k==10 ) prediction10 <- fitted.results
}
ideal <- rbind(ideal1, ideal2, ideal3, ideal4, ideal5, ideal6, ideal7, ideal8, ideal9, ideal10 )
prediction <- rbind(prediction1, prediction2, prediction3, prediction4, prediction5, 
                    prediction6, prediction7, prediction8, prediction9, prediction10 )

confusionMatrix(prediction, ideal)

#Confusion Matrix and Statistics
#
#Reference
#Prediction      0      1
#0 185824   9085
#1   1324   3297
#
#Accuracy : 0.9478          
#95% CI : (0.9468, 0.9488)
#No Information Rate : 0.9379          
#P-Value [Acc > NIR] : < 2.2e-16       
#
#Kappa : 0.3664          
#Mcnemar's Test P-Value : < 2.2e-16       
#
#Sensitivity : 0.9929          
#Specificity : 0.2663          
#Pos Pred Value : 0.9534          
#Neg Pred Value : 0.7135          
#Prevalence : 0.9379          
#Detection Rate : 0.9313          
#Detection Prevalence : 0.9768          
#Balanced Accuracy : 0.6296          
#
#'Positive' Class : 0 


###################################################
# MODEL 3 : Logistic Regression with zeros = NA
###################################################

train <- train.raw

# extract instance_weight and the features with excess of NA
# and consider that the columns with too many zeros had in fact too many NA
columns_to_drop = c("capital_gains", 
                    "wage_per_hour",
                    "capital_losses",
                    "dividends_from_stocks",
                    "migration_code-change_in_msa",
                    "migration_code-change_in_reg",
                    "migration_code-move_within_reg",
                    "migration_prev_res_in_sunbelt",
                    "instance_weight")

train = train[,-which(names(train) %in% columns_to_drop)]

# create get_mode function
get_mode <- function(x){
  uniq <- na.omit(unique(x)) 
  uniq[which.max(tabulate(match(x, uniq)))]
}

#replace the NA values for the mode of the column
for(c in columns_with_na){
  if (!(c %in% columns_to_drop)){
    train[is.na(train[,c]), c] <- get_mode(train[[c]])  
  }
}


train.names <- names(train)[1:ncol(train) ]
cat("convert categorical variables from string to integer\n")
for (f in train.names) {
  if (!(f %in% numeric_columns_data)){
    train[[f]] <- as.integer(train[[f]])
  }
}

train$target = train$target - 1
#library(Metrics)
#library(caret)
n_folds = 10
folds_i = sample(rep(1:n_folds, length.out = nrow(train)))
cv_tmp <- matrix(NA,nrow = n_folds, ncol = length(train))
global_misClasificError = 0

for(k in 1:n_folds){
  test_i <- which(folds_i == k)
  train_ <- train[-test_i,]
  test_ <- train[test_i,]
  
  model <- glm(train_$target ~.,family=binomial(link='logit'),data=train_)
  
  prediction = predict(model, test_, type="response")
  
  fitted.results <- ifelse(prediction > 0.5,1,0)
  misClasificError <- mean(fitted.results != test_$target)
  print(paste('Accuracy fold',k,'-' ,1-misClasificError))
  global_misClasificError = global_misClasificError + (1-misClasificError)
  # TODO: there should be a more elegant way to concatenate dataframes
  # for the moment this work (not my proudest code, but it makes the job)
  if( k==1 ) ideal1 <- test_$target
  if( k==2 ) ideal2 <- test_$target
  if( k==3 ) ideal3 <- test_$target
  if( k==4 ) ideal4 <- test_$target
  if( k==5 ) ideal5 <- test_$target
  if( k==6 ) ideal6 <- test_$target
  if( k==7 ) ideal7 <- test_$target
  if( k==8 ) ideal8 <- test_$target
  if( k==9 ) ideal9 <- test_$target
  if( k==10 ) ideal10 <- test_$target
  if( k==1 ) prediction1 <- fitted.results
  if( k==2 ) prediction2 <- fitted.results
  if( k==3 ) prediction3 <- fitted.results
  if( k==4 ) prediction4 <- fitted.results
  if( k==5 ) prediction5 <- fitted.results
  if( k==6 ) prediction6 <- fitted.results
  if( k==7 ) prediction7 <- fitted.results
  if( k==8 ) prediction8 <- fitted.results
  if( k==9 ) prediction9 <- fitted.results
  if( k==10 ) prediction10 <- fitted.results
}
ideal <- rbind(ideal1, ideal2, ideal3, ideal4, ideal5, ideal6, ideal7, ideal8, ideal9, ideal10 )
prediction <- rbind(prediction1, prediction2, prediction3, prediction4, prediction5, 
                    prediction6, prediction7, prediction8, prediction9, prediction10 )

confusionMatrix(prediction, ideal)

#Confusion Matrix and Statistics
#
#Reference
#Prediction      0      1
#0 185649  10315
#1   1499   2067
#
#Accuracy : 0.9408          
#95% CI : (0.9397, 0.9418)
#No Information Rate : 0.9379          
#P-Value [Acc > NIR] : 5.697e-08       
#
#Kappa : 0.2381          
#Mcnemar's Test P-Value : < 2.2e-16       
#
#Sensitivity : 0.9920          
#Specificity : 0.1669          
#Pos Pred Value : 0.9474          
#Neg Pred Value : 0.5796          
#Prevalence : 0.9379          
#Detection Rate : 0.9304          
#Detection Prevalence : 0.9821          
#Balanced Accuracy : 0.5795          
#
#'Positive' Class : 0 


###################################################
# MODEL 4 : Decision Trees
###################################################
train <- train.raw
#extract instance_weight (ignore column as specified in metadata file)
columns_to_drop = c("instance_weight")

train = train[,-which(names(train) %in% columns_to_drop)]

# create get_mode function
get_mode <- function(x){
  uniq <- na.omit(unique(x)) 
  uniq[which.max(tabulate(match(x, uniq)))]
}

#replace the NA values for the mode of the column
for(c in columns_with_na){
  if (!(c %in% columns_to_drop)){
    train[is.na(train[,c]), c] <- get_mode(train[[c]])  
  }
}

n_folds = 10
folds_i = sample(rep(1:n_folds, length.out = nrow(train)))
cv_tmp <- matrix(NA,nrow = n_folds, ncol = length(train))
global_misClasificError = 0

for(k in 1:n_folds){
  test_i <- which(folds_i == k)
  train_ <- train[-test_i,]
  test_ <- train[test_i,]
  
  tree <- rpart(target ~.,method = "class", data =train_)
  printcp(tree)
  #post(tree, file = "tree.ps", title = "Decision tree")
  
  prediction = predict(tree, test_[,-41])
  fitted.results <- ifelse(prediction[," - 50000."] > prediction[," 50000+."]," - 50000."," 50000+.")
  m <- as.data.frame(fitted.results)
  misClasificError <- mean(m$fitted.results != test_$target)
  print(paste('Accuracy fold',k,'-' ,1-misClasificError))
  global_misClasificError = global_misClasificError + (1-misClasificError)
  # TODO: there should be a more elegant way to concatenate dataframes
  # for the moment this work (not my proudest code, but it makes the job)
  if( k==1 ) ideal1 <- test_$target
  if( k==2 ) ideal2 <- test_$target
  if( k==3 ) ideal3 <- test_$target
  if( k==4 ) ideal4 <- test_$target
  if( k==5 ) ideal5 <- test_$target
  if( k==6 ) ideal6 <- test_$target
  if( k==7 ) ideal7 <- test_$target
  if( k==8 ) ideal8 <- test_$target
  if( k==9 ) ideal9 <- test_$target
  if( k==10 ) ideal10 <- test_$target
  if( k==1 ) prediction1 <- m$fitted.results
  if( k==2 ) prediction2 <- m$fitted.results
  if( k==3 ) prediction3 <- m$fitted.results
  if( k==4 ) prediction4 <- m$fitted.results
  if( k==5 ) prediction5 <- m$fitted.results
  if( k==6 ) prediction6 <- m$fitted.results
  if( k==7 ) prediction7 <- m$fitted.results
  if( k==8 ) prediction8 <- m$fitted.results
  if( k==9 ) prediction9 <- m$fitted.results
  if( k==10 ) prediction10 <- m$fitted.results
}
ideal <- rbind(ideal1, ideal2, ideal3, ideal4, ideal5, ideal6, ideal7, ideal8, ideal9, ideal10 )
prediction <- rbind(prediction1, prediction2, prediction3, prediction4, prediction5, 
                    prediction6, prediction7, prediction8, prediction9, prediction10 )

confusionMatrix(prediction, ideal)

#Confusion Matrix and Statistics
#
#Reference
#Prediction      1      2
#1 185630   8811
#2   1518   3571
#
#Accuracy : 0.9482          
#95% CI : (0.9473, 0.9492)
#No Information Rate : 0.9379          
#P-Value [Acc > NIR] : < 2.2e-16       
#
#Kappa : 0.3866          
#Mcnemar's Test P-Value : < 2.2e-16       
#
#Sensitivity : 0.9919          
#Specificity : 0.2884          
#Pos Pred Value : 0.9547          
#Neg Pred Value : 0.7017          
#Prevalence : 0.9379          
#Detection Rate : 0.9303          
#Detection Prevalence : 0.9745          
#Balanced Accuracy : 0.6401          
#
#'Positive' Class : 1 

###################################################
# MODEL 5 : SVM
###################################################
train <- train.raw
#extract instance_weight (ignore column as specified in metadata file)
columns_to_drop = c("instance_weight")

train = train[,-which(names(train) %in% columns_to_drop)]

# create get_mode function
get_mode <- function(x){
  uniq <- na.omit(unique(x)) 
  uniq[which.max(tabulate(match(x, uniq)))]
}

#replace the NA values for the mode of the column
for(c in columns_with_na){
  if (!(c %in% columns_to_drop)){
    train[is.na(train[,c]), c] <- get_mode(train[[c]])  
  }
}


#train.names <- names(train)[1:ncol(train) ]
#cat("convert categorical variables from string to integer\n")
#for (f in train.names) {
#  if (!(f %in% numeric_columns_data)){
#    train[[f]] <- as.integer(train[[f]])
#  }
#}

n_folds = 10
folds_i = sample(rep(1:n_folds, length.out = nrow(train)))
cv_tmp <- matrix(NA,nrow = n_folds, ncol = length(train))
global_misClasificError = 0

for(k in 1:n_folds){
  test_i <- which(folds_i == k)
  train_ <- train[-test_i,]
  test_ <- train[test_i,]
  
  model <- svm(target ~., train_)
  
  prediction = predict(model, test_[,-41])
  
  misClasificError <- mean(prediction != test_$target)
  print(paste('Accuracy fold',k,'-' ,1-misClasificError))
  global_misClasificError = global_misClasificError + (1-misClasificError)
  # TODO: there should be a more elegant way to concatenate dataframes
  # for the moment this work (not my proudest code, but it makes the job)
  if( k==1 ) ideal1 <- test_$target
  if( k==2 ) ideal2 <- test_$target
  if( k==3 ) ideal3 <- test_$target
  if( k==4 ) ideal4 <- test_$target
  if( k==5 ) ideal5 <- test_$target
  if( k==6 ) ideal6 <- test_$target
  if( k==7 ) ideal7 <- test_$target
  if( k==8 ) ideal8 <- test_$target
  if( k==9 ) ideal9 <- test_$target
  if( k==10 ) ideal10 <- test_$target
  if( k==1 ) prediction1 <- prediction
  if( k==2 ) prediction2 <- prediction
  if( k==3 ) prediction3 <- prediction
  if( k==4 ) prediction4 <- prediction
  if( k==5 ) prediction5 <- prediction
  if( k==6 ) prediction6 <- prediction
  if( k==7 ) prediction7 <- prediction
  if( k==8 ) prediction8 <- prediction
  if( k==9 ) prediction9 <- prediction
  if( k==10 ) prediction10 <- prediction
}
ideal <- rbind(ideal1, ideal2, ideal3, ideal4, ideal5, ideal6, ideal7, ideal8, ideal9, ideal10 )
prediction <- rbind(prediction1, prediction2, prediction3, prediction4, prediction5, 
                    prediction6, prediction7, prediction8, prediction9, prediction10 )

confusionMatrix(prediction, ideal)

#Confusion Matrix and Statistics
#
#Reference
#Prediction      1      2
#1 186690   9876
#2    458   2506
#
#Accuracy : 0.9482          
#95% CI : (0.9472, 0.9492)
#No Information Rate : 0.9379          
#P-Value [Acc > NIR] : < 2.2e-16       
#
#Kappa : 0.3101          
#Mcnemar's Test P-Value : < 2.2e-16       
#
#Sensitivity : 0.9976          
#Specificity : 0.2024          
#Pos Pred Value : 0.9498          
#Neg Pred Value : 0.8455          
#Prevalence : 0.9379          
#Detection Rate : 0.9356          
#Detection Prevalence : 0.9851          
#Balanced Accuracy : 0.6000          
#
#'Positive' Class : 1 



###################################################
# APPLY MODEL TO TEST DATASET
###################################################

# Altough there was a tie between the accuracy of SVM and Desicion Tree,
# the DT took arround 5~10 minutes to fit, 
# meanwhile the SVM took arround 7~8 hours
# And thats the reason why I choosed DT to SVM
# and also because the results of a DT can be more easy to interpret

train <- train.raw
test <- test.raw
#extract instance_weight (ignore column as specified in metadata file)
columns_to_drop = c("instance_weight")

train = train[,-which(names(train) %in% columns_to_drop)]
test = test[,-which(names(test) %in% columns_to_drop)]

# create get_mode function
get_mode <- function(x){
  uniq <- na.omit(unique(x)) 
  uniq[which.max(tabulate(match(x, uniq)))]
}

#replace the NA values for the mode of the column
for(c in columns_with_na){
  if (!(c %in% columns_to_drop)){
    train[is.na(train[,c]), c] <- get_mode(train[[c]])
    test[is.na(test[,c]), c] <- get_mode(test[[c]])
  }
}

train_ <- train
test_ <- test

# tarin the model
tree <- rpart(target ~.,method = "class", data =train_)
# print tree info
printcp(tree)
# generate a graphic of the tree
post(tree, file = "tree.ps", title = "Decision tree")

# make predictions with the model
prediction = predict(tree, test_[,-41])
fitted.results <- ifelse(prediction[," - 50000."] > prediction[," 50000+."]," - 50000."," 50000+.")
m <- as.data.frame(fitted.results)

confusionMatrix(m$fitted.results,test_$target)

#Classification tree:
#  rpart(formula = target ~ ., data = train_, method = "class")
#
#Variables actually used in tree construction:
#  [1] age                   capital_gains         dividends_from_stocks education            
#[5] major_occupation_code sex                  
#
#Root node error: 12382/199523 = 0.062058
#
#n= 199523 
#
#CP nsplit rel error  xerror      xstd
#1 0.040260      0   1.00000 1.00000 0.0087035
#2 0.017869      2   0.91948 0.92013 0.0083707
#3 0.017687      6   0.84801 0.87482 0.0081742
#4 0.010000      7   0.83032 0.83549 0.0079986
#
#
#
#
#
#Confusion Matrix and Statistics
#
#Reference
#Prediction   - 50000.  50000+.
#- 50000.     92739     4349
#50000+.        837     1837
#
#Accuracy : 0.948           
#95% CI : (0.9466, 0.9494)
#No Information Rate : 0.938           
#P-Value [Acc > NIR] : < 2.2e-16       
#
#Kappa : 0.3919          
#Mcnemar's Test P-Value : < 2.2e-16       
#
#Sensitivity : 0.9911          
#Specificity : 0.2970          
#Pos Pred Value : 0.9552          
#Neg Pred Value : 0.6870          
#Prevalence : 0.9380          
#Detection Rate : 0.9296          
#Detection Prevalence : 0.9732          
#Balanced Accuracy : 0.6440          
#
#'Positive' Class :  - 50000. 
