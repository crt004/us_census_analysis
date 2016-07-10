
library(gridExtra)
library(plyr)
library(Amelia)

getwd()
path="/Users/ctrabuco/Desktop/us_census_full/code"
setwd(path)

# Input data files are available in the "../input/" directory.

cat("reading the train and test data\n")

train  <- read.csv("../input/census_income_learn.csv",head=FALSE, na.strings=c("?"," ?"))
test  <- read.csv("../input/census_income_test.csv",head=FALSE, na.strings=c("?"," ?"))
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

colnames(train) <- columns_data
attach(train)
names(train)

## view how many unique values are for each variable
## (just to check with the metadata file)
uniques <- as.data.frame(sapply(train, function(x) length(unique(x)))) 
uniques <- rename(uniques,c("sapply(train, function(x) length(unique(x)))" = "number_of_values"))

##################################
# percentage of missing values per variable
##################################
missing_values <- as.data.frame(colMeans(is.na(train))) 
missing_values <- rename(missing_values, c("colMeans(is.na(train))"="percentage"))
#show only the columns with missing values
missing_values<-subset(missing_values,missing_values$percentage!=0)
#export into pdf
pdf("../reports/missing_values_percentage.pdf")
grid.table(missing_values)
dev.off()

## !!! commented because the graphic takes too long to be created, 
## but the result was saved in the reports folder
#pdf("../reports/missing_values_graphic.pdf")
#missmap(train, main = "Missing values vs observed")
#dev.off()

##################################
# for the variable distribution I use hist() for the continous variables, 
# and barplot() for the discretes
##################################

##################################
#First I make an analysis over the continous variables
##################################

# Age histogram and boxplot
pdf("../reports/continuous_variables/00_age.pdf")
par(mfrow=c(1,2))
colors = c("chartreuse4","chartreuse3", "darkgreen", "darkgrey")
hist(age, breaks=10, right=FALSE, col=colors, main="Person's Age", xlab="Age (Years)", las=2)
boxplot(age, cex.axis=0.5, las=2, main="Box Plot")
dev.off()

# Wage per hour histogram and boxplot
pdf("../reports/continuous_variables/05_wage_per_hour.pdf")
par(mfrow=c(1,2))
hist(wage_per_hour, breaks=10, right=FALSE, col=colors, main="Wage per hour", xlab="w/hs", las=2)
boxplot(wage_per_hour, cex.axis=0.5, las=2)
dev.off()

# capital_gains histogram and boxplot
pdf("../reports/continuous_variables/16_capital_gains.pdf")
par(mfrow=c(1,2))
hist(capital_gains, breaks=10, right=FALSE, col=colors, main="capital_gains", xlab="capital_gains", las=2)
boxplot(capital_gains, cex.axis=0.5, las=2)
dev.off()

# !!! extra analysis for strange graphics results
pdf("../reports/continuous_variables/16_capital_gains2.pdf")
par(mfrow=c(1,2))
boxplot(capital_gains, cex.axis=0.5, las=2, outline = FALSE)
boxplot(capital_gains, cex.axis=0.5, las=2)
dev.off()
# after the the previous plot, I notice that they were too many zeros in capital_gains
# same thing with other continous variables, so I didn't make another extra analysis with them

# capital_losses histogram and boxplot
pdf("../reports/continuous_variables/17_capital_losses.pdf")
par(mfrow=c(1,2))
hist(capital_losses, breaks=10, right=FALSE, col=colors, main="capital_losses", xlab="capital_losses", las=2)
boxplot(capital_losses, cex.axis=0.5, las=2)
dev.off()

# dividends_from_stocks histogram and boxplot
pdf("../reports/continuous_variables/18_dividends_from_stocks.pdf")
par(mfrow=c(1,2))
hist(dividends_from_stocks, breaks=10, right=FALSE, col=colors, main="dividends_from_stocks", xlab="dividends_from_stocks", las=2)
boxplot(dividends_from_stocks, cex.axis=0.5, las=2)
dev.off()

# num_persons_worked_for_employer histogram and boxplot
pdf("../reports/continuous_variables/29_num_persons_worked_for_employer.pdf")
par(mfrow=c(1,2))
hist(num_persons_worked_for_employer, breaks=10, right=FALSE, col=colors, main="num_persons_worked_for_employer", xlab="num_persons_worked_for_employer", las=2)
boxplot(num_persons_worked_for_employer, cex.axis=0.5, las=2)
dev.off()

# weeks_worked_in_year histogram and boxplot
pdf("../reports/continuous_variables/38_weeks_worked_in_year.pdf")
par(mfrow=c(1,2))
hist(weeks_worked_in_year, breaks=10, right=FALSE, col=colors, main="weeks_worked_in_year", xlab="weeks_worked_in_year", las=2)
boxplot(weeks_worked_in_year, cex.axis=0.5, las=2)
dev.off()

###################################
# Note: too many zeros in: capital_gains, wage_per_hour, capital_losses, dividends_from_stocks 
# this make that any value different from zero be considered as outlier
# probably make a pre-processing of data and consider as NA before training the model
###################################


##################################
# analysis over nominal variables
##################################


# the feature index according to the metadata file
# 33 categorical variables (also according to metadata file)
features.index = c("01", "02", "03", "04", "06", "07", "08", "09", "10","11", "12", "13","14",
                   "15", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "30",
                   "31", "32", "33", "34", "35", "36", "37", "39")
features.name = c("class_of_worker", "detailed_industry_recode", 
                      "detailed_occupation_recode", "education",
                      "enroll_in_edu_inst_last_wk", "marital_stat",
                      "major_industry_code", "major_occupation_code",
                      "race", "hispanic_origin",
                      "sex", "member_of_a_labor_union",
                      "reason_for_unemployment", "full_or_part_time_employment_stat",
                      "tax_filer_stat", "region_of_previous_residence",
                      "state_of_previous_residence", "detailed_household_and_family_stat",
                      "detailed_household_summary_in_hous",
                      "migration_code-change_in_msa", "migration_code-change_in_reg",
                      "migration_code-move_within_reg", "live_in_this_house_1_year_ago",
                      "migration_prev_res_in_sunbelt", "family_members_under_18",
                      "country_of_birth_father", "country_of_birth_mother",
                      "country_of_birth_self", "citizenship",
                      "own_business_or_self_employed","fill_inc_questionnaire_for_veteran",
                      "veterans_benefits","year")

# print the reports
for (i in 1:33) {
  plot.new()
  par(mfrow=c(1,1))
  # path to store the pdf report
  pdf_path <- paste("../reports/categorical_variables/", features.index[i],"_",features.name[i],".pdf", sep="")
  pdf(pdf_path)
  # barplot of the feature
  feature.freq <- table(train[[features.name[i]]])
  barplot(feature.freq, las=2, cex.names = 0.5)
  dev.off()
  
  # pie chart of the feature
  piepercent <- round(100*feature.freq/sum(feature.freq), 1)
  
  pdf_path <- paste("../reports/categorical_variables/", features.index[i],"_",features.name[i],"2.pdf", sep="")
  pdf(pdf_path)
  pie(feature.freq, labels = piepercent,radius = 0.9, col = rainbow(length(feature.freq)))
  legend("topleft",names(feature.freq), fill = rainbow(length(feature.freq)), cex=0.5)
  dev.off()
}




