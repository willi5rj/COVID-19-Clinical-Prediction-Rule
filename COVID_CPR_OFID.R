library(ranger)
library(vip)
library(dplyr)
library(pROC)
library(parallel)
library(tibble)
library(skimr)
library(caret)
library(MASS)


set.seed(12)


setwd('~/Documents/COVID_CPR_OFID_Revisions_Code')

covid1 <- read.csv('covid_imputed.csv')



### MODEL WITH ONLY CLINICAL PREDICTORS


colnames(covid1)

covid <- covid1 %>% dplyr::select(c('AGE_IN_YEARS','SEX_FLG', 'DIABETES_FLG',
                                'CHRONIC_PULMONARY_FLG', 'RENAL_DISEASE_FLG', 'LIVER_DX_FLG', 'OBESITY_FLG', 'CVA_FLG', 'HYPERTENSION_FLG',
                                'NEURO_FLG', 'CAR_ARR_FLG', 'COV_VAX_STATUS_CAT',
                                'RESPIRATORY_RATE', 'TEMPERATURE', 'HEARTRATE', 'SYSTOLIC', 'DIASTOLIC', 'SPO2', 'covid_ed'))

covid$COV_VAX_STATUS_CAT <- gsub("^\\s*", "", covid$COV_VAX_STATUS_CAT) 
covid$COV_VAX_STATUS_CAT <- factor(covid$COV_VAX_STATUS_CAT, levels = c('0', '1-2', '3+'))


clinical_predictors <- c('AGE_IN_YEARS','SEX_FLG', 'DIABETES_FLG',
                         'CHRONIC_PULMONARY_FLG', 'RENAL_DISEASE_FLG', 'LIVER_DX_FLG', 'OBESITY_FLG', 'CVA_FLG', 'HYPERTENSION_FLG',
                         'NEURO_FLG', 'CAR_ARR_FLG', 'COV_VAX_STATUS_CAT', 'RESPIRATORY_RATE', 'TEMPERATURE', 'HEARTRATE', 'SYSTOLIC', 'DIASTOLIC', 'PREVIOUS_COVID_EPISODES_CNT',
                         'SPO2')

bin_vars <- c('SEX_FLG', 'DIABETES_FLG',
'CHRONIC_PULMONARY_FLG', 'RENAL_DISEASE_FLG', 'LIVER_DX_FLG', 'OBESITY_FLG', 'CVA_FLG', 'HYPERTENSION_FLG',
'NEURO_FLG', 'CAR_ARR_FLG')


covid[bin_vars] <- lapply(covid[bin_vars], factor)

covid$covid_ed <- factor(covid$covid_ed, levels = c("0","1"))


str(covid)

merged_df1 <- covid

pred_fun = function(X.model, newdata) {
  predict(X.model, newdata)$predictions[,2]
}





N <- nrow(merged_df1)

# Preallocate result storage: 100 outer × 16 inner
total_inner <- 100 * 16
res_list <- vector("list", total_inner)
idx <- 1L

num_threads <- detectCores()   

for (each in 1:100) {
  
  # 80/20 split
  trn_idx <- sample.int(N, size = floor(0.8 * N))
  train   <- merged_df1[trn_idx, ]
  test    <- merged_df1[-trn_idx, ]
  
  # Base model for VIP
  mod <- ranger(
    covid_ed ~ ., 
    data = train,
    num.trees = 500,              
    probability = TRUE,
    num.threads = num_threads
  )
  
  VIP <- vip(
    object = mod,
    method = "permute",
    target = "covid_ed",
    metric = "roc_auc",
    train = train,
    pred_wrapper = pred_fun,
    event_level = "second",
    num_features = 30
  )$data
  
  # Filter VIP once
  VIP <- VIP %>%
    filter(Variable %in% clinical_predictors) %>%
    arrange(desc(Importance))
  
  vars <- VIP$Variable
  
  # Inner loop (i = 18:3)
  for (i in 18:3) {
    
    selected <- vars[1:i]
    train1 <- train[, c(selected, "covid_ed")]
    test1  <- test[,  c(selected, "covid_ed")]
    mod <- ranger(
      covid_ed ~ ., 
      data = train1,
      num.trees = 1000,
      probability = TRUE,
      num.threads = num_threads
    )
    
    preds_test <- predict(mod, test1)$predictions[, 2]
    r <- roc(test1$covid_ed, preds_test, quiet = TRUE)
    CI <- as.numeric(ci.auc(r))
    
    res_list[[idx]] <- tibble(
      i = i,
      CImin = CI[1],
      AUC   = CI[2],
      CImax = CI[3]
    )
    idx <- idx + 1L
  }
}

CI_df <- bind_rows(res_list)

averages <- CI_df %>%
  group_by(i) %>%
  summarize(
    Avg_CImin = mean(CImin),
    Avg_CImax = mean(CImax),
    Avg_AUC   = mean(AUC),
    .groups = "drop"
  )




mod=ranger(covid_ed~., data=covid, num.trees = 1000, probability=T)
VIP <- vip(object = mod, method = 'permute', target = 'covid_ed', metric = 'roc_auc', train= covid,
           pred_wrapper= pred_fun, event_level = 'second', num_features = 35)
VIP <- VIP$data



#### MODEL WITH CLINICAL + NON_CLINICAL PREDICTORS

set.seed(123)

merged_df <- read.csv('merged_df.csv')

# colnames(merged_df)
bin_vars <- c('SEX_FLG', 'DIABETES_FLG','CHRONIC_PULMONARY_FLG', 'RENAL_DISEASE_FLG',
              'LIVER_DX_FLG', 'OBESITY_FLG', 'CVA_FLG', 'HYPERTENSION_FLG',
              'NEURO_FLG', 'CAR_ARR_FLG')

merged_df[bin_vars] <- lapply(merged_df[bin_vars], function(x) if (!is.factor(x)) factor(x) else x)

merged_df$covid_ed <- factor(merged_df$covid_ed, levels = c("0", "1"))

merged_df$INSURANCE <- factor(merged_df$INSURANCE, levels=c('No Insurance', 'Public', 'Private'))

merged_df$COV_VAX_STATUS_CAT <- factor(merged_df$COV_VAX_STATUS_CAT, levels = c('0', '1-2', '3+'))


merged_df <- merged_df %>% dplyr::select(-c(X, County))

skim(merged_df)




clinical_predictors <- c('AGE_IN_YEARS','SEX_FLG', 'DIABETES_FLG',
                         'CHRONIC_PULMONARY_FLG', 'RENAL_DISEASE_FLG', 'LIVER_DX_FLG', 'OBESITY_FLG', 'CVA_FLG', 'HYPERTENSION_FLG',
                         'NEURO_FLG', 'CAR_ARR_FLG', 'COV_VAX_STATUS_CAT',
                         'RESPIRATORY_RATE', 'TEMPERATURE', 'HEARTRATE', 'SYSTOLIC', 'DIASTOLIC',
                         'SPO2')

non_clinical <- c('covid_ed',  "HCHO_avg" ,  "NO2_avg", "O3_avg" ,
                  "CO_avg" , "estimate" ,"INSURANCE", "LST_avg", 'UV_avg', 'Seasonal_sine', 'Seasonal_cosine'
)


pred_fun = function(X.model, newdata) {
  predict(X.model, newdata)$predictions[,2]
}


CI_df <- data.frame(
  i = integer(0),
  CImin = double(0),
  AUC = double(0),
  CImax = double(0)
)



N=dim(merged_df)[1]
CI=c()
for(each in 1:100){
  trn_idx = sample(1:N, .8*N)
  train=merged_df[trn_idx,]
  test=merged_df[-trn_idx,]
  mod=ranger(covid_ed~., data=train, num.trees = 1000, probability=T)
  VIP <- vip(object = mod, method = 'permute', target = 'covid_ed', metric = 'roc_auc', train= train,
             pred_wrapper= pred_fun, event_level = 'second', num_features = 45)
  VIP <- VIP$data
  
  for(i in 20:1) {
    selected_variables <- VIP %>%
      filter(Variable %in% clinical_predictors) %>%
      slice(1:i) %>%
      pull(Variable)
    
    train1 <- train[,c(selected_variables, non_clinical)]
    test1 <- test[,c(selected_variables, non_clinical)]
    mod=ranger(covid_ed~., data=train1, num.trees = 1000, probability=T)
    pred_train1=predict(mod,train1)$predictions[,2][which(train1$covid_ed==1)]
    pred_train0=predict(mod,train1)$predictions[,2][which(train1$covid_ed==0)]
    preds_test=predict(mod,test1)$predictions[,2]
    
    CI= c(as.numeric(ci(roc(test1$covid_ed, preds_test, plot=F))))
    CI_df <- rbind(CI_df, data.frame(i = i, CImin = CI[1], AUC = CI[2], CImax = CI[3]))
  }
}


averages <- CI_df %>%
  group_by(i) %>%
  summarize(
    Avg_CImin = mean(CImin),
    Avg_CImax = mean(CImax),
    Avg_AUC = mean(AUC)
  )




mod=ranger(covid_ed~., data=merged_df, num.trees = 1000, probability=T)
VIP <- vip(object = mod, method = 'permute', target = 'covid_ed', metric = 'roc_auc', train= merged_df,
           pred_wrapper= pred_fun, event_level = 'second', num_features = 35)
VIP <- VIP$data



## JUST TOP 3 PREDICTORS WITH LR

in_val_lr <- merged_df %>% dplyr::select(c(covid_ed, AGE_IN_YEARS,SPO2, RESPIRATORY_RATE))



set.seed(12)

# Initialize variables to store results
auc_results <- numeric(100)
lower_ci <- numeric(100)
upper_ci <- numeric(100)

for(i in 1:100){
  # Split the data
  index <- createDataPartition(in_val_lr$covid_ed, p=0.80, list=FALSE)
  train_set <- in_val_lr[index,]
  test_set <- in_val_lr[-index,]
  
  # Fit the full model first
  full_model <- glm(covid_ed ~ ., data = train_set, family = binomial)
  
  # Perform backward elimination
  step_model <- stepAIC(full_model, direction = "backward")
  
  # Make predictions
  prob_predictions <- predict(step_model, newdata = test_set, type = "response")
  roc_obj <- roc(test_set$covid_ed, prob_predictions)
  
  # Calculate AUC and 95% CI
  ci <- ci.auc(roc_obj)
  auc_results[i] <- auc(roc_obj)
  lower_ci[i] <- ci[1]
  upper_ci[i] <- ci[3]
}


# Calculate average AUC and CIs
avg_auc <- mean(auc_results)
avg_lower_ci <- mean(lower_ci)
avg_upper_ci <- mean(upper_ci)






