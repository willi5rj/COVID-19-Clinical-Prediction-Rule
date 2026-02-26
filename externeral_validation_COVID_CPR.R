library(dplyr)
library(caret)
library(mice)
library(haven)
library(missRanger)
library(ggplot2)
library(ranger) 
library(pROC)
library(ROCR)
library(skimr) 

setwd('~/Documents/covid_cepheid')

covid <- read.csv('covid_imputed.csv')

in_val_lr <- covid %>% dplyr::select(c(covid_ed, AGE_IN_YEARS,SPO2, RESPIRATORY_RATE))



in_val_lr$covid_ed <- as.factor(in_val_lr$covid_ed)

auc_results <- numeric(100)
lower_ci <- numeric(100)
upper_ci <- numeric(100)

for(i in 1:100){
  index <- createDataPartition(in_val_lr$covid_ed, p=0.80, list=FALSE)
  train_set <- in_val_lr[index,]
  test_set <- in_val_lr[-index,]
  model <- glm(covid_ed ~ AGE_IN_YEARS + SPO2 + RESPIRATORY_RATE,
               data = train_set, family = binomial)
  
  prob_predictions <- predict(model, newdata = test_set, type = "response")
  roc_obj <- roc(test_set$covid_ed, prob_predictions)
  
  ci <- ci.auc(roc_obj)
  auc_results[i] <- auc(roc_obj)
  lower_ci[i] <- ci[1]
  upper_ci[i] <- ci[3]
}


# Calculate average AUC and CIs
avg_auc <- mean(auc_results)
avg_lower_ci <- mean(lower_ci)
avg_upper_ci <- mean(upper_ci)

print(paste0("Average AUC: ", avg_auc))
print(paste0("Average 95% CI: [", avg_lower_ci, ", ", avg_upper_ci, "]"))


data <- read_dta('FinalDataset.dta')
d1 <- read_dta('PulseOx.dta')

d1<- data.frame(d1)

d2 <- data %>% left_join(d1, by = 'DeID' )

d3 <- d2 %>% dplyr::select(c(Hospitalized, Age, PulseOx, RespRate))

skim(d3)

d3 <- na.omit(d3)

summary(d3)


d3 <- rename(d3, 'covid_ed'='Hospitalized')
d3 <- rename(d3, 'AGE_IN_YEARS'='Age')
d3 <- rename(d3, 'SPO2'='PulseOx')
d3 <- rename(d3, 'RESPIRATORY_RATE'='RespRate')

attr(d3$covid_ed, "format.stata") <- NULL
attr(d3$AGE_IN_YEARS, "format.stata") <- NULL
attr(d3$SPO2, "format.stata") <- NULL
attr(d3$RESPIRATORY_RATE, "format.stata") <- NULL

attr(d3$AGE_IN_YEARS, "label") <- NULL
attr(d3$SPO2, "label") <- NULL
attr(d3$RESPIRATORY_RATE, "label") <- NULL

attr(d3, "na.action") <- NULL


d3$covid_ed <- as.factor(d3$covid_ed)



###excluding outliers

skim(d3)
skim(in_val_lr)

boxplot(d3$RESPIRATORY_RATE)
boxplot(d3$SPO2)
d3 <- d3 %>% filter(d3$RESPIRATORY_RATE<70)
d3 <- d3 %>% filter(d3$SPO2>60)



skim(d3)

boxplot(in_val_lr$RESPIRATORY_RATE)

in_val_lr <- in_val_lr %>% filter(in_val_lr$RESPIRATORY_RATE<70)
in_val_lr <- in_val_lr %>% filter(in_val_lr$RESPIRATORY_RATE>2)

boxplot(in_val_lr$SPO2)
in_val_lr <- in_val_lr %>% filter(in_val_lr$SPO2>60)


skim(in_val_lr)
skim(d3)







####COMPARING DISTRIBUTION#####
d3$Source <- "d3"
in_val_lr$Source <- "in_val_lr"
combined_df <- rbind(d3, in_val_lr)

# Plot for AGE_IN_YEARS
ggplot(combined_df, aes(x = AGE_IN_YEARS, fill = Source)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of AGE_IN_YEARS", x = "AGE_IN_YEARS", y = "Density") +
  theme_minimal()




# Plot for SPO2
ggplot(combined_df, aes(x = SPO2, fill = Source)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of SPO2", x = "SPO2", y = "Density") +
  theme_minimal()

# Plot for RESPIRATORY_RATE
ggplot(combined_df, aes(x = RESPIRATORY_RATE, fill = Source)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of RESPIRATORY_RATE", x = "RESPIRATORY_RATE", y = "Density") +
  theme_minimal()





d3 <- d3 %>% dplyr::select(c(covid_ed, AGE_IN_YEARS, SPO2, RESPIRATORY_RATE))
in_val_lr <- in_val_lr %>% dplyr::select(c(covid_ed, AGE_IN_YEARS, SPO2, RESPIRATORY_RATE))




# Fit  model on the internal dataset
fit <- glm(covid_ed ~ AGE_IN_YEARS + SPO2 + RESPIRATORY_RATE, data = in_val_lr, family = binomial())
summary(fit)

# Calculate probabilities
probabilities <- predict(fit, newdata = d3, type = "response")

roc_obj <- roc(d3$covid_ed, probabilities)

auc(roc_obj)

# Calculate 95% Confidence Interval for AUC
ci(roc_obj, method = "bootstrap")


boxplot(d3$RESPIRATORY_RATE)


skim(d3)
skim(in_val_lr)


### REVERSE IN/EX VAL Datasets

# Fit  model on the internal dataset
fit <- glm(covid_ed ~ AGE_IN_YEARS + SPO2 + RESPIRATORY_RATE, data = d3, family = binomial())
summary(fit)

# Calculate probabilities using the model
probabilities <- predict(fit, newdata = in_val_lr, type = "response")

roc_obj <- roc(in_val_lr$covid_ed, probabilities)

auc(roc_obj)

# Calculate 95% Confidence Interval for AUC
ci(roc_obj, method = "bootstrap")














ex_val_lr <- d3

library(pROC)
library(rms)
library(gridExtra)
library(ggplot2)
library(ggplotify)
library(isotone)


# Fit the logistic regression model on the internal validation dataset
model <- glm(covid_ed ~ AGE_IN_YEARS + RESPIRATORY_RATE + SPO2, data = in_val_lr, family = binomial)

ex_val_lr$predicted_prob <- predict(model, newdata = ex_val_lr, type = "response")

roc_obj <- roc(ex_val_lr$covid_ed, ex_val_lr$predicted_prob)
roc_plot <- ggplotify::as.ggplot(function() plot(roc_obj, print.auc=TRUE))


cal_obj <- lrm(covid_ed ~ predicted_prob, data = ex_val_lr, x = TRUE, y = TRUE)
cal_plot <- calibrate(cal_obj, method = "boot", B = 100)
calibration_plot <- ggplotify::as.ggplot(function() plot(cal_plot))

hist_plot <- ggplot(ex_val_lr, aes(x=predicted_prob)) + geom_histogram(binwidth=0.01)

hist_plot

p <- grid.arrange(calibration_plot, hist_plot, nrow = 1)

p

actual_outcome <- as.numeric(as.character(ex_val_lr$covid_ed))
predicted_prob <- ex_val_lr$predicted_prob

calibration_model <- lm(actual_outcome ~ predicted_prob)

slope <- coef(calibration_model)[2]
intercept <- coef(calibration_model)[1]



p




library(ggplot2)
library(dplyr)



calibration_data <- ex_val_lr %>%
  mutate(actual_outcome = as.numeric(as.character(covid_ed)))

calibration_data <- calibration_data %>%
  mutate(bin = cut(predicted_prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE))

# Calculate mean predicted probability and mean actual outcome for each bin
calibration_summary <- calibration_data %>%
  group_by(bin) %>%
  summarise(mean_predicted_prob = mean(predicted_prob),
            mean_actual_outcome = mean(actual_outcome))

a <- ggplot(calibration_summary, aes(x = mean_predicted_prob, y = mean_actual_outcome)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "orchid3") +
  geom_smooth(method = "lm", se = FALSE, color = "skyblue") +
  labs(x = "Mean Predicted Probability", y = "Mean Actual Outcome",
       title = "Calibration Plot of Predicted Probabilites\nfor COVID-19 Hospitalizations") +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    plot.title = element_text(size = 16, face = "bold")
  )

a


ggsave('cal_plot.pdf', plot = last_plot(), device = 'pdf', width=7, height =5, dpi=600)



# Density plot of predicted probabilities
 b<- ggplot(ex_val_lr, aes(x = predicted_prob)) +
  geom_density(fill = "lightgreen", alpha = 0.6) +
  labs(title = "Density of Predicted Probabilities",
       x = "Predicted Probability",
       y = "Density") +
  theme_minimal() +
   theme(
     axis.title = element_text(size = 16),
     axis.text = element_text(size = 14),
     plot.title = element_text(size = 16, face = "bold")
   )
 


b

library(cowplot)

plotab <- plot_grid(a,b, labels = c('A', 'B'), nrow=2 )
ggsave('cal_plot_density.pdf', plot = last_plot(), device = 'pdf', width=8, height =6, dpi=600)


# Load necessary packages
library(dcurves)
library(dplyr)
library(ggplot2)

# Select relevant columns from the datasets
d3 <- d3 %>% dplyr::select(c(covid_ed, AGE_IN_YEARS, SPO2, RESPIRATORY_RATE))
in_val_lr <- in_val_lr %>% dplyr::select(c(covid_ed, AGE_IN_YEARS, SPO2, RESPIRATORY_RATE))

# Fit the model on the internal dataset
fit <- glm(covid_ed ~ AGE_IN_YEARS + SPO2 + RESPIRATORY_RATE, data = in_val_lr, family = binomial())
summary(fit)

# Calculate probabilities using the model
probabilities <- predict(fit, newdata = d3, type = "response")

# Create a data frame with the predicted probabilities and actual outcomes
decision_data <- d3 %>%
  mutate(predicted_prob = probabilities)

# Perform decision curve analysis
dca_result <- dca(covid_ed ~ predicted_prob, data = decision_data)

# Plot the decision curve
dca_plot <- plot(dca_result)

# Rename the legend
dca_plot <- dca_plot + 
  scale_color_manual(name = "Predicted Probability", values = c("predicted_prob" = "blue"))

# Rename the legend for predicted_prob while keeping Treat All and Treat None lines
dca_plot <- dca_plot + 
  scale_color_manual(name = "", 
                     values = c("predicted_prob" = "skyblue", "Treat All" = "orchid3", "Treat None" = "forestgreen"),
                     labels = c("predicted_prob" = "Predicted Probability", "Treat All" = "Treat All", "Treat None" = "Treat None"))
# Display the plot
print(dca_plot)





# Save the plot
ggsave('dca_plot.pdf', plot = dca_plot, device = 'pdf', width = 6, height = 4, dpi = 600)


net_benefit_data <- dca_result$dca$net_benefit
threshold <- dca_result$dca$threshold

# Combine the net benefit and threshold data into a data frame
net_benefit_df <- data.frame(threshold = threshold, net_benefit = net_benefit_data)

# Find the threshold probability that corresponds to a net benefit of 0.01
threshold_prob <- net_benefit_df %>%
  filter(net_benefit >= 0.00) %>%
  slice(1) %>%
  pull(threshold)




##### PERFORMANCE METRICS #################
##### PERFORMANCE METRICS #################
##### PERFORMANCE METRICS #################
##### PERFORMANCE METRICS #################
##### PERFORMANCE METRICS #################
##### PERFORMANCE METRICS #################
##### PERFORMANCE METRICS #################
##### PERFORMANCE METRICS #################
##### PERFORMANCE METRICS #################



# Check the first few predicted probabilities
head(probabilities)

# Apply threshold again carefully
predicted_class <- ifelse(probabilities >= as.numeric(threshold_90sens), 1, 0)

# Check the result
table(predicted_class)
length(predicted_class)
length(d3$covid_ed)



# Convert actual outcome to numeric
actual_outcome <- as.numeric(as.character(d3$covid_ed))


# True Positives (TP): predicted 1, actual 1
TP <- sum(predicted_class == 1 & actual_outcome == 1)

# False Negatives (FN): predicted 0, actual 1
FN <- sum(predicted_class == 0 & actual_outcome == 1)

# False Positives (FP): predicted 1, actual 0
FP <- sum(predicted_class == 1 & actual_outcome == 0)

# True Negatives (TN): predicted 0, actual 0
TN <- sum(predicted_class == 0 & actual_outcome == 0)

# Now construct the matrix in the correct format
conf_matrix <- matrix(c(TP, FP, FN, TN), 
                      nrow = 2, byrow = TRUE,
                      dimnames = list(
                        Test = c("Test +", "Test -"),
                        Outcome = c("Outcome +", "Outcome -")
                       
                      ))



# Check the matrix
print(conf_matrix)

# Run epi.tests
library(epiR)
results <- epi.tests(conf_matrix)

# View results
print(results)


##### PERFORMANCE METRICS #################
##### SET SENSITIVITY #################
##### PERFORMANCE METRICS #################
##### SET SENSITIVITY#################
##### PERFORMANCE METRICS #################
#####SET SENSITIVITY #################
##### SET SENSITIVITY #################
##### PERFORMANCE METRICS #################
##### SET SENSITIVITY #################

library(pROC)
library(epiR)

# Step 1: Get all thresholds and sensitivities from ROC
roc_obj <- roc(actual_outcome, probabilities)

# Step 2: Find threshold where sensitivity is closest to 90%
roc_coords <- coords(roc_obj, x = "all", ret = c("threshold", "sensitivity", "specificity"), transpose = FALSE)
target_sens <- 0.90
closest_row <- which.min(abs(roc_coords$sensitivity - target_sens))
threshold_90sens <- roc_coords$threshold[closest_row]

# Step 3: Classify using this threshold
predicted_class <- ifelse(probabilities >= threshold_90sens, 1, 0)

# Step 4: Build confusion matrix manually
TP <- sum(predicted_class == 1 & actual_outcome == 1)
FN <- sum(predicted_class == 0 & actual_outcome == 1)
FP <- sum(predicted_class == 1 & actual_outcome == 0)
TN <- sum(predicted_class == 0 & actual_outcome == 0)

conf_matrix <- matrix(c(TP, FP, FN, TN), 
                      nrow = 2, byrow = TRUE,
                      dimnames = list(
                        Test = c("Test +", "Test -"),
                        Outcome = c("Outcome +", "Outcome -")
                      ))

results <- epi.tests(conf_matrix)

cat("Threshold for ~90% sensitivity:", threshold_90sens, "\n")
print(conf_matrix)
print(results)


##### PERFORMANCE METRICS #################
##### SET THRESHOLD #################
##### PERFORMANCE METRICS #################
##### SET THRESHOLD#################
##### PERFORMANCE METRICS #################
#####SET THRESHOLD #################
##### SET THRESHOLD #################
##### PERFORMANCE METRICS #################
##### SET THRESHOLD #################


probabilities


library(epiR)

# Step 1: Set threshold to 1%
threshold_01 <- 0.01

# Step 2: Classify using this threshold
predicted_class <- ifelse(probabilities >= threshold_01, 1, 0)

# Step 3: Convert actual outcome to numeric
actual_outcome <- as.numeric(as.character(d3$covid_ed))

# Step 4: Build confusion matrix manually
TP <- sum(predicted_class == 1 & actual_outcome == 1)
FN <- sum(predicted_class == 0 & actual_outcome == 1)
FP <- sum(predicted_class == 1 & actual_outcome == 0)
TN <- sum(predicted_class == 0 & actual_outcome == 0)

# Step 5: Construct matrix in correct format
conf_matrix <- matrix(c(TP, FP, FN, TN), 
                      nrow = 2, byrow = TRUE,
                      dimnames = list(
                        Test = c("Test +", "Test -"),
                        Outcome = c("Outcome +", "Outcome -")
                      ))

# Step 6: Run epi.tests
results <- epi.tests(conf_matrix)

# Step 7: View results
cat("Threshold used: 0.01\n")
print(conf_matrix)
print(results)






