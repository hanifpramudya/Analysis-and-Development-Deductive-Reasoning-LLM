data <-read.csv("/content/human_vs_gpt_3_5_proofwriter.csv")


# Effect of number of generalizations
# Human
model <- glm(correct_pred ~ humanvsgpt + type, data=data, family="binomial")
summary(model)
exp(cbind(OR = coef(model), confint(model)))

#human
model <- glm(correct_pred ~ type, data=subset(data, humanvsgpt==0), family="binomial")
summary(model)
exp(cbind(OR = coef(model), confint(model)))

#gpt
model <- glm(correct_pred ~ type, data=subset(data, humanvsgpt==1), family="binomial")
summary(model)
exp(cbind(OR = coef(model), confint(model)))