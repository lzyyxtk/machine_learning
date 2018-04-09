# Artificial Neural Network

# Importing the churn
churn = read.csv('/Users/leefrank/Desktop/CS/machine_learning/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Artificial_Neural_Networks/Churn_Modelling.csv')
churn = churn[4:14]

# Encoding the categorical variables as factors
churn$Geography = as.numeric(factor(churn$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
churn$Gender = as.numeric(factor(churn$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the churn into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(churn$Exited, SplitRatio = 0.8)
training_set = subset(churn, split == TRUE)
test_set = subset(churn, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Training set
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 100,
                         train_samples_per_iteration = -2)

# Predicting the Test set results
y_pred = h2o.predict(model, newdata = as.h2o(test_set[-11]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
cm
h2o.shutdown()
