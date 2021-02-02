# Citation - Some information regarding the scikit learn package was derived from: http://blog.yhat.com/posts/random-forests-in-python.html

from sklearn.ensemble import RandomForestClassifier;
import pandas as pd;
import numpy as np;
np.random.seed(0);

# Read data from file and divide into train and test
mushroomData = pd.read_csv('mushroom.data');
mushroomData = mushroomData.apply(lambda x: pd.factorize(x)[0]);
mushroomData['trainingData'] = np.random.uniform(0, 1, len(mushroomData)) <= .8;
train, test = mushroomData[mushroomData['trainingData']==True], mushroomData[mushroomData['trainingData']==False];

# Separate features and labels
features = mushroomData.columns[1:23];
labels = pd.factorize(train['class'])[0];

# Train random forest classifier
classifier = RandomForestClassifier();
classifier.fit(train[features], labels);

# Make predictions using test data
predictions = classifier.predict(test[features]);

# Produce confusion matrix
confMatrix = pd.crosstab(test['class'], predictions, rownames=['Actual Class'], colnames=['Predicted Class']);

# Calculate accuracy of prediction
misclassifiedCount = confMatrix[0][1] + confMatrix[1][0];
accuracy = 1 - (misclassifiedCount / len(test));

# Calculate probability of being poisoned
poisonMisclassified = float(confMatrix[0][1]);
predictedEdibleCount = confMatrix[0][1] + confMatrix[1][1];
poisonProbability = poisonMisclassified / predictedEdibleCount;

# Output results
print(confMatrix);
print();
print('Accuracy: {0}'.format(accuracy));

print('Probability of being poisoned based on prediction: {0}'.format(poisonProbability));
