import pandas as pd;
import numpy as np;
import sys;
from svmUtils import *;

# Prediction Function for SVM
def predictLabel(features, a, b):
  signedPrediction = np.sign(np.dot(a, features) + b);
  return int(signedPrediction);

# Update function to perform Stochastic Gradient Descent
def gradientDescent(label, features, a, b, regConst, stepLength):
  lossAmount = label * (predictLabel(features, a, b));

  if (lossAmount >= 1):
    updatedA = a - (stepLength * (regConst * a));
    return (updatedA, b);
    
  updatedA = a - (stepLength * ((regConst * a) - (label * features)));
  updatedB = b - (stepLength * (-label));
  return (updatedA, updatedB);

# Function to determine accuracy of predicted data
def calculateAccuracy(labels, rows, a, b):
  actualLabels = labels[rows.index];
  predictedLabels = (rows.apply(lambda x: predictLabel(x, a, b), axis=1)).values;

  correctPredictionCount = float(np.sum(actualLabels == predictedLabels));
  return (correctPredictionCount / len(actualLabels));

# Support Vector Machine (SVM) implementation
def trainSVM(dataset, validationSet, labels, regConst, a, b, constAccuracy):
  p = 1.0;
  q = 50.0;
  bestA = a;
  bestB = b;
  maxAccuracy = -sys.maxsize;
  regConstAccuracy = [];

  for epoch in range(50):
    stepLength = p / ((epoch * 0.01) + q);

    for step in range(100):
      currRow = dataset.sample().iloc[0];
      currLabel = labels[currRow.name];

      a, b = gradientDescent(currLabel, currRow.values, a, b, regConst, stepLength);

      # Calculate accuracy on validation set every 10 steps
      if (step % 10 == 0):
        accuracy = calculateAccuracy(labels, validationSet, a, b);
        regConstAccuracy.append(accuracy);
        # If current accuracy is best one yet, record A and B values
        if (accuracy > maxAccuracy):
          bestA = a;
          bestB = b;
          maxAccuracy = accuracy;

  constAccuracy.append(regConstAccuracy);
  return (bestA, bestB, maxAccuracy);

if __name__ == '__main__':
  # Read data from file
  df = pd.read_csv('wdbc.data', header=None);

  # Drop irrelevant columns
  df_norm = df.drop(columns=[0, 1]);
  df_norm.drop(columns=df_norm.columns[10:32], inplace=True);

  # Normalize the data
  df_norm = (df_norm  - df_norm.mean()) / df_norm.std()

  # Factorize labels as -1 (Malignant) and 1 (Benign)
  labels = df[1].factorize()[0];
  factorFunc = np.vectorize(lambda x: -1 if x == 0 else 1);
  labels = factorFunc(labels);

  # Randomly divide dataset into training, test, and validation data
  train, test, validation = np.split(df_norm.sample(frac=1), [int(0.65*len(df_norm)), int(0.82*len(df_norm))]);

  # Zero values for initial a and b
  a = np.zeros(shape=10);
  b = np.zeros(shape=1)[0];

  # Find best value of regularization parameter
  regConsts = [0.001, 0.01, 0.1, 1.0];
  constAccuracy = [];
  maxAccuracy = -sys.maxsize;
  bestConst = None;
  bestA = None;
  bestB = None;

  # Find regularization constant that performs best on validation set
  for const in regConsts:
    # Train classifier
    currA, currB, accuracy = trainSVM(train, validation, labels, const, a, b, constAccuracy);

    # If current regularization parameter had better accuracy, record it
    if (accuracy > maxAccuracy):
        maxAccuracy = accuracy;
        bestConst = const;
        bestA = currA;
        bestB = currB;

  # Part A: Plot of accuracy every 10 steps
  print('Part A: Producing accuracy plots for regularization parameters');
  for i in range(len(regConsts)):
    plotAccuracy(regConsts[i], constAccuracy[i]);

  # Part B: Best option for regularization parameter
  print('Part B: Best regularization parameter = {0}'.format(bestConst));

  # Part C: Find accuracy of best classifier on test data
  print('Part C: {0}'.format(calculateAccuracy(labels, test, bestA, bestB)));
