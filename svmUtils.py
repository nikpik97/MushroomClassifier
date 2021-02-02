import numpy as np;
import matplotlib.pyplot as plt;

# Accuracy Plot Function
def plotAccuracy(regConst, regConstAccuracy):
    xData = range(len(regConstAccuracy));
    plt.plot(xData, regConstAccuracy);
    plt.xlabel('Steps');
    plt.ylabel('Accuracy');
    plt.title('Accuracy for λ = {0}'.format(regConst));
    plt.ylim([0.00,1.00])
    plt.savefig('accuracy_{0}.png'.format(regConst));
    plt.close();

# Helper function to split dataset into 2 sections
def splitDataset(dataset, countInA):
    randomIdx = np.arange(dataset.shape[0]);
    np.random.shuffle(randomIdx);
    return (dataset.iloc[randomIdx[0:countInA], :], dataset.iloc[randomIdx[countInA:], :]);
