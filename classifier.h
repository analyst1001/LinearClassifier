#include <vector>

using namespace std;

class DataItem{
	public:
		int classLabel;
		vector<float> feature;
};


class ConfusionMatrix {
	public:
	int *classes;
	int numClasses;
	long int **matrix;
	void print();
};

// Container for data feature vectors, class labels for each
// feature vector and functions to read and write data.
// Use constructor and destructor to initialize and clear data.
class Dataset{
	public:

  // Assume that the data file is plain text with each row
  // containing the class label followed by the features, 
  // separated by blank spaces.
  bool readData(char *filename);

  // Write data in the above format.
  bool writeData(char *filename);

  // Variables for data
  vector<DataItem*> data;
//  ...
};


// Partition 'complete' dataset randomly into 'folds' parts and 
// returns a pointer to an array of pointers to the partial datasets.
// seed is an argument to random number generator. The function can
// be used to divide data for training, testing and cross validation.
// This need not replicate the data.
Dataset** splitDataset(Dataset complete, int folds=2, int seed=0);

// Merge the datasets indexed by indicesToMerge in the toMerge list and return a
// single dataset. This need not replicate the data.
Dataset* mergeDatasets(Dataset** toMerge, int numDatasets, int* indicesToMerge);

// Class that carries out training and classification as well as
// store and read the learned model in/from a file.
class Results {
	public:
	int *classes;
	vector<float> *OVRRes;
	vector<float> **OVORes;
};
class LinearClassifier{
	public:

  // Loads classifier model from a file
  bool loadModel(char *modelfilename);

  // Saves the learned model parameters into a file
  bool saveModel(char *modelfilename);

  // learn the parameters of the classifier from possibly multiple training datasets
  // using a specific learning algorithm and combination strategy. 
  // The function should return the training error in [0,1].
  // Algorithms:
  //	1: Single Sample Perceptron Learning (fixed eta)
  //	2: Batch Perceptron Learning (variable eta)
  //	3: Single sample Relaxation (variable eta)
  //	4: Batch Relaxation Learning (variable eta)
  //	5: MSE using Pseudo Inverse
  //	6: MSE using LMS Procedure
  // Combination:
  //	1: 1 vs. Rest
  //	2: 1 vs. 1 with Majority voting
  //	3: 1 vs. 1 with DDAG
  //	4: BHDT.
  float learnModel(Dataset* trainData, int algorithm, int combination);
  
  // Classifies a DataItem and returns the class-label
  int classifySample(DataItem);

  // classify a set of testDataItems and return the error rate in [0,1].
  // Also fill the entries of the confusionmatrix.
  float classifyDataset(Dataset testSet, ConfusionMatrix &cm);
	int classifySampleOVRest(DataItem sample, int numClasses);
	int classifySampleOVO(DataItem sample, int numClasses);
	void OVRest(Dataset **classifiedDatasets, int *classes, int numClasses, int algorithm);
	void OVO(Dataset **classifiedDatasets, int *classes, int numClasses, int algorithm);
	int* findClasses(Dataset *dataset, int *numClasses);
	void initializeConfusionMatrix(ConfusionMatrix &cm);
	int findClassIndex(int classLabel);
  
  // Variables
  int numClasses;
  // Other variables to hold classifier parameters.
  //...
	int algorithm;
	int combination;
	Results result;
};


// Divide the dataset and performa an n-fold cross-validation. Compute the
// average error rate in [0,1]. Fill in the standard deviation and confusion matrix.
float crossValidate(Dataset complete, int folds, float &stdDev, ConfusionMatrix &cm, 
			int algo, int comb);


