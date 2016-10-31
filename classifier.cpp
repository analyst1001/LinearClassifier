#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include "newmat/include.h"
#include "newmat/newmat.h"
#include "newmat/newmatio.h"
#include "classifier.h"

#define MAXA 5

using namespace std;

bool Dataset::readData(char *filename) {
	DataItem *tempItem = new DataItem();
	char temp;
	float tempFeature;
	FILE *fp = fopen(filename, "r");
	if (fp == NULL) {
		return false;
	}
	bool done = false;
	done = (fscanf(fp, "%d", &tempItem->classLabel) == EOF);
	while (!done) {
		fscanf(fp, "%c", &temp);
		while (temp != '\n') {
			fscanf(fp, "%f", &tempFeature);
			tempItem->feature.push_back(tempFeature);
			done = (fscanf(fp, "%c", &temp) == EOF);
		}
		data.push_back(tempItem);
		tempItem = new DataItem();
		done = (fscanf(fp, "%d", &tempItem->classLabel) == EOF);
		if (done) {
			delete tempItem;
		}
	}

	fclose(fp);
	return true;
}

bool Dataset::writeData(char *filename) {
	FILE *fp = fopen(filename, "w");
	if (fp == NULL) {
		return false;
	}
	vector<DataItem*>::iterator dataIt;
	vector<float>::iterator featureIt;
	for (dataIt = data.begin(); dataIt < data.end(); dataIt++) {
		fprintf(fp, "%d ", (*dataIt)->classLabel);
		for (featureIt = (*dataIt)->feature.begin(); featureIt < (*dataIt)->feature.end() - 1; featureIt++) {
			fprintf(fp, "%.1f ", *featureIt);
		}
		featureIt = (*dataIt)->feature.end() - 1;
		fprintf(fp, "%f\n", *featureIt);
	}
	fclose(fp);
	return true;
}

Dataset** splitDataset(Dataset complete, int folds, int seed) {
	srandom(seed);
	Dataset **partialDatasets = (Dataset **) calloc(folds, sizeof(Dataset *));
	for (int i = 0; i < folds; i++) {
		partialDatasets[i] = new Dataset();
	}
	vector<DataItem*>::iterator dataIt;
	long int srcIndex;
	int destIndex = 0;
	while (complete.data.size() > 0) {
			srcIndex = random() % complete.data.size(); 
			partialDatasets[destIndex]->data.push_back(complete.data.back());
			complete.data.pop_back();
			destIndex = (destIndex + 1) % folds;
	}
	return partialDatasets;
}

Dataset* mergeDatasets(Dataset** toMerge, int numDatasets, int *indicesToMerge) {
	Dataset *mergedDataset = new Dataset();
	vector<DataItem*>::iterator dataIt;
	int i;
	for (i = 0; i < numDatasets; i++) {
		if (indicesToMerge[i] >= 0) {
			printf("Merging data of %d set\n", indicesToMerge[i]);
			vector<DataItem*>::iterator dataIt;
			for (dataIt = toMerge[indicesToMerge[i]]->data.begin(); dataIt < toMerge[indicesToMerge[i]]->data.end(); dataIt++) {
//				printf("\tClasslabel %d\n", (*dataIt)->classLabel);
				mergedDataset->data.push_back(*dataIt);
			}
		//	while (toMerge[indicesToMerge[i]]->data.size() > 0) {
	//			mergedDataset->data.push_back(toMerge[indicesToMerge[i]]->data.back());
//				toMerge[indicesToMerge[i]]->data.pop_back();  //uncomment to empty original data
		//	}
		}
	}
	return mergedDataset;
}

bool LinearClassifier::loadModel(char *modelfilename) {

	return true;
}

bool LinearClassifier::saveModel(char *modelfilename) {

	return true;
}

void copyFeatures(vector<float> feature, float *y, bool negate) {
	int i;
	vector<float>::iterator featureIt;
	if (negate) {
		y[0] = -1;
	}
	else {
		y[0] = 1;
	}
	i = 1;
//	printf("features: ");
	for (featureIt = feature.begin(); featureIt < feature.end(); featureIt++) {
		y[i] = (*featureIt);
		if (negate) {
			y[i] *= -1;
		}
//		printf("%f ", y[i]);
		i++;
	}
//	printf("\n");
	return;
}

float dotProduct(float *a, float *b, int n) {
	float result = 0.0;
	for (int i = 0; i < n; i++) {
		result += (a[i] * b[i]);
	}
	return result;
}

void subtractFromFirst(float *a, float *b, int n) {
	for (int i = 0; i < n; i++) {
		a[i] -= b[i];
	}
	return;
}

void addToFirst(float *a, float *b, int n) {
	for (int i = 0; i < n; i++) {
		a[i] += b[i];
	}
	return;
}

vector<float> SSPLFI(Dataset *trainDataClassA, Dataset *trainDataClassB) {
	int i;
	int noOfFeatures = trainDataClassA->data.back()->feature.size();
	float *a = new float[noOfFeatures + 1];
	float *y = new float[noOfFeatures + 1];
	long int noOfPointsA = trainDataClassA->data.size();
	long int noOfPointsB = trainDataClassB->data.size();
//	printf("noOfPointsA %ld, noOfPointsB %ld\n", noOfPointsA, noOfPointsB);
	srand(time(NULL));
	for (i = 0; i < noOfFeatures + 1; i++) {
		a[i] = -1.0 * (float) ((rand() % MAXA) + 1);
		y[i] = 0.0;
	}
	long int k = 0;
	long int correctTillNow = 0;
	float tempDotProduct = 0.0;
	while ((correctTillNow <= (noOfPointsA + noOfPointsB)) && (k < 1000000)) {
		tempDotProduct = 0.0;
		if ((k%(noOfPointsA + noOfPointsB)) < noOfPointsA) {
			//printf("In positive %ld\n", k);
			copyFeatures(trainDataClassA->data[(k % (noOfPointsA + noOfPointsB))]->feature, y, false);
		}
		else {
			//printf("In negative %ld\n", k-noOfPointsA);
			copyFeatures(trainDataClassB->data[(k%(noOfPointsA + noOfPointsB)) - noOfPointsA]->feature, y, true);
		}
		if ((tempDotProduct = dotProduct(a, y, noOfFeatures + 1)) <= 0.0) {
			correctTillNow = 0;
			addToFirst(a, y, noOfFeatures + 1);
		}
		else {
			correctTillNow += 1;
		}
		printf("dot prod = %f correctTillNow %ld %ld\n", tempDotProduct, correctTillNow, k);
		k = (k + 1);
	}
	vector<float> retVal;
	for (i = 0; i < noOfFeatures + 1; i++) {
		retVal.push_back(a[i]);
	}
	return retVal;
}

void scalarProduct(float scalar, float *vec, int n) {
	for (int i = 0; i < n; i++) {
		vec[i] *= scalar;
	}
	return;
}

float magnitude(float *vec, int n) {
	float result = 0.0;
	for (int i = 0; i < n; i++) {
		result += (vec[i]*vec[i]);
	}
	return sqrtf(result);
}

vector<float> BPLVI(Dataset *trainDataClassA, Dataset *trainDataClassB) {
	int i;
	int noOfFeatures = trainDataClassA->data.back()->feature.size();
	float *a = new float[noOfFeatures + 1];
	float *y = new float[noOfFeatures + 1];
	float *correction = new float[noOfFeatures + 1];
	srand(time(NULL));
	for (i = 0; i < noOfFeatures + 1; i++) {
		a[i] = -1.0 * (float) ((rand() % MAXA) + 1);
	}
	long int k = 0;
	float eta = 1.0;
	float margin = 0.5;
	float criterion = 0.005;
	vector<DataItem*>::iterator dataIt;
	do {
		k = k + 1;
		eta = (1.00 / k);
		for (i = 0; i < noOfFeatures + 1; i++) {
			correction[i] = 0.0;
		}
		for (dataIt = trainDataClassA->data.begin(); dataIt < trainDataClassA->data.end(); dataIt++) {
			copyFeatures((*dataIt)->feature, y, false);
			if (dotProduct(a, y, noOfFeatures + 1) <= margin) {
				addToFirst(correction, y, noOfFeatures + 1);
			}
		}
		for (dataIt = trainDataClassB->data.begin(); dataIt < trainDataClassB->data.end(); dataIt++) {
			copyFeatures((*dataIt)->feature, y, true);
			if (dotProduct(a, y, noOfFeatures + 1) <= margin) {
				addToFirst(correction, y, noOfFeatures + 1);
			}
		}
		scalarProduct(eta, correction, noOfFeatures + 1);	
		addToFirst(a, correction, noOfFeatures + 1);
	} while(magnitude(correction, noOfFeatures + 1) >= criterion);

	vector<float> retVal;
	for (i = 0; i < noOfFeatures + 1; i++) {
		retVal.push_back(a[i]);
	}
	return retVal;
}

vector<float> SSRVI(Dataset *trainDataClassA, Dataset *trainDataClassB) {
	int i;
	int noOfFeatures = trainDataClassA->data.back()->feature.size();
	float *a = new float[noOfFeatures + 1];
	float *y = new float[noOfFeatures + 1];
	long int noOfPointsA = trainDataClassA->data.size();
	long int noOfPointsB = trainDataClassB->data.size();
	srand(time(NULL));
	for (i = 0; i < noOfFeatures + 1; i++) {
		a[i] = -1.0 * (float) ((rand() % MAXA) + 1);
	}
	long int k = 1;
	long int correctTillNow = 0;
	float eta = 1.0;
	float margin = 0.5;
	float criterion = 0.05;
	float *correction = new float[noOfFeatures + 1];
	float tempDotProduct = 0.0;
	printf("noOfPointsA %ld, noOfPointsB %ld\n", noOfPointsA, noOfPointsB);
	while (correctTillNow <= (noOfPointsA + noOfPointsB)) {
		eta = 1.0 / k;
		if ((k % (noOfPointsA + noOfPointsB)) < noOfPointsA) {
			printf("In positive\n");
			copyFeatures(trainDataClassA->data[(k % (noOfPointsA + noOfPointsB))]->feature, y, false);
		}
		else {
			printf("In negative\n");
			copyFeatures(trainDataClassB->data[(k % (noOfPointsA + noOfPointsB)) - noOfPointsA]->feature, y, true);
		}
		if (((tempDotProduct = dotProduct(a, y, noOfFeatures + 1)) + margin) < 0.000) {
			correctTillNow = 0;
			float magnitudeSquared = 0.0;
			for (i = 0; i < noOfFeatures + 1; i++) {
				correction[i] = y[i];
				magnitudeSquared += (y[i] * y[i]);
			}
			scalarProduct((margin - tempDotProduct), correction, noOfFeatures + 1);
			scalarProduct(eta * (1.0/magnitudeSquared), correction, noOfFeatures + 1);
			addToFirst(a, correction, noOfFeatures + 1);
			printf("changed a\n");
		}
		else {
			correctTillNow += 1;
		}
		printf("dot prod = %f, correctTillNow %ld\n", tempDotProduct, correctTillNow);
		if (magnitude(correction, noOfFeatures + 1) < criterion) {
			break;
		}
		k = (k + 1);
	}
	vector<float> retVal;
	for (i = 0; i < noOfFeatures + 1; i++) {
		retVal.push_back(a[i]);
	}
	return retVal;
}

vector<float> MSEUPI(Dataset *trainDataClassA, Dataset *trainDataClassB) {
	int i;
	int noOfFeatures = trainDataClassA->data.back()->feature.size();
	long int noOfPointsA = trainDataClassA->data.size();
	long int noOfPointsB = trainDataClassB->data.size();
	/*float **y = new float*[noOfPointsA + noOfPointsB];
	for (i = 0; i < (noOfPointsA + noOfPointsB); i++) {
		y[i] = new float[noOfFeatures + 1];
	}*/
	Matrix margin(noOfPointsA + noOfPointsB, 1);
	for (i = 1; i <= (noOfPointsA + noOfPointsB); i++) {
		margin(i, 1) = 2.0;
	}
	long int k = 1;
	long int j = 1;
	vector<DataItem*>::iterator dataIt;
	vector<float>::iterator featureIt;
	cout<<noOfPointsA<<" "<<noOfPointsB<<endl;
	Matrix A(noOfPointsA + noOfPointsB, noOfFeatures + 1);
	for (dataIt = trainDataClassA->data.begin(); dataIt < trainDataClassA->data.end(); dataIt++) {
		A(k, 1) = 1.00;
		j = 2;
		for (featureIt = (*dataIt)->feature.begin(); featureIt < (*dataIt)->feature.end(); featureIt++) {
			A(k,j) = (*featureIt);
			j++;
		}
		k++;
	}
	for (dataIt = trainDataClassB->data.begin(); dataIt < trainDataClassB->data.end(); dataIt++) {
		A(k, 1) = -1.00;
		j = 2;
		for (featureIt = (*dataIt)->feature.begin(); featureIt < (*dataIt)->feature.end(); featureIt++) {
			A(k,j) = -1 * (*featureIt);
			j++;
		}
		k++;
	}
	Matrix B = A.t();
//	cout<<A<<endl<<B<<endl;
	Matrix C = B * A;
//	cout<<C<<endl;
	vector<float> retVal;
	try {
		Matrix D = C.i();
	//	cout<<D<<endl;
		Matrix E = D * B;
	//	cout<<B<<endl<<E<<endl;
		Matrix ans = E * margin;
	//	cout<<ans<<endl;
		for (i = 1; i <= noOfFeatures + 1; i++) {
			retVal.push_back(ans(i, 1));
		}
	}
	catch (SingularException e) {
		cout<<"Matrix is Singular"<<endl;
	}
	return retVal;
}

vector<float> BRLVI(Dataset *trainDataClassA, Dataset *trainDataClassB) {
	int i;
	int noOfFeatures = trainDataClassA->data.back()->feature.size();
	float *a = new float[noOfFeatures + 1];
	float *y = new float[noOfFeatures + 1];
	float *correction = new float[noOfFeatures + 1];
	float criterion = 0.05;
	srand(time(NULL));
	for (i = 0; i < noOfFeatures + 1; i++) {
		a[i] = -1.0 * (float) ((rand() % MAXA) + 1);
	}
	long int k = 0;
	float eta = 1.0;
	float margin = 0.0;
	//float criterion = 1.0;
	float *singleCorrection = new float[noOfFeatures + 1];
	bool noError = true;
	vector<DataItem*>::iterator dataIt;
	do {
		k = k + 1;
		eta = (1.00 / k);
		noError = true;
		for (i = 0; i < noOfFeatures + 1; i++) {
			correction[i] = 0.0;
		}
		for (dataIt = trainDataClassA->data.begin(); dataIt < trainDataClassA->data.end(); dataIt++) {
			copyFeatures((*dataIt)->feature, y, false);
			if (dotProduct(a, y, noOfFeatures + 1) <= margin) {
				float magnitudeSquared = 0.0;
				noError = false;
				for (i = 0; i < noOfFeatures + 1; i++) {
					singleCorrection[i] = y[i];
					magnitudeSquared += (y[i] * y[i]);
				}
				scalarProduct((eta * (margin - dotProduct(a, y, noOfFeatures + 1)) / magnitudeSquared), singleCorrection, noOfFeatures + 1);

				addToFirst(correction, singleCorrection, noOfFeatures + 1);
			}
		}
		for (dataIt = trainDataClassB->data.begin(); dataIt < trainDataClassB->data.end(); dataIt++) {
			copyFeatures((*dataIt)->feature, y, true);
			if (dotProduct(a, y, noOfFeatures + 1) <= margin) {
				float magnitudeSquared = 0.0;
				noError = false;
				for (i = 0; i < noOfFeatures + 1; i++) {
					singleCorrection[i] = y[i];
					magnitudeSquared += (y[i] * y[i]);
				}
				scalarProduct((eta * (margin - dotProduct(a, y, noOfFeatures + 1)) / magnitudeSquared), singleCorrection, noOfFeatures + 1);
				addToFirst(correction, singleCorrection, noOfFeatures + 1);
			}
		}
		addToFirst(a, correction, noOfFeatures + 1);
	} while((!noError) && (magnitude(correction, noOfFeatures + 1) >= criterion));

	vector<float> retVal;
	for (i = 0; i < noOfFeatures + 1; i++) {
		retVal.push_back(a[i]);
	}
	return retVal;
}

vector<float> MSEULMS(Dataset *trainDataClassA, Dataset *trainDataClassB) {
	int i;
	int noOfFeatures = trainDataClassA->data.back()->feature.size();
	float *a = new float[noOfFeatures + 1];
	float *y = new float[noOfFeatures + 1];
	long int noOfPointsA = trainDataClassA->data.size();
	long int noOfPointsB = trainDataClassB->data.size();
	float *correction = new float[noOfFeatures + 1];
	srand(time(NULL));
	printf("noOfPointsA %ld, noOfPointsB %ld\n", noOfPointsA, noOfPointsB);
	for (i = 0; i < noOfFeatures + 1; i++) {
		a[i] = -1.0 * (float) ((rand() % MAXA) + 1);
	}
	long int k = 0;
	float eta = 1.0;
	float *margin = new float[noOfPointsA + noOfPointsB];
	for (i = 0; i < (noOfPointsA + noOfPointsB); i++) {
		margin[i] = 1.0;
	}
	float criterion = 1.0;
	float tempDotProduct = 0.0;
	vector<DataItem*>::iterator dataIt;
	do {
		k = k + 1;
		eta = (1.00 / k);
		if ((k % (noOfPointsA + noOfPointsB)) < noOfPointsA) {
			printf("In positive\n");
			copyFeatures(trainDataClassA->data[(k % (noOfPointsA + noOfPointsB))]->feature, y, false);
		}
		else {
			printf("In negative\n");
			copyFeatures(trainDataClassB->data[(k % (noOfPointsA + noOfPointsB)) - noOfPointsA]->feature, y, true);
		}
//		if ((tempDotProduct = dotProduct(a, y, noOfFeatures + 1)) != margin[(k % (noOfPointsA + noOfPointsB))]) {
			for (i = 0; i < noOfFeatures + 1; i++) {
				correction[i] = y[i];
	//		}
			tempDotProduct = dotProduct(a, y, noOfFeatures + 1);
			printf("dot product = %f\n", tempDotProduct);
			scalarProduct((eta * (margin[(k % (noOfPointsA + noOfPointsB))] - tempDotProduct)), correction, noOfFeatures + 1);
			addToFirst(a, correction, noOfFeatures + 1);
		}
	} while((magnitude(correction, noOfFeatures + 1) >= criterion));

	vector<float> retVal;
	for (i = 0; i < noOfFeatures + 1; i++) {
		retVal.push_back(a[i]);
	}
	return retVal;
}

Dataset **splitByClass(Dataset complete, int *classes, int numClasses) {
	int i;
	Dataset **classifiedDatasets = (Dataset **) calloc(numClasses, sizeof(Dataset *));
	for (i = 0; i < numClasses; i++) {
		classifiedDatasets[i] = new Dataset();
	}
	int classLabel = 0;
	int destIndex;
	while (complete.data.size() > 0) {
		classLabel = complete.data.back()->classLabel;
		destIndex = -1;
		for (i = 0; i < numClasses; i++) {
			if (classLabel == classes[i]) {
				destIndex = i;
				break;
			}
		}
		if (destIndex == -1) {
			cerr<<"Error in finding class index "<<classes[i]<<endl;
			return NULL;
		}
		classifiedDatasets[destIndex]->data.push_back(complete.data.back());
		complete.data.pop_back();
	}
	return classifiedDatasets;
}

void LinearClassifier::OVRest(Dataset **classifiedDatasets, int *classes, int numClasses, int algorithm) {
	Dataset *trainDataA, *trainDataB;
	int *indicesToMerge = new int[numClasses];
	result.OVRRes = new vector<float>[numClasses];
	int i;
	for (i = 0; i < numClasses; i++) {
		indicesToMerge[i] = i;
	}
	for (i = 0; i < numClasses; i++) {
		indicesToMerge[i] = -1;
		trainDataA = classifiedDatasets[i];
		trainDataB = mergeDatasets(classifiedDatasets, numClasses, indicesToMerge);
		switch (algorithm) {
			case 1:
				result.OVRRes[i] = SSPLFI(trainDataA, trainDataB);
				break;
			case 2:
				result.OVRRes[i] = BPLVI(trainDataA, trainDataB);
				break;
			case 3:
				result.OVRRes[i]= SSRVI(trainDataA, trainDataB);
				break;
			case 4:
				result.OVRRes[i]= BRLVI(trainDataA, trainDataB);
				break;
			case 5:
				result.OVRRes[i]= MSEUPI(trainDataA, trainDataB);
				break;
			case 6:
				result.OVRRes[i]= MSEULMS(trainDataA, trainDataB);
				break;
			default:
				printf("Wrong value for algorithm\n");
				break;
		}
		printf("a for %d iteration is: ", i);
		vector<float>::iterator aIt;
		for (aIt = result.OVRRes[i].begin(); aIt < result.OVRRes[i].end(); aIt++) {
			printf("%f ", *aIt);
		}
		printf("\n");
		indicesToMerge[i] = i;
	}

	return;
}

void LinearClassifier:: OVO(Dataset **classifiedDatasets, int *classes, int numClasses, int algorithm) {
	Dataset *trainDataA, *trainDataB;
	result.OVORes = new vector<float>*[numClasses];
	int i, j;
	for (i = 0; i < numClasses; i++) {
		result.OVORes[i] = new vector<float>[numClasses];
	}
	for (i = 0; i < numClasses; i++) {
		for (j = i + 1; j < numClasses; j++) {
			trainDataA = classifiedDatasets[i];
			trainDataB = classifiedDatasets[j];
			switch (algorithm) {
				case 1:
					result.OVORes[i][j] = SSPLFI(trainDataA, trainDataB);
					break;
				case 2:
					result.OVORes[i][j] = BPLVI(trainDataA, trainDataB);
					break;
				case 3:
					result.OVORes[i][j] = SSRVI(trainDataA, trainDataB);
					break;
				case 4:
					result.OVORes[i][j] = BRLVI(trainDataA, trainDataB);
					break;
				case 5:
					result.OVORes[i][j] = MSEUPI(trainDataA, trainDataB);
					break;
				case 6:
					result.OVORes[i][j] = MSEULMS(trainDataA, trainDataB);
					break;
				default:
					printf("Wrong value for algorithm\n");
					break;
			}
			printf("a for %d %d iteration is: ", i, j);
			vector<float>::iterator aIt;
			for (aIt = result.OVORes[i][j].begin(); aIt < result.OVORes[i][j].end(); aIt++) {
				printf("%f ", *aIt);
			}
			printf("\n");
		}
	}

	return;
}

int* LinearClassifier::findClasses(Dataset *dataset, int *numClasses) {
	vector<int> classesVec;
	vector<int>::iterator cvIt;
	vector<DataItem*>::iterator dataIt;

	for (dataIt = dataset->data.begin(); dataIt < dataset->data.end(); dataIt++) {
		if ((cvIt = find(classesVec.begin(), classesVec.end(), (*dataIt)->classLabel)) == classesVec.end()) {
			classesVec.push_back((*dataIt)->classLabel);
		}
	}
	*numClasses = classesVec.size();
	int *classes = NULL;
	if (*numClasses > 0) {
		classes = new int[*numClasses];
		result.classes = new int[*numClasses];
		int i;
		for (i = 0; i < *numClasses; i++) {
			result.classes[i] = classesVec[i];
			classes[i] = classesVec[i];
		}
	}
	return classes;
}

float LinearClassifier::learnModel(Dataset *trainData, int algorithm, int combination) {
	this->algorithm = algorithm;
	this->combination = combination;
	int numClasses;
	int *classes = findClasses(trainData, &numClasses);
	printf("Number of classes found to be: %d\n", numClasses);
	this->numClasses = numClasses;
	Dataset **classifiedDatasets = splitByClass(*trainData, classes, numClasses);
	switch (combination) {
		case 1:
			OVRest(classifiedDatasets, classes, numClasses, algorithm);
			break;
		case 2:
			OVO(classifiedDatasets, classes, numClasses, algorithm);
			break;
		case 3:
			printf("DDAG not implemented!!!\n");
			break;
		case 4:
			printf("BHDT not implemented!!!\n");
			break;
		default:
			printf("Invalid combination value\n");
	}
	ConfusionMatrix testCM;
	float errorRate = 0.0;
	errorRate = classifyDataset(*trainData, testCM);
	return errorRate;
}

int LinearClassifier::classifySampleOVRest(DataItem sample, int numClasses) {
	bool ambiguous = false, done = false;
	int noOfFeatures = sample.feature.size();
	int retVal = -1;
	float *a = new float[noOfFeatures + 1];
	float *y = new float[noOfFeatures + 1];
	copyFeatures(sample.feature, y, false);
	int i, j;
	float tempDotProduct = 0.0;
	for (i = 0; i < numClasses; i++) {
		for (j = 0; j < noOfFeatures + 1; j++) {
			a[j] = result.OVRRes[i][j];
		}
		tempDotProduct = dotProduct(a, y, noOfFeatures + 1);
		if (tempDotProduct > 0.0) {
			if (done) {
				ambiguous = true;
			}
			retVal = i;
			done = true;
		}
	}
	if (ambiguous) {
		return -1;
	}
	else {
		return retVal;
	}
}

int LinearClassifier::classifySampleOVO(DataItem sample, int numClasses) {
	int *resVals = new int[numClasses];
	int noOfFeatures = sample.feature.size();
	float *a = new float[noOfFeatures + 1];
	float *y = new float[noOfFeatures + 1];
	copyFeatures(sample.feature, y, false);
	int i, j, k;
	for (i = 0; i < numClasses; i++) {
		resVals[i] = 0;
	}
	float tempDotProduct = 0.0;
	for (i = 0; i < numClasses; i++) {
		for (j = i + 1; j < numClasses; j++) {
			for (k = 0; k < noOfFeatures + 1; k++) {
				a[k] = result.OVORes[i][j][k];
			}
			tempDotProduct = dotProduct(a, y, noOfFeatures + 1);
			if (tempDotProduct < 0.0) {
				resVals[j]++;
			}
			else {
				resVals[i]++;
			}
		}
	}
	int maxIndex = 0, retVal = -1;
	for (i = 1; i < numClasses; i++) {
		if (resVals[i] > resVals[maxIndex]) {
			maxIndex = i;
		}
	}
	if (resVals[maxIndex] > 0) {
		retVal = maxIndex;
	}
	else {
		printf("returning -1\n");
	}
	for (i = 0; i < numClasses; i++) {
		printf("%d ", resVals[i]);
	}
	printf(" %d\n", maxIndex);
	return retVal;
}

int LinearClassifier::classifySample(DataItem sample) {
	int classIndex = -1;
	switch (combination) {
		case 1:
			classIndex = classifySampleOVRest(sample, numClasses);
			break;
		case 2:
			classIndex = classifySampleOVO(sample, numClasses);
			break;
		case 3:
			break;
		case 4:
			break;
		default:
			printf("Invalid combination value\n");
	}
	if (classIndex == -1) {
//		printf("Ambiguous Class\n");
		return -1;
	}
	return result.classes[classIndex];
}

void LinearClassifier::initializeConfusionMatrix(ConfusionMatrix &cm) {
	cm.numClasses = numClasses;
	int i, j;
	cm.classes = new int[numClasses];
	cm.matrix = new long int*[numClasses];
	for (i = 0; i < numClasses; i++) {
		cm.classes[i] = result.classes[i];
		cm.matrix[i] = new long int[numClasses];
		for (j = 0; j < numClasses; j++) {
			cm.matrix[i][j] = 0;
		}
	}
}

int LinearClassifier::findClassIndex(int classLabel) {
	int i;
	for (i = 0; i < numClasses; i++) {
		if (result.classes[i] == classLabel) {
			return i;
		}
	}
	return -1;
}

float LinearClassifier::classifyDataset(Dataset testset, ConfusionMatrix &cm) {
	initializeConfusionMatrix(cm);
	vector<DataItem*>::iterator dataIt;
	float errorRate = 0.0;
	long int totalItems = 0, wrongItems = 0;
	int originalClass, newClass, originalClassIndex, newClassIndex;
	for (dataIt = testset.data.begin(); dataIt < testset.data.end(); dataIt++) {
		originalClass = (*dataIt)->classLabel;
		newClass = classifySample(*(*dataIt));
		if (originalClass != newClass) {
			wrongItems++;
		}
		totalItems++;
		originalClassIndex = findClassIndex(originalClass);
		newClassIndex = findClassIndex(newClass);
		if ((originalClassIndex == -1) || (newClassIndex == -1)) {
			printf("Confusion matrix indexing error %d %d\n", originalClass, newClass);
		}
		else {
			cm.matrix[originalClassIndex][newClassIndex]++;
		}
	}
	errorRate = ((1.0 * wrongItems) / totalItems);
	return errorRate;
}

void ConfusionMatrix::print() {
	int i,j;
	for (i = 0; i < numClasses; i++) {
		printf("%d ", classes[i]);
	}
	printf("\n");
	for (i = 0; i < numClasses; i++) {
		for (j = 0; j < numClasses; j++) {
			printf("%ld ", matrix[i][j]);
		}
		printf("\n");
	}
}

float findStdDev(float *errorRates, float avgErrorRate, int count) {
	float squaredSum = 0.0;
	float variance = 0.0;
	float stdDev = 0.0;
	int i;
	for (i = 0; i < count; i++) {
		squaredSum += (errorRates[i] * errorRates[i]);
	}
	variance = (squaredSum/count) - (avgErrorRate * avgErrorRate) ;
	printf("%f %f\n", avgErrorRate, squaredSum/count);
	stdDev = sqrt(variance);
	return stdDev;
}

float crossValidate(Dataset complete, int folds, float &stdDev, ConfusionMatrix &cm, int algorithm, int combination) {
	LinearClassifier *classifier = new LinearClassifier();
	Dataset **partialSets = splitDataset(complete, folds, time(NULL));
	Dataset *trainData, *testData;
	int i;
	float *errorRates = new float[folds];
	float avgErrorRate = 0.0, totalErrorRate = 0.0;
	int *indicesToMerge = new int[folds];
	for (i = 0; i < folds; i++) {
		indicesToMerge[i] = i;
	}
	for (i = 0; i < folds; i++) {
		indicesToMerge[i] = -1;
		testData = partialSets[i];
		trainData = mergeDatasets(partialSets, folds, indicesToMerge);
		classifier->learnModel(trainData, algorithm, combination);
		errorRates[i] = classifier->classifyDataset((*testData), cm);
		indicesToMerge[i] = i;
	}
	for (i = 0; i < folds; i++) {
		totalErrorRate += errorRates[i];
	}
	avgErrorRate = totalErrorRate / folds;
	cm.print();
	stdDev = findStdDev(errorRates, avgErrorRate, folds);
	return avgErrorRate;
}

int main() {
	Dataset *temp1 = new Dataset();
	ConfusionMatrix cm;
	float stdDev = -1.0;
	float avgError = 0.0;
//	Dataset *temp2 = new Dataset();
	temp1->readData("balance-scale.data");
//	temp1->writeData("temp.dat");
	avgError = crossValidate(*temp1, 5, stdDev, cm, 5, 1);
	printf("%f %f\n", avgError, stdDev);
//	temp2->readData("testinput1");
//	vector<float> a;
//	a = SSPLFI(temp1, temp2);
//	vector<float>::iterator aIt;
//	for (aIt = a.begin(); aIt < a.end(); aIt++) {
//		printf("%f ", *aIt);
//	}
//	printf("\n");
//	LinearClassifier classifier;
//	classifier.learnModel(temp1, 5, 1);
//	classifier.classifyDataset(*temp1, cm);
	delete temp1;
//	delete temp2;
	return 0;
}
