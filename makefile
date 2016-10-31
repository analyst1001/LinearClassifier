all: 
	g++ -Wall classifier.cpp ./newmat/*.o -lm -g
