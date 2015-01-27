#ifndef TEXTRECOGNITION_HH
#define TEXTRECOGNITION_HH

#include "FileReader.hh"
#include "Dictionary.hh"
#include <vector>
#include <opencv2/ml/ml.hpp>

class TextRecognition {
  
public:
  TextRecognition(const FileReader & conf, const Dictionary & d) : parameters(conf), dict(d){
    
    readLocationData(parameters.getStringParameter("recognition_train_path") + "locations.xml", trainImageNames, trainTextBoxes);
    readLocationData(parameters.getStringParameter("test_path") + "locations2.xml", testImageNames, testTextBoxes);
    
    svmparams.svm_type    = CvSVM::C_SVC;
    svmparams.kernel_type = CvSVM::LINEAR;
    svmparams.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
  
  }

  ~TextRecognition(void){
    
    for(std::vector<cv::Rect*>* it2 : trainTextBoxes){
      for(cv::Rect* it3 : *it2){
	delete(it3);
      }
      delete(it2);
    }
  }
  
  void train(void);
  void loadTrainData( void );
  void test(std::string testFile);
  void testAll(void);
  void testOne(uint i);
  void testN(uint n);
  void readScores(void);
  
  void computeFeatureRepresentation(cv::Mat1f & subimage, cv::Mat1f & reducedfeatures );
  void printText(std::string image);
  
  void generatePRData(void);
  void generatePlotData(void);
private:

  void readLocationData(std::string fileName, std::vector<std::string> & imageNames, std::vector<std::vector<cv::Rect*>*> & textBoxes);
  bool isText(std::string imageName, int x, int y, int boxSize, std::vector<std::string> & imageNames, std::vector<std::vector<cv::Rect*>*> & textBoxes);

  void reduceFeatures( cv::Mat1f & featurerepresentation, cv::Mat1f & reducedfeatures );
  void printm(cv::Mat mat, int prec);
  
  void saveTrainData(void);
  
  void storeScores(cv::Mat1f & image, std::string imageName);
  void saveScores(void);
  
  cv::Mat1f precision;
  cv::Mat1f recall;
  
  const FileReader & parameters;
  const Dictionary & dict;
  
  cv::SVM linearSVM;
  cv::SVMParams svmparams;
  
  std::vector<std::string> trainImageNames;
  std::vector<std::vector<cv::Rect*>*> trainTextBoxes;
  
  std::vector<std::string> testImageNames;
  std::vector<std::vector<cv::Rect*>*> testTextBoxes;
  
  cv::Mat textScores;
  cv::Mat nonTextScores;
  
  cv::Mat traindata;
  cv::Mat datatype;
  
};



#endif