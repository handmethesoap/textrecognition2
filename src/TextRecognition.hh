#ifndef TEXTRECOGNITION_HH
#define TEXTRECOGNITION_HH

#include "FileReader.hh"
#include "Dictionary.hh"
#include <vector>
#include <opencv2/ml/ml.hpp>

class TextRecognition {
  
public:
  TextRecognition(const FileReader & conf, const Dictionary & d) : parameters(conf), dict(d){
    
    readLocationData();
    
    svmparams.svm_type    = CvSVM::C_SVC;
    svmparams.kernel_type = CvSVM::LINEAR;
    svmparams.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
  
  }

  ~TextRecognition(void){
    
    for(std::vector<cv::Rect*>* it2 : textBoxes){
      for(cv::Rect* it3 : *it2){
	delete(it3);
      }
      delete(it2);
    }
  }
  
  void train(void);
  void loadTrainData( void );
  void test(void);

  
  void normalise( cv::Mat1f & matrix );
  void zcaWhiten( cv::Mat1f & matrix);
  void computeFeatureRepresentation(cv::Mat1f & subimage, cv::Mat1f & reducedfeatures );
private:

  void readLocationData(void);
  bool isText(std::string imageName, int x, int y, int boxSize);
  void printText(std::string imageName);
  void reduceFeatures( cv::Mat1f & featurerepresentation, cv::Mat1f & reducedfeatures );
  
  
  void saveTrainData(void);
  void readTrainData(void);
  
  
  const FileReader & parameters;
  const Dictionary & dict;
  
  cv::SVM linearSVM;
  cv::SVMParams svmparams;
  
  std::vector<std::string> imageNames;
  std::vector<std::vector<cv::Rect*>*> textBoxes;
  cv::Mat traindata;
  cv::Mat datatype;
  
};



#endif