#ifndef DICTIONARY_HH
#define DICTIONARY_HH

#include "FileReader.hh"
#include "Debug.hh"
#include "Array.hh"
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

class Dictionary
{
public:
  Dictionary( const FileReader & conf): parameters(conf)
  {
    CHECK_MSG( parameters.getIntParameter("dictionary_length") > 0, "The provided dictionary length must be larger than zero, please adjust this in your parameter file" );
    CHECK_MSG( parameters.getIntParameter("kmeans_attempts") > 0, "The provided number of kmeans attempts must be larger than zero, please adjust this in your parameter file" );
    CHECK_MSG( parameters.getIntParameter("kmeans_iter") > 0, "The provided number of kmeans iterations must be larger than zero, please adjust this in your parameter file" );
    CHECK_MSG( parameters.getRealParameter("kmeans_eps") > 0.0, "The provided kmeans epsilon value must be larger than zero, please adjust this in your parameter file" );    
  }
  
  
  void generate();
  void read();
  void zcawhitener( cv::Mat & samplesin ) const;
  cv::Mat1f centers;
  cv::Mat w, u, zca;
  cv::Scalar matmean;
  
private:
  
  void printfiles();
  void getfilenames( std::vector<std::string>& filenames );
  void getsubimages( cv::Mat& samples, std::string filename );
  const FileReader & parameters;
  void zcawhiten( cv::Mat & samples );
  
  
  
};














#endif //DICTIONARY_HH