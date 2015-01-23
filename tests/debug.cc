#include <iostream>
#include "FileReader.hh"
#include "Dictionary.hh"
#include "TextRecognition.hh"

int main( int argc, char** argv )
{
  
  //Initialise file reader
  FileReader parameters;
  
  //Register parameters to be read from file
  parameters.registerStringParameter("recognition_train_path");
  parameters.registerStringParameter("test_path");
  parameters.registerStringParameter("train_file");	
  parameters.registerStringParameter("test_file");	
  parameters.registerStringParameter("data_file");
  parameters.registerStringParameter("scores_file");
  parameters.registerStringParameter("graph_file");
  
  parameters.registerIntParameter("kmeans_attempts");
  parameters.registerIntParameter("kmeans_iter");
  parameters.registerIntParameter("subimage_size");
  parameters.registerRealParameter("kmeans_eps");
  
  parameters.registerIntParameter("debug_flag");
  
  parameters.registerIntParameter("min_window_size");  
  parameters.registerIntParameter("max_window_size");  
  parameters.registerIntParameter("window_increment");  
  
  parameters.registerIntParameter("num_samples");  
  
  parameters.registerIntParameter("train_bit");  
  parameters.registerIntParameter("generate_bit"); 
  
  parameters.registerIntParameter("dictionary_length");
  parameters.registerStringParameter("dictionary_save_path");
  parameters.registerStringParameter("dictionary_path");
  parameters.registerIntParameter("dictionary_source_images");  
  
  
  //Read Parameters from file
  std::string parameterfile = "debug.par";
  CHECK_MSG( parameters.readFile(parameterfile), "Could not read config file");
  
//   //using namespace cv;
//   
//     cv::FileStorage fs(parameters.getStringParameter("dictionary_save_path") + "dictdata.yml", cv::FileStorage::WRITE);
//     std::cout << "here" << std::endl;
//     fs << "frameCount" << 5;
//   if(fs.isOpened() != 1){
//     std::cout << "Cannot open " << parameters.getStringParameter("dictionary_save_path") + "dictdata.yml" << " exiting.." << std::endl;
//     exit(0);
//   }
//   cv::Mat1f one;
//   cv::Mat1f two;
//   cv::Mat1f three;
//   one.push_back(4);
//   one.push_back(7);
//   two.push_back(5);
//   three.push_back(6);
//     std::cout << one << std::endl;
//   fs << "Dictionary" << one;
//   fs << "matmean" << two;
//   fs << "zca" << three;
//   fs.release();
//   std::cout << "here" << std::endl;
//   
//   
//   
//   
//     cv::FileStorage fs2(parameters.getStringParameter("dictionary_save_path") + "dictdata.yml", cv::FileStorage::READ);
//   if(fs2.isOpened() != 1){
//     std::cout << "Cannot open " << parameters.getStringParameter("dictionary_save_path") + "dictdata.yml" << " exiting.." << std::endl;
//     exit(0);
//   }
//       cv::Mat1f tone;
//   cv::Mat1f ttwo;
//   cv::Mat1f tthree;
//   fs2["Dictionary"] >> tone;
//   fs2 ["matmean"] >> ttwo;
//   fs2 ["zca"] >> tthree;
//   fs2.release();
//   
//   std::cout << tone << std::endl;
//   std::cout << ttwo << std::endl;
//   std::cout << tthree << std::endl;
//   exit(0);
  
  
  
  
  
  //Initialise objects
  Dictionary dict(parameters);
  if(parameters.getIntParameter("generate_bit") == 1){
    dict.generate();
  }
  else{
    dict.read();
  }
  
  
//   Train text recogniser
  TextRecognition recogniseText(parameters, dict);
  if(parameters.getIntParameter("train_bit") == 1){
    recogniseText.train();
  }
  else{
    recogniseText.loadTrainData();
  }
  recogniseText.testN(1);
  
  exit(0);
  
//   parameters.setParameter("data_file", "train_data1.txt");
//   parameters.setParameter("scores_file", "tempscores1.txt");
//   parameters.setParameter("graph_file", "graphdata1.dat");
//   parameters.setParameter("dictionary_length", 1);
//   parameters.setParameter("dictionary_save_path", "dicttest1/");
//   
//   //Initialise objects
//   Dictionary dict1(parameters);
//   if(parameters.getIntParameter("generate_bit") == 1){
//     dict1.generate();
//   }
//   else{
//     dict1.read();
//   }
//   
//   TextRecognition recogniseText1(parameters, dict1);
//   if(parameters.getIntParameter("train_bit") == 1){
//     recogniseText1.train();
//   }
//   else{
//     recogniseText1.loadTrainData();
//   }
//   recogniseText1.testN(5);
//   
//   parameters.setParameter("data_file", "train_data10.txt");
//   parameters.setParameter("scores_file", "tempscores10.txt");
//   parameters.setParameter("graph_file", "graphdata10.dat");
//   parameters.setParameter("dictionary_length", 10);
//   parameters.setParameter("dictionary_save_path", "dicttest10/");
//   
//   //Initialise objects
//   Dictionary dict10(parameters);
//   if(parameters.getIntParameter("generate_bit") == 1){
//     dict10.generate();
//   }
//   else{
//     dict10.read();
//   }
//   
//   TextRecognition recogniseText10(parameters, dict10);
//   if(parameters.getIntParameter("train_bit") == 1){
//     recogniseText10.train();
//   }
//   else{
//     recogniseText10.loadTrainData();
//   }
//   recogniseText10.testN(5);
//   
//   system("gnuplot gp.p");
//   
//   return 0;
}