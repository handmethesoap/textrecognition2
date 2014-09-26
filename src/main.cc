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
  parameters.registerStringParameter("train_file");	
  parameters.registerStringParameter("test_file");	
  parameters.registerStringParameter("data_file");
  
  parameters.registerIntParameter("kmeans_attempts");
  parameters.registerIntParameter("kmeans_iter");
  parameters.registerIntParameter("subimage_size");
  parameters.registerRealParameter("kmeans_eps");
  
  parameters.registerIntParameter("min_window_size");  
  parameters.registerIntParameter("max_window_size");  
  parameters.registerIntParameter("window_increment");  
  
  parameters.registerIntParameter("num_samples");  
  
  parameters.registerIntParameter("train_bit");  
  
  parameters.registerIntParameter("dictionary_length");
  parameters.registerStringParameter("dictionary_save_path");
  
  //Read Parameters from file
  std::string parameterfile = "parameters.par";
  CHECK_MSG( parameters.readFile(parameterfile), "Could not read config file");
  
  //Initialise objects
  Dictionary dict(parameters);
  
  dict.read();
  TextRecognition recogniseText(parameters, dict);
  
  //Train text recogniser
  if(parameters.getIntParameter("train_bit") == 1){
    recogniseText.train();
  }
  else{
    recogniseText.loadTrainData();
  }
  
  recogniseText.test();
  
  
  
  return 0;
}