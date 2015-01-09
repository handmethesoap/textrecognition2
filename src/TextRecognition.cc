#include "TextRecognition.hh"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

void TextRecognition::train(void){
  std::cout << "-----Training SVM-----" << std::endl;
  int sample_max = parameters.getIntParameter("num_samples");
  int text_samples = 0;
  int notext_samples = 0;
  uint image_n = 0;
  
  while( (image_n < trainImageNames.size()) && ((notext_samples < sample_max) || (text_samples < sample_max)) )
  {
    
    cv::Mat1f trainImage = cv::imread(  parameters.getStringParameter("recognition_train_path") + trainImageNames[image_n] , cv::IMREAD_GRAYSCALE );
    trainImage = trainImage/255;
    
    
    
    int w_size = 64;  
    
    for(int j = 0; j < trainImage.size().height - w_size; j+=(w_size/4)){
      for(int i = 0; i < trainImage.size().width - w_size; i+=(w_size/4)){
	
	  bool type = isText(trainImageNames[image_n], i, j, w_size, trainImageNames, trainTextBoxes);
	  
	  if( (type == 0) && (notext_samples < sample_max) ){
	    ++notext_samples;
	    datatype.push_back(type);
	    cv::Mat1f usubimage = trainImage(cv::Range(j,j+w_size), cv::Range(i,i+w_size)).clone();
	    cv::Mat1f subimage(32,32);
	    cv::resize(usubimage, subimage, subimage.size());
	    cv::Mat1f reducedfeatures;
	    computeFeatureRepresentation(subimage, reducedfeatures);
	    traindata.push_back(reducedfeatures.reshape(0,1));
	  }
	  else if( (type == 1) && (text_samples < sample_max) ){
	    ++text_samples;
	    datatype.push_back(type);
	    cv::Mat1f usubimage = trainImage(cv::Range(j,j+w_size), cv::Range(i,i+w_size)).clone();
	    cv::Mat1f subimage(32,32);
	    cv::resize(usubimage, subimage, subimage.size());
	    cv::Mat1f reducedfeatures;
	    computeFeatureRepresentation(subimage, reducedfeatures);
	    traindata.push_back(reducedfeatures.reshape(0,1));
	  }
      }
    }
    
    ++image_n;
    std::cout << "num text samples: " << text_samples << std::endl << "num notext samples: " << notext_samples << std::endl << std::endl;
  }
  datatype.convertTo(datatype, CV_32F);
  traindata.convertTo(traindata, CV_32F);
  
  saveTrainData();
  
  bool t = linearSVM.train(traindata, datatype.reshape(0,1), cv::Mat(), cv::Mat(), svmparams);
  
  if( t == 1 ){
    std::cout << "linear SVM successfully trained" << std::endl;
  }
  else{
    std::cout << "linear SVM training failed" << std::endl;
    exit(0);
  }
}

void TextRecognition::test(std::string testFile){
  
  cv::Mat1f testImage = cv::imread(  parameters.getStringParameter("test_path") + testFile , cv::IMREAD_GRAYSCALE );
  std::cout << testImage.size().width << ", " << testImage.size().height << std::endl;
  testImage = testImage/255;
  //normalise(testImage);
  
  int min_win = parameters.getIntParameter("min_window_size");  
  int max_win = parameters.getIntParameter("max_window_size");  
  int win_inc = parameters.getIntParameter("window_increment");  
  
  cv::Mat1f textImage = cv::Mat1f::ones(testImage.size().height, testImage.size().width);
  textImage = textImage*(-100);
  

  for(int w_size = min_win; w_size <= max_win; w_size += win_inc){
    
    std::cout << "testing with subimage size of " << w_size << std::endl;
    
    cv::Mat1f imageMask = cv::Mat1f::ones(w_size, w_size);
  
    for(int j = 0; j <= testImage.size().height - w_size; j+=w_size){
      
      if(testImage.size().height < w_size){
	break;
      }
      
      for(int i = 0; i <= testImage.size().width - w_size; i+=w_size){
	  
	  if(testImage.size().width < w_size){
	    break;
	  }
	  
	  //std::cout << testImage.size().width << ", " << testImage.size().height << ", " << i << ", " << j << std::endl;
	  cv::Mat1f usubimage = testImage(cv::Range(j,j+w_size), cv::Range(i,i+w_size)).clone();
	  //std::cout << "here1.1" << std::endl;
	  cv::Mat1f subimage(32,32);
	  cv::resize(usubimage, subimage, subimage.size());
	  
	  cv::Mat1f reducedfeatures;
	  cv::Mat reducedfeatures2;
	  
	  computeFeatureRepresentation(subimage, reducedfeatures);
	  
	  reducedfeatures.convertTo(reducedfeatures2, CV_32F);
	  
	  float predictResult = linearSVM.predict(reducedfeatures2.reshape(0,1), true);
	  cv::max(imageMask*predictResult, textImage(cv::Range(j,j+w_size), cv::Range(i,i+w_size)), textImage(cv::Range(j,j+w_size), cv::Range(i,i+w_size)));
		  
	  //adjust iterators in order to process edges of image
	  if( ((testImage.size().width - 2*w_size) < i) && (i < (testImage.size().width - w_size)) ){
	    i = testImage.size().width - (2*w_size); 
	  }
      }
      
      if( ((testImage.size().height - 2*w_size) < j) && (j < (testImage.size().height - w_size)) ){
	j = testImage.size().height - (2*w_size); 
      }
//       std::cout << "Exiting" << std::endl;
//       exit(0);
    }
    
  }
  storeScores(textImage, testFile);

//   std::cout << testFile << std::endl;
//   normalise(textImage); 
//   cv::imwrite("sample_image.jpg", textImage*255);
//   cv::namedWindow( "Display window", cv::WINDOW_NORMAL );// Create a window for display.
//   cv::imshow( "Display window", textImage);
//   cv::waitKey(0);
//   cv::destroyWindow("Display window");
}

void TextRecognition::readLocationData(std::string fileName, std::vector<std::string> & imageNames, std::vector<std::vector<cv::Rect*>*> & textBoxes){
  
  //open text file containing image file data
  std::ifstream infile(fileName.c_str());
  CHECK_MSG(infile.good(),"Error reading '" << fileName << "'.  Please check file exists and is named correctly");
  int numimages = 0;

  //Read image file names
  while (!infile.eof()){
    std::string line, param;
    std::string file;
    std::stringstream tt;
    
    getline(infile,line);
    tt<<line;
    
    while( tt>>param ){

      //get image file name
      if(param == "<image>" ){
	getline(infile,line);
	tt<<line;
	tt>>param;
	file = param.substr(param.find('>')+1);
	file = file.erase(file.find('<'));
	
	imageNames.push_back(file);
	textBoxes.push_back( new std::vector<cv::Rect*> );
		
	++numimages;
	
	int x,y,w,h;
	std::string value;

	//get textbox bounds of image
	while( param != "</taggedRectangles>" ){
	  
	  if(param == "<taggedRectangle" ){
	    
	    while( param[0] != 'x' ){
	      tt>>param;
	    }
	    value = param.substr(param.find('"')+1);
	    value = value.erase(value.find('"'));
	    x =  int(atof(value.c_str()));
	    
	    while( param[0] != 'y' ){
	      tt>>param;
	    }
	    value = param.substr(param.find('"')+1);
	    value = value.erase(value.find('"'));
	    y =  int(atof(value.c_str()));
	    
	    while( param[0] != 'w' ){
	      tt>>param;
	    }
	    value = param.substr(param.find('"')+1);
	    value = value.erase(value.find('"'));
	    w =  int(atof(value.c_str()));
	    
	    while( param[0] != 'h' ){
	      tt>>param;
	    }
	    value = param.substr(param.find('"')+1);
	    value = value.erase(value.find('"'));
	    h =  int(atof(value.c_str()));

	    (*textBoxes[numimages - 1]).push_back(new cv::Rect(x,y,w,h));

	  }
	  getline(infile,line);
	  tt<<line;
	  tt>>param;
	}
      }
    }
  }
  
}

bool TextRecognition::isText(std::string imageName, int x, int y, int boxSize, std::vector<std::string> & imageNames, std::vector<std::vector<cv::Rect*>*> & textBoxes){
  
  auto it = find(imageNames.begin(), imageNames.end(), imageName);
  
  if(it == imageNames.end()){
    std::cout << "File name " << imageName << " not found in file list" << std::endl;
    std::exit(0);
  }
  
  auto pos = std::distance(imageNames.begin(), it);
  
  int istextsum = 0;
  cv::Rect box(x,y,boxSize, boxSize);
  
  //find area intersection between location and textboxes
  for( auto it2 =  textBoxes[pos]->begin(); it2 != textBoxes[pos]->end(); ++it2 ){
    cv::Rect temprect = box & **it2;
    istextsum += temprect.area();
  }
  
  //return true if over 80% of the block is text
  if(istextsum > 0.8*box.area()){
    return 1;
  }
  else{
    return 0;
  }
}

void TextRecognition::printText(std::string imageName){
  cv::Mat image = cv::imread(  parameters.getStringParameter("test_path") + imageName , cv::IMREAD_GRAYSCALE );
  
  for(int j = 0; j < image.size().height; ++j){
    for(int i = 0; i < image.size().width; ++i){
      if(isText(imageName, i, j, 1, testImageNames, testTextBoxes) != 1){
	image.at<uchar>(j,i) = 255;
      }
    }
  }
	
  cv::namedWindow( "Display window", cv::WINDOW_NORMAL );// Create a window for display.
  cv::imshow( "Display window", image );  
  cv::waitKey(0);   
}

void TextRecognition::computeFeatureRepresentation(cv::Mat1f & subimage, cv::Mat1f & reducedfeatures ){
  
  // TODO: make this float, then dot product will not overflow //DONE
  // maybe cv::Mat1f
  cv::Mat1f features;
  
  //extract 8*8 subsubimages

    for(int k = 0; k < subimage.size().height - 7; ++k){
      for( int l = 0; l < subimage.size().width - 7; ++l){
	
	//extract subsubimages
	cv::Mat1f subsubimage = subimage(cv::Range(k,k+8), cv::Range(l,l+8)).clone();
	
	//normalise and zca whiten
	// TODO: normalize correctly, i.e. zca whitening over all (or many random) train patches
	// save w and u and use that w and u to transform now this particular subimage
	normalise(subsubimage);
	//zcaWhiten( subsubimage );
	
	
	//compute dot product with dictionary
	// possible speedup: use gemm (i.e. matrix multiplication)
	features.push_back((subsubimage.reshape(0, 64)));
	
	// TODO: max(..., alpha)  
      }
    }
  
  //std::cout << features.size() << std::endl;
  //std::cout << "features size = " << features.size() << std::endl;
  cv:: Mat1f features2 = features.reshape(0,625);
  //std::cout << features.row(1) << std::endl;
  //std::cout << "features size = " << features2.size() << std::endl;
  cv::Mat1f features3;
  //std::cout << std::endl << std::endl << "starting" << std::endl;
  for( int m = 0; m < parameters.getIntParameter("dictionary_length"); ++m){
    cv::Mat1f temp1 = (dict.centers.row(m).reshape(0,64));
    cv::Mat1f temp = features2*temp1;
    features3.push_back(temp);
    //std::cout << temp << std::endl;
  }
  //reshape so each row is the result the dot product of each 8*8 subsubimage with a single dictionary element 
  cv::Mat1f features4 = features3.reshape(0,parameters.getIntParameter("dictionary_length"));
//   std::cout << "features4 size = " << features4.size() << std::endl;
//   //std::cout << features4 << std::endl;
//   std::cout << (dict.centers.row(0)).reshape(0,8) << std::endl;
//   std::cout << "centers size = " << dict.centers.size() << std::endl;
  // TODO: test shape of reducedfeatures //DONE
  reduceFeatures(features4, reducedfeatures);
}

void TextRecognition::reduceFeatures( cv::Mat1f & featurerepresentation, cv::Mat1f & reducedfeatures ){
  
  for( int i = 0; i < parameters.getIntParameter("dictionary_length"); ++i){
    
    cv::Mat1f dictmatrix = featurerepresentation.row(i);
    //std::cout << "dictmatrix = " << dictmatrix.size() << std::endl;
    cv::Mat1f dictmatrix2 = dictmatrix.reshape(0,25);
    //std::cout << dictmatrix2 << std::endl;
    
    for(int k = 0; k < 3; ++k){
      for(int l = 0; l < 3; ++l){
	
	reducedfeatures.push_back( (cv::sum(dictmatrix2(cv::Range(k*8, k*8+9), cv::Range(l*8,l*8+9)))[0])/9 );
	
      }
    }
  }
  //std::cout << reducedfeatures << std::endl << std::endl;
}

void TextRecognition::normalise( cv::Mat1f & matrix ){
  
  //brightness normalise
  cv::Scalar matmean;
  cv::Scalar stddev;
  cv::meanStdDev(matrix, matmean, stddev);
  cv::subtract(matrix, matmean[0], matrix);
  //std::cout << "here1.1" << std::endl;
  // FIXME
  //contrast normalise and remap to 0-1 range
  cv::meanStdDev(matrix, matmean, stddev);
  if( stddev[0] != 0.0 ){
    matrix = matrix*(0.5/(1.0*stddev[0]));
  }
  //std::cout << "here1.2" << std::endl;
  cv::add(matrix, 0.5, matrix);
  //std::cout << "here1.3" << std::endl;
  for(int i=0; i<matrix.size().width; ++i){
    for(int j=0; j<matrix.size().height; ++j){
      if( matrix.at<float>(j,i) < 0.0 )
	matrix.at<float>(j,i) = 0.0;
      else if ( matrix.at<float>(j,i) > 1)
	matrix.at<float>(j,i) = 1.0;      
    }
  }
  //std::cout << "here1.4" << std::endl;
}

void TextRecognition::zcaWhiten( cv::Mat1f & matrix){
  //apply ZCA whitening
  cv::Mat sigma, w_, u, vt;
  cv::Mat subimagetranspose, utranspose;
  cv::transpose(matrix, subimagetranspose);
  sigma = matrix*subimagetranspose/8;
  cv::SVD::compute(sigma, w_, u, vt);
  cv::transpose(u, utranspose);
  w_ = w_+0.1;
  cv::sqrt(w_,w_);
  w_ = 1./w_;
  cv::Mat D(8,8,CV_32F); 
  D = D.diag(w_);
  matrix = u*D*utranspose*matrix;
}	

void TextRecognition::saveTrainData(void){
  std::ofstream outputfile;
  outputfile.open( parameters.getStringParameter("data_file") );
  
  outputfile << traindata << std::endl;
  outputfile << "data type" << std::endl;
  outputfile << datatype << std::endl;
  
  outputfile.close();
}

void TextRecognition::loadTrainData( void ){

  std::cout << "loading training data" << std::endl;
  
  readTrainData();
  traindata.convertTo(traindata, CV_32F);
  datatype.convertTo(datatype, CV_32F);
  
  std::cout << "loading complete" << std::endl;
  
  std::cout << "linear SVM training underway" << std::endl;
  
  bool output = linearSVM.train(traindata, datatype, cv::Mat(), cv::Mat(), svmparams);
  
  if( output == 1 ){
    std::cout << "linear SVM successfully trained" << std::endl;
  }
  else{
    std::cout << "linear SVM training failed" << std::endl;
  }
  
}

void TextRecognition::readTrainData(void){
  std::ifstream infile;
  infile.open( parameters.getStringParameter("data_file") );
  cv::Mat *outputmatrix = &traindata;
  int num;
  
  while (!infile.eof()){
    std::string line, param;
    std::string file;
    std::stringstream tt;
    cv::Mat row;
    getline(infile,line);
    tt<<line;
    while( tt>>param ){
      if(param == "data"){
	outputmatrix = &datatype;
	break;
      }
      
      if( param[0] == '[' ){
	std::istringstream(param.substr(param.find('[')+1)) >> num;
	row.push_back(num);
      }
      else if( param.find(']') != std::string::npos ){
	std::istringstream(param.erase(param.find(']'))) >> num;
	row.push_back(num);
	break;
      }
      else{
	std::istringstream(param) >> num;
	row.push_back(num);
      }
    }
    if((param != "data") && (param != "")){
      outputmatrix->push_back(row.reshape(0,1));
    }
  }    
  
  infile.close();
}

void TextRecognition::storeScores(cv::Mat1f & image, std::string imageName){
  //std::cout << image << std::endl;
  for(int j = 0; j < image.size().height; ++j){
    for(int i = 0; i < image.size().width; ++i){
      if(isText(imageName, i, j, 1, testImageNames, testTextBoxes)){
	textScores.push_back(image.at<float>(j,i));
      }
      else{
	nonTextScores.push_back(image.at<float>(j,i));
      }
    }
  }
}

void TextRecognition::saveScores(void){
  std::ofstream outputfile;
  outputfile.open( parameters.getStringParameter("scores_file") );
  
  int i = 0;
  
  for(i = 0; i < (textScores.size().height - 1000); i = i+1000){
    outputfile << textScores(cv::Range(i,i+1000), cv::Range::all()) << std::endl;
  }
  outputfile << textScores(cv::Range(i,textScores.size().height), cv::Range::all()) << std::endl;
  
  outputfile << "nonTextScores" << std::endl;
  
  for(i = 0; i < (nonTextScores.size().height - 1000); i = i+1000){
    outputfile << nonTextScores(cv::Range(i,i+1000), cv::Range::all()) << std::endl;
  }
  outputfile << nonTextScores(cv::Range(i,nonTextScores.size().height), cv::Range::all()) << std::endl;
  
  outputfile.close();
}

void TextRecognition::readScores(void){
  std::ifstream infile;
  infile.open( parameters.getStringParameter("scores_file") );
  cv::Mat *outputmatrix = &textScores;
  float num;
  
  std::cout << "loading scores file" << std::endl;
  
  while (!infile.eof()){
    std::string line, param;
    std::string file;
    std::stringstream tt;
    cv::Mat row;
    getline(infile,line);
    tt<<line;
    while( tt>>param ){
      if(param == "nonTextScores"){
	tt>>param;
	tt>>param;
	std::cout << param << std::endl;
	std::cout << "loading non text scores" << std::endl;
	outputmatrix = &nonTextScores;
	break;
      }
      
      if( param[0] == '[' ){
	std::istringstream(param.substr(param.find('[')+1)) >> num;
	row.push_back(num);
      }
      else if( param.find(']') != std::string::npos ){
	std::istringstream(param.erase(param.find(']'))) >> num;
	row.push_back(num);
	break;
      }
      else{
	std::istringstream(param) >> num;
	row.push_back(num);
      }
    }
    if((param != "nonTextScores") && (param != "")){
      outputmatrix->push_back(row);
    }
  }    
  
  infile.close();
  
  std::cout << textScores.size() << ", " << nonTextScores.size() << std::endl;
}

void TextRecognition:: testAll(void){
  
  for( uint i = 0; i < testImageNames.size() ; ++i){
    std::cout << "-----Testing image " << i << " of " << testImageNames.size() << "-----" << std::endl;
    test(testImageNames[i]);
  }
  generatePRData();
  //saveScores();
  generatePlotData();
  
}

void TextRecognition:: testOne(uint i){
  
    std::cout << "-----Testing image " << i << " of " << testImageNames.size() << "-----" << std::endl;
    test(testImageNames[i]);
    generatePRData();
    //saveScores();
    generatePlotData();
  
}

void TextRecognition:: testN(uint n){
  
  for( uint i = 0; i < n ; ++i){
    std::cout << "-----Testing image " << i << " of " << testImageNames.size() << "-----" << std::endl;
    test(testImageNames[i]);
  }
  generatePRData();
  //saveScores();
  generatePlotData();
  
}
void TextRecognition::generatePRData(void){
  
  std::cout << "sorting scores" << std::endl;
  
  cv::Mat sortedTextScores;
  cv::Mat sortedNonTextScores;
  cv::Mat temp;
  
  cv::sort(textScores, sortedTextScores, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
  cv::sort(nonTextScores, sortedNonTextScores, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
  cv::vconcat(sortedTextScores, sortedNonTextScores, temp);
  
  double min1, max1;
  cv::minMaxLoc(temp, &min1, &max1);
  
  cv::subtract(sortedTextScores, min1, sortedTextScores);
  sortedTextScores = sortedTextScores*(1000.0/(max1 - min1));
  
  cv::subtract(sortedNonTextScores, min1, sortedNonTextScores);
  sortedNonTextScores = sortedNonTextScores*(1000.0/(max1 - min1));
  
  int tempText = 0;
  int tempNonText = 0;
  
  for(int i = 1000; i > 0; --i){
   

    while(sortedTextScores.at<float>(tempText,0) >= i){
      ++tempText;
      //std::cout << tempText << std::endl;
    }
   while(sortedNonTextScores.at<float>(tempNonText,0) >= i){
      ++tempNonText;
    }
    //std::cout << tempText << std::endl;
    float p = float(tempText)/float(tempText + tempNonText);
    float r = float(tempText)/(float(sortedTextScores.size().height));
    
    precision.push_back(p);
    recall.push_back(r);
  }
}

void TextRecognition::generatePlotData(void){
  
  std::ofstream outputfile;
  outputfile.open( parameters.getStringParameter("graph_file") );
  
  for(int i = 0; i < precision.size().height; ++i){
    outputfile << recall.at<float>(i,0) << " " << precision.at<float>(i,0) << std::endl;
  }
  outputfile.close();
}