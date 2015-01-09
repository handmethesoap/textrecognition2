#include "Dictionary.hh"
#include <fstream>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>

void Dictionary::generate(){
  
  std::cout << "-----Generating Dictionary-----" << std::endl;
  
  cv::Mat samples;
  cv::Mat labels;
  cv::TermCriteria criteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, parameters.getIntParameter("kmeans_iter"), parameters.getRealParameter("kmeans_eps"));
  std::vector<std::string> filenames;
  
  //open text file containing image file data
  std::string name = parameters.getStringParameter("dictionary_path") + "char.xml";
  std::ifstream infile(name.c_str());
  CHECK_MSG(infile.good(),"Error reading '" << name << "'.  Please check file exists and is named correctly");
  
  //Read image file names
  getfilenames( filenames );

  //get subimages from each image to be analysed
  for( std::vector<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it ){   
    getsubimages( samples, *it );
  }

  zcawhiten(samples);
     
  std::cout << "Performing kmeans calculation" << std::endl;
  cv::kmeans(samples, parameters.getIntParameter("dictionary_length"), labels, criteria, parameters.getIntParameter("kmeans_attempts"), cv::KMEANS_PP_CENTERS, centers);

  //save dictionary images to file
  printfiles();

}

void Dictionary::printfiles(){
  
  //calculate parameters to scale images to the 0-255 range
  double min;
  double max;
  cv::Point minLoc;
  cv::Point maxLoc;
  cv::minMaxLoc(centers, &min, &max, &minLoc, &maxLoc);
  cv::subtract(centers, min, centers);
  
  for( int j = 0; j < parameters.getIntParameter("dictionary_length"); ++ j){
    cv::Mat image;
    std::stringstream tt;
    std::string name;

    for( int i = 0; i <= 7 ; ++i ){
      image.push_back(centers(cv::Range(j, j+1), cv::Range(i*8,(i+1)*8)));
    }
    
    image = image*(255.0/(max-min));  
    
    tt << parameters.getStringParameter("dictionary_save_path") << j << ".jpg";
    tt >> name; 
    
    imwrite(name, image);
  }  
  
  std::stringstream tt;
  std::string name;
  
  tt << parameters.getStringParameter("dictionary_save_path") << "zcadata.txt";
  tt >> name;
  
  std::ofstream outputfile;
  outputfile.open( name );
  
  outputfile << u << std::endl;
  outputfile << "w" << std::endl;
  outputfile << w << std::endl;
  
  outputfile.close();
  
}

void Dictionary::read(){
  std::cout << "-----Loading Dictionary-----" << std::endl;
  //int sz = 64;
  
  for(int i = 0; i < parameters.getIntParameter("dictionary_length"); ++ i){
    cv::Mat1f image;
    std::stringstream tt;
    std::string name;
    std::cout << "reading dictionary element " << i+1 << " of " << parameters.getIntParameter("dictionary_length") << std::endl;
    tt << parameters.getStringParameter("dictionary_save_path") << i << ".jpg";
    tt >> name;
    
    image = cv::imread(  name , cv::IMREAD_GRAYSCALE );
    if(! image.data ){
      std::cout << "could not read image " << name << std::endl;
    }
    image = image/255;
    centers.push_back((image.reshape(0, 1)));
  }
  
  std::ifstream infile;
  std::stringstream ss;
  std::string name;
  ss << parameters.getStringParameter("dictionary_save_path") << "zcadata.txt";
  ss >> name;
  
//   infile.open( name );
//   cv::Mat *outputmatrix = &u;
//   float num;
//   
//   while (!infile.eof()){
//     std::string line, param;
//     std::string file;
//     std::stringstream tt;
//     cv::Mat row;
//     getline(infile,line);
//     tt<<line;
//     while( tt>>param ){
//       if(param == "w"){
// 	outputmatrix = &w;
// 	break;
//       }
//       
//       if( param[0] == '[' ){
// 	std::istringstream(param.substr(param.find('[')+1)) >> num;
// 	row.push_back(num);
//       }
//       else if( param.find(']') != std::string::npos ){
// 	std::istringstream(param.erase(param.find(']'))) >> num;
// 	row.push_back(num);
// 	break;
//       }
//       else{
// 	std::istringstream(param) >> num;
// 	row.push_back(num);
//       }
//     }
//     if((param != "w") && (param != "")){
//       outputmatrix->push_back(row.reshape(0,1));
//     }
//   }    
//   
//   infile.close();
}
  
void Dictionary::zcawhiten( cv::Mat & samples ){
  
  //apply ZCA whitening
  cv::Mat sigma, vt;
  cv::Mat samplestranspose, utranspose;
  cv::transpose(samples, samplestranspose);
  samples.convertTo(samples, CV_32FC1);
  samplestranspose.convertTo(samplestranspose, CV_32FC1);
  sigma = samplestranspose*samples/(samples.rows);
  cv::SVD::compute(sigma, w, u, vt);
  cv::transpose(u, utranspose);
  w = w+0.1;
  cv::sqrt(w,w);
  w = 1./w;
  cv::Mat D(w.rows,w.rows,CV_32F); 
  D = D.diag(w);
  samplestranspose = u*D*utranspose*samplestranspose;
  cv::transpose(samplestranspose, samples);
  
  double min;
  double max;
  cv::Point minLoc;
  cv::Point maxLoc;
  
  cv::minMaxLoc(samples, &min, &max, &minLoc, &maxLoc);
		
  cv::add(samples, -min, samples);
  
  samples = samples*(255.0/(max-min));
  
}

void Dictionary::getfilenames( std::vector<std::string>& filenames ){
  
  //open text file containing image file data
  std::string name = parameters.getStringParameter("dictionary_path") + "char.xml";
  std::ifstream infile(name.c_str());
  CHECK_MSG(infile.good(),"Error reading '" << name << "'.  Please check file exists and is named correctly");
  int numimages = 0;
  
  //Read image file names
  while (!infile.eof()){
    
    std::string line, param, file;
    std::stringstream tt;
    getline(infile,line);
    tt<<line;
    
    while( tt>>param && numimages < parameters.getIntParameter("dictionary_source_images") ){
      
      if(param == "<image" ){
	tt>>param;
	file = param.substr(param.find('"')+1);
	file = file.erase(file.find('"'));
    
	cv::Mat1f image(32, 32);
	cv::Mat1f originalimage;
	
	std::cout << parameters.getStringParameter("dictionary_path") + file << std::endl;
	filenames.push_back(parameters.getStringParameter("dictionary_path") + file);
	
	numimages++;
      }
    }
  }
}

void Dictionary::getsubimages( cv::Mat& samples, std::string filename ){
  
  cv::Mat1f image(32, 32);
  cv::Mat1f originalimage;
  
  //read in image file as gray scale
  originalimage = cv::imread(  filename , cv::IMREAD_GRAYSCALE );
  if(! originalimage.data ){
    std::cout << "could not read image " << filename << std::endl;
  }
  
  cv::resize(originalimage, image, image.size());
  
  //extract 8x8 subpca whiteningimages
  for(int x = 0; x < image.size().width - 7; ++x){
    for( int y = 0; y < image.size().height - 7; ++y){
      cv::Mat1f subimage(8,8);
      subimage = image(cv::Range(y,y+8), cv::Range(x,x+8)).clone();
      
      normalise(subimage);
      
      //add processed subimage to sample matrix to be passed to kmeans algorithim
      samples.push_back((subimage.reshape(0, 1)));
    }
  }
}

void Dictionary::normalise( cv::Mat1f & matrix ){
  
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
//   std::vector<Rectangle*> textboxes;
//   
//   getTextBoxes(parameters.getStringParameter("image_path_1"),  "apanar_06.08.2002/IMG_1305.JPG", textboxes );
//   
//   for( auto it = textboxes.begin(); it != textboxes.end(); ++it ){
//     (**it).print();
//   }
  // //   cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
// //   cv::imshow( "Display window", image);
// //   cv::namedWindow( "Sub window", cv::WINDOW_AUTOSIZE );
// //   cv::imshow( "Sub window", subimage);
//   
//   cv::waitKey(0);
  
//   cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
//   cv::imshow( "Display window", image);
//   cv::namedWindow( "Sub window", cv::WINDOW_AUTOSIZE );
//   cv::imshow( "Sub window", subimage);
//   
//   cv::waitKey(0);



// 	    //apply ZCA whitening
// 	    cv::Mat sigma, w, u, vt;
// 	    cv::Mat subimagetranspose, utranspose;
// 	    cv::transpose(subimage, subimagetranspose);
// 	    sigma = subimage*subimagetranspose/8;
// 	    cv::SVD::compute(sigma, w, u, vt);
// 	    cv::transpose(u, utranspose);
// 	    w = w+0.1;
// 	    cv::sqrt(w,w);
// 	    w = 1./w;
// 	    cv::Mat D(8,8,CV_32F); 
// 	    D = D.diag(w);
// 	    subimage = u*D*utranspose*subimage;


  //convert samples to floating point for insertion into kmeans function
  //samples.convertTo(samples, CV_32F);


