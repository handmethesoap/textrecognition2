#include "Dictionary.hh"
#include <fstream>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>

void Dictionary::generate(){
  
  std::cout << "-----Generating Dictionary-----" << std::endl;
  
  //open text file containing image file data
  std::string name = parameters.getStringParameter("dictionary_path") + "char.xml";
  std::ifstream infile(name.c_str());
  CHECK_MSG(infile.good(),"Error reading '" << name << "'.  Please check file exists and is named correctly");
  
  //Read image file names
  std::vector<std::string> filenames;
  getfilenames( filenames );

  //get subimages from each image to be analysed
  cv::Mat samples;
  for( std::vector<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it ){   
    getsubimages( samples, *it );
  }

  zcawhiten(samples);
     
  std::cout << "Performing kmeans calculation" << std::endl;
  cv::Mat labels;
  cv::TermCriteria criteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,
                            parameters.getIntParameter("kmeans_iter"),
                            parameters.getRealParameter("kmeans_eps"));

  cv::kmeans(samples, parameters.getIntParameter("dictionary_length"),
             labels, criteria, parameters.getIntParameter("kmeans_attempts"), cv::KMEANS_PP_CENTERS,
             centers);

  //save dictionary images to file
  printfiles();

}

void Dictionary::printfiles(){
  
  cv::FileStorage fs(parameters.getStringParameter("dictionary_save_path") + "dictdata.yml", cv::FileStorage::WRITE);
  if(fs.isOpened() != 1){
    std::cout << "Cannot open " << parameters.getStringParameter("dictionary_save_path") + "dictdata.yml" << " exiting.." << std::endl;
    exit(0);
  }
  fs << "Dictionary" << centers;
  fs << "matmean" << matmean;
  fs << "zca" << zca;
  fs.release();  
}

void Dictionary::read(){
  
  cv::FileStorage fs(parameters.getStringParameter("dictionary_save_path") + "dictdata.yml", cv::FileStorage::READ);
  if(fs.isOpened() != 1){
    std::cout << "Cannot open " << parameters.getStringParameter("dictionary_save_path") + "dictdata.yml" << " exiting.." << std::endl;
    exit(0);
  }
    
  fs["Dictionary"] >> centers;
  fs ["matmean"] >> matmean;
  fs ["zca"] >> zca;
  fs.release();
}
  
void Dictionary::zcawhiten( cv::Mat & samples ){
  // TODO: make sure samples are zero-mean
  cv::Scalar stddev;
  cv::meanStdDev(samples, matmean, stddev);
  cv::subtract(samples, matmean[0], samples);
  
  //apply ZCA whitening
  cv::Mat sigma, vt;
  cv::Mat samplestranspose, utranspose;
  cv::transpose(samples, samplestranspose);
  sigma = samplestranspose*samples/(samples.rows);
  cv::SVD::compute(sigma, w, u, vt, cv::SVD::FULL_UV);
  cv::transpose(u, utranspose);
  w = w + 0.001;
  cv::sqrt(w,w);
  w = 1./w;
  cv::Mat D(w.rows,w.rows,CV_32F); 
  D = D.diag(w);
  zca = u*D*utranspose;
  samples = samples*zca;
  
}

void Dictionary::zcawhitener( cv::Mat & samplesin ) const{
 
  //apply ZCA whitening
  cv::Mat samples = samplesin;
  samples = samples.reshape(0,1);
  cv::subtract(samples, matmean[0], samples);
  samples = samples*zca;
  
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
      
      subimage /= 255.f;
      
      //add processed subimage to sample matrix to be passed to kmeans algorithim
      samples.push_back((subimage.reshape(0, 1)));
    }
  }
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


