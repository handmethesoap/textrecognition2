
#include <FileReader.hh>
#include <fstream>
#include <sstream>
#include <iostream>
#include <fstream>

void FileReader::registerIntParameter(const std::string &key, int init)
{
   _intParameters[key] = init;
}

void FileReader::registerRealParameter(const std::string &key, real init)
{
   _realParameters[key] = init;
}

void FileReader::registerStringParameter(const std::string &key, const std::string &init)
{
   _stringParameters[key] = init;
}

void FileReader::setParameter(const std::string &key, const std::string &in)
{
   _stringParameters[key] = in;
}

void FileReader::setParameter(const std::string &key, real in)
{
   _realParameters[key] = in;
}

void FileReader::setParameter(const std::string &key, int in)
{
   _intParameters[key] = in;
}


bool FileReader:: checkIntParameterExists( const std::string & key ) const {
  
   std::map< const std::string, int >::const_iterator parameter = _intParameters.find(key);
   
   if(parameter == _intParameters.end()){
     return 0;
   }
   else{
     return 1;
   }
}

bool FileReader::checkRealParameterExists( const std::string & key ) const {
  
  std::map< const std::string, real >::const_iterator parameter =_realParameters.find(key);
   
   if(parameter == _realParameters.end()){
     return 0;
   }
   else{
     return 1;
   }
}

bool FileReader::checkStringParameterExists( const std::string & key ) const {
  
   std::map< const std::string, std::string >::const_iterator parameter = _stringParameters.find(key);
   
   if(parameter == _stringParameters.end()){
     return 0;
   }
   else{
     return 1;
   }
}
   

bool FileReader::readFile(const std::string &name)
{
  std::ifstream infile(name.c_str());
   
  CHECK_MSG(infile.good(),"Error reading '" << name << "'.  Please check file exists and is named correctly");

  while (!infile.eof()){
    std::string line, param;
    std::stringstream tt;
    getline(infile,line);
    line.assign(line.substr(0,line.find('#')));
    tt<<line;
    tt>>param;
    
    if(_intParameters.find(param) != _intParameters.end())
    {
      tt >> _intParameters[param];
      CHECK_MSG(!tt.fail(), "Parameter " << param << " in file " << name << " is not an integer");
    }
    else if(_realParameters.find(param) != _realParameters.end())
    {
      tt >> _realParameters[param];
      CHECK_MSG(!tt.fail(), "Parameter " << param << " in file " << name << " is not a real");
    }
    else if(_stringParameters.find(param) != _stringParameters.end())
    {
      tt >> _stringParameters[param];
      CHECK_MSG(!tt.fail(), "Parameter " << param << " in file " << name << " is not a string");
    }
  }
  infile.close();

  return true;
}



void FileReader::printParameters() const
{
   for(auto it = _intParameters.begin(); it != _intParameters.end(); ++it)
   {
     std::cout << it->first << "  " << it->second << std::endl;
   }
   for(auto it = _realParameters.begin(); it != _realParameters.end(); ++it)
   {
     std::cout << it->first << "  " << it->second << std::endl;
   }
   for(auto it = _stringParameters.begin(); it != _stringParameters.end(); ++it)
   {
     std::cout << it->first << "  " << it->second << std::endl;
   }
}


