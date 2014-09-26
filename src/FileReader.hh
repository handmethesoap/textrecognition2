#ifndef FILEREADER_HH
#define FILEREADER_HH

#include <string>
#include <map>
#include <utility>

#include "Types.hh"
#include "Debug.hh"

//*******************************************************************************************************************
/*! Class for reading configuration from a file
*
* Configuration File Syntax:
*   - everything between a '#' character and the beginning of the next line is ignored
*   - lines can be empty
*   - line contain parameter key followed by white-spaces followed by their value
*
*  All possible keys (and the datatype of the value) have to be registered first:
*   - for example usage have a look at the FileReaderTest
*
*
*
*  This Skeleton is just a suggestion. If you are familiar with template programming
*  a possible alternative would be a version that does not need registering of all keys
*  and has a getter function like the following:
*      template<typename ExpectedType>
*      ExpectedType getValue( const std::string & key );
*/
//*******************************************************************************************************************
class FileReader
{
public:

	//register a new parameter with name key and initial int value
	void registerIntParameter( const std::string & key, int init = 0 );

	//register a new parameter with name key and initial double value
	void registerRealParameter( const std::string & key, real init = 0 );

	//register a new parameter with name key and initial string value
	void registerStringParameter( const std::string & key, const std::string & init = "" );

	//set a value for the key string with value in
	void setParameter( const std::string & key, const std::string & in );

	//set a ,"Failed attempt to access string parameter"value for the key string with value in
	void setParameter( const std::string & key, real in );

	//set a value for the key string with value in
	void setParameter( const std::string & key, int in );
	
	//check an int parameter exists
	bool checkIntParameterExists( const std::string & key ) const;
	
	//check a real parameter exists
	bool checkRealParameterExists( const std::string & key ) const;
	
	//check a string parameter exists
	bool checkStringParameterExists( const std::string & key ) const;

	// get the int value of key 
	inline int getIntParameter( const std::string & key ) const;

	// get the double value of key 
	inline real getRealParameter( const std::string & key ) const;

	// get the string value of key 
	inline std::string getStringParameter( const std::string & key ) const;

	//try to read all registered parameters from file name
	bool readFile( const std::string & name );

	//print out all parameters to std:out
	void printParameters() const;

private:
	
	std::map< const std::string, int > _intParameters;
	std::map< const std::string, real > _realParameters;
	std::map< const std::string, std::string > _stringParameters;
	
};


inline int FileReader::getIntParameter(const std::string &key) const
{
   std::map< const std::string, int >::const_iterator parameter = _intParameters.find(key);
   
   ASSERT_MSG(parameter != _intParameters.end(),"Failed attempt to access int parameter " << key);
   
   return parameter->second;
}

inline real FileReader::getRealParameter(const std::string &key) const
{
   std::map< const std::string, real >::const_iterator parameter =_realParameters.find(key);
   
   ASSERT_MSG(parameter != _realParameters.end(),"Failed attempt to access real parameter " << key);
   
   return parameter->second;
}

inline std::string FileReader::getStringParameter(const std::string &key) const
{
   std::map< const std::string, std::string >::const_iterator parameter = _stringParameters.find(key);
   
   ASSERT_MSG(parameter != _stringParameters.end(),"Failed attempt to access string parameter " << key);
   
   return parameter->second;
}




#endif //FILEREADER_HH

