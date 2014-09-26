#ifndef ARRAY_HH
#define ARRAY_HH


#include <Types.hh>
#include "Debug.hh"
#include <iostream>
#include <iomanip>


//*******************************************************************************************************************
/*!  Array class for 1,2 and 3 dimensions
*
*    - all elements should be stored in a contiguous chunk of memory ( no vector<vector> ! )
*/
//*******************************************************************************************************************
template <class T>
class Array
{
public:
   // Constructors for 1D,2D and 3D
   Array( int xSize );
   Array( int xSize, int ySize );
   //Array( int xSize, int ySize, int zSize );


   // Depending on your implementation you might need the following:
   ~Array();
   Array(const Array& s);
   Array& operator= (const Array& s);


   // Access Operators for 1D, 2D and 3D
   inline T & operator () ( int i );
   inline T & operator () ( int i ,int j );
   //inline T & operator () ( int i, int j, int k );

   // for const Arrays the following access operators are required
   inline const T & operator () ( int i ) const;
   inline const T & operator () ( int i ,int j ) const;
   //inline const T & operator () ( int i, int j, int k ) const;



   // initialize the whole array with a constant value
   void fill( T value );


   // return total size of the array
   int getSize() const;

   // return xSize for dimension==0, ySize for dimension==1 and zSize for dimension==2
   // other dimension values are not allowed
   int getSize( int dimension ) const;


   // Print the whole array ( for debugging purposes )
   void print();

private:

    T* array_;
    int xSize_;
    int ySize_;
    int zSize_;

};


//===================================================================================================================
//
//  Inline Access Operators and Sizes
//
//===================================================================================================================


// Operator() 1D
template <class T>
inline T & Array<T>::operator ()(int i)
{
  
  ASSERT_MSG(i < xSize_, "You have attempted to access an array element larger than the upper bound in the x dimension");
  ASSERT_MSG(i >= 0, "You have attempted to access an array element smaller than the lower bound in the x dimension");
  
   return array_[i];
}

// Operator() 2D
template <class T>
inline T & Array<T>::operator ()(int i,int j)
{
    
  ASSERT_MSG(i < xSize_, "You have attempted to access an array element larger than the upper bound in the x dimension");
  ASSERT_MSG(j < ySize_, "You have attempted to access an array element larger than the upper bound in the y dimension");
  ASSERT_MSG(i >= 0, "You have attempted to access an array element smaller than the lower bound in the x dimension");
  ASSERT_MSG(j >= 0, "You have attempted to access an array element smaller than the lower bound in the y dimension");
 
   return array_[i + j*xSize_];
}

// Operator() 3D
// template <class T>
// inline T & Array<T>::operator ()(int i, int j, int k)
// {  
//   
//   ASSERT_MSG(i < xSize_, "You have attempted to access an array element larger than the upper bound in the x dimension");
//   ASSERT_MSG(j < ySize_, "You have attempted to access an array element larger than the upper bound in the y dimension");
//   ASSERT_MSG(k < zSize_, "You have attempted to access an array element larger than the upper bound in the z dimension");
//   ASSERT_MSG(i >= 0, "You have attempted to access an array element smaller than the lower bound in the x dimension");
//   ASSERT_MSG(j >= 0, "You have attempted to access an array element smaller than the lower bound in the y dimension");
//   ASSERT_MSG(k >= 0, "You have attempted to access an array element smaller than the lower bound in the z dimension");
//  
//    return array_[i + j*xSize_ + k*xSize_*ySize_];
// }

template <class T>
inline const T & Array<T>::operator () ( int i ) const
{

  ASSERT_MSG(i < xSize_, "You have attempted to access an array element larger than the upper bound in the x dimension");
  ASSERT_MSG(i >= 0, "You have attempted to access an array element smaller than the lower bound in the x dimension");
  
   return array_[i];
}

template <class T>
inline const T & Array<T>::operator () ( int i ,int j ) const
{
 
  ASSERT_MSG(i < xSize_, "You have attempted to access an array element larger than the upper bound in the x dimension");
  ASSERT_MSG(j < ySize_, "You have attempted to access an array element larger than the upper bound in the y dimension");
  ASSERT_MSG(i >= 0, "You have attempted to access an array element smaller than the lower bound in the x dimension");
  ASSERT_MSG(j >= 0, "You have attempted to access an array element smaller than the lower bound in the y dimension");
  
   return array_[i + j*xSize_];
}

// template <class T>
// inline const T & Array<T>::operator () ( int i, int j, int k ) const
// {
//   
//   ASSERT_MSG(i < xSize_, "You have attempted to access an array element larger than the upper bound in the x dimension");
//   ASSERT_MSG(j < ySize_, "You have attempted to access an array element larger than the upper bound in the y dimension");
//   ASSERT_MSG(k < zSize_, "You have attempted to access an array element larger than the upper bound in the z dimension");
//   ASSERT_MSG(i >= 0, "You have attempted to access an array element smaller than the lower bound in the x dimension");
//   ASSERT_MSG(j >= 0, "You have attempted to access an array element smaller than the lower bound in the y dimension");
//   ASSERT_MSG(k >= 0, "You have attempted to access an array element smaller than the lower bound in the z dimension");
//   
//    return array_[i + j*xSize_ + k*xSize_*ySize_];
// }

//===================================================================================================================
//
//  Constructors
//
//===================================================================================================================

template <class T>
Array<T>::Array( int xSize ) : xSize_(xSize), ySize_(1), zSize_(1)
{

  CHECK_MSG(xSize >= 0, "the x dimension of the array cannot be negative");

   array_ = new T[ xSize ];
}

template <class T>
Array<T>::Array( int xSize, int ySize ) : xSize_(xSize), ySize_(ySize), zSize_(1)
{

  CHECK_MSG(xSize >= 0, "the x dimension of the array cannot be negative");
  CHECK_MSG(ySize >= 0, "the y dimension of the array cannot be negative");

   array_ = new T[ xSize*ySize ];
}

// template <class T>
// Array<T>::Array( int xSize, int ySize, int zSize ) : xSize_(xSize), ySize_(ySize), zSize_(zSize)
// {
// 
//   CHECK_MSG(xSize >= 0, "the x dimension of the array cannot be negative");
//   CHECK_MSG(ySize >= 0, "the y dimension of the array cannot be negative");
//   CHECK_MSG(zSize >= 0, "the z dimension of the array cannot be negative");
// 
//    array_ = new T[ xSize*ySize*zSize ];
// }

template <class T>
Array<T>::Array(const Array& s) : xSize_(s.xSize_), ySize_(s.ySize_), zSize_(s.zSize_)
{
  array_ = new T[ s.xSize_*s.ySize_*s.zSize_ ];
  for( int i = 0; i < s.xSize_*s.ySize_*s.zSize_; ++i )
  {
    array_[i] = s.array_[i];
  }
}

template <class T>
Array<T>& Array<T>::operator= (const Array& s)
{
  Array tmp( s );
  
  std::swap( xSize_, tmp.xSize_ );
  std::swap( ySize_, tmp.ySize_ );
  std::swap( zSize_, tmp.zSize_ );
  std::swap( array_, tmp.array_ );
  
  return *this;
}



template <class T>
Array<T>::~Array()
{
  delete [] array_;
}


//===================================================================================================================
//
//  Convenience Functions
//
//===================================================================================================================


//initialize the whole array with a constant value
template <class T>
void Array<T>::fill( T value )
{
   // you might want to use std::fill() here
  std::fill(array_, array_ + this->getSize(), value);
}

// Print the whole array (for debugging purposes)
template <class T>
void Array<T>::print()
{
   // For 2D Arrays the positive x-coordinate goes to the right
   //                   positive y-coordinate goes upwards
   //      -> the line with highest y-value should be printed first
    for( int y = ySize_ - 1; y >= 0; --y )
    {
      for( int x = 0; x < xSize_; ++x ) 
      {
	std::cout << std::left << " "/*std::setw(13)*/ << (*this)(x, y);
      }
      std::cout << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << std::endl;
}

template <class T>
int Array<T>::getSize( int dimension ) const
{
   if( dimension == 0 )
   {
     return xSize_;
   }
   else if( dimension == 1 )
   {
     return ySize_;
   }
   else if( dimension == 2 )
   {
     return zSize_;
   }
   else
   {
     return 0;
   }
}

//return total size of the array
template <class T>
int Array<T>::getSize() const
{
   return xSize_*ySize_*zSize_;
}

#endif //ARRAY_HH

