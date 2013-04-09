#include <panda/SerializedItemCollection.h>
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace panda
{
  const int SerializedItemCollection::INITIAL_CAPACITY = 1024 * 1024;
  SerializedItemCollection::SerializedItemCollection()
  {
    capacity = INITIAL_CAPACITY;
    totalSize = sizeof(int);
    storage = new char[capacity];
    numItems = reinterpret_cast<int * >(storage);
    *numItems = 0;
    lastItemIndex = -1;
  }
  SerializedItemCollection::SerializedItemCollection(const int initialCapacity)
  {
    capacity = std::max(INITIAL_CAPACITY, initialCapacity);
    totalSize = sizeof(int);
    storage = new char[capacity];
    numItems = reinterpret_cast<int * >(storage);
    *numItems = 0;
    lastItemIndex = -1;
  }
  SerializedItemCollection::SerializedItemCollection(const int pCapacity, const int pTotalSize, void * const relinquishedBuffer)
    : capacity(pCapacity),
      totalSize(pTotalSize),
      numItems(reinterpret_cast<int * >(relinquishedBuffer)),
      storage(reinterpret_cast<char * >(relinquishedBuffer))
  {
    lastItemIndex = -1;
  }
  SerializedItemCollection::~SerializedItemCollection()
  {
    delete [] storage;
  }
  // TBI
  
  void SerializedItemCollection::clear(){}

  int SerializedItemCollection::getTotalSize() const
  {
	  return 0;
  }

  int SerializedItemCollection::getItemCount() const
  {
	  return 0 ;
  }

  const void * SerializedItemCollection::getStorage() const{
	  
	  return 0;

  }//const

  void * SerializedItemCollection::getItem(const int itemNumber, int & key){
	  return 0;
  }//void

 
  const void * SerializedItemCollection::getItem(const int itemNumber, int & key) const{
 return 0;
  }

  //const Ray * SerializedItemCollection::getItem(const int itemNumber, int & key) const{
  //Ray *res = new Ray();
  //return res;
  //}//const

  //TODO
  void SerializedItemCollection::addItem(const int key, const Ray & value){
  }//void

    
  void SerializedItemCollection::addItem(const int key, const void * const value, const int valueSize){
  }

  //TODO
  void SerializedItemCollection::addItems(const int pTotalSize, const void * const buffer){
  }//void

}
