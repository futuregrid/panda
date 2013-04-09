#ifndef __GPMR_SERIALIZEDITEMCOLLECTION_H__
#define __GPMR_SERIALIZEDITEMCOLLECTION_H__

#include <panda/ItemCollection.h>

namespace panda
{
	struct Ray
	{
    int i;
	};

  class SerializedItemCollection : public ItemCollection
  {
    protected:
      int capacity, totalSize;
      int * numItems;
      char * storage;
      mutable char * lastItem;
      mutable int lastItemIndex;

      static const int INITIAL_CAPACITY;
    public:
      SerializedItemCollection();
	  
      SerializedItemCollection(const int initialCapacity);
      SerializedItemCollection(const int pCapacity, const int pTotalSize, void * const relinquishedBuffer);
      virtual ~SerializedItemCollection();

      virtual void clear();
      virtual int getTotalSize() const;
      virtual int getItemCount() const;
      virtual const void * getStorage() const;
      virtual       void * getItem(const int itemNumber, int & key);
      virtual const void * getItem(const int itemNumber, int & key) const;

	  virtual void addItem(const int key, const Ray & value);
      virtual void addItem(const int key, const void * const value, const int valueSize);
      virtual void addItems(const int pTotalSize, const void * const buffer);
  };
}

#endif
