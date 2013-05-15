#ifndef __GPMR_ITEMCOLLECTION_H__
#define __GPMR_ITEMCOLLECTION_H__

namespace panda
{
  class ItemCollection
  {
    public:
      ItemCollection();
      ItemCollection(const int totalSize, const void * const buffer);
      virtual ~ItemCollection();

      virtual void clear() = 0;

      virtual int getTotalSize() const = 0;
      virtual int getItemCount() const = 0;
      virtual       void * getItem(const int itemNumber, int & key) = 0;
      virtual const void * getItem(const int itemNumber, int & key) const = 0;

      virtual void addItem(const int key, const void * const data) = 0;
      virtual void addItems(const int pTotalSize, const void * const buffer) = 0;
  };
}

#endif
