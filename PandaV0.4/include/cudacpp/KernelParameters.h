#ifndef __CUDACPP_KERNELPARAMETERS_H__
#define __CUDACPP_KERNELPARAMETERS_H__

namespace cudacpp
{
  /**
   * This class stores arguments to be passed to a kernel. The number of
   * arguments is unchecked, as is the type of the arguments. It is up to to the
   * user to perform proper type coercion and to pass the proper number of
   * arguments.
   */

  class KernelParameters
  {
    protected:
      /// An array of pointers to the arguments.
      void ** mem;

      /// An array of sizes of the arguments.
      int * sizes;

      /// The number of arguments for which space has been allocated.
      int numArgs;

      /// The total number of arguments to be passed.
      int paramCount;

      /**
       * Allocates enough space to store this parameter and then stores a copy
       * on the heap.
       *
       * @param buf The bytes used to represent this parameter.
       * @param size The number of bytes needed to represent this parameter.
       */
      void storeParam(const void * const buf, const int size);
    public:
      /**
       * Allocated enough space for the specified number of arguments.
       *
       * @param numParams The total number of arguments to pass to the kernel.
       */
      KernelParameters(const int numParams);

      /// Destructor, 'nough said.
      ~KernelParameters();

      /// @return The total number of parameters to pass to the kernel.
      inline int getNumParams() const
      {
        return paramCount;
      }

      /// Clear all arguments and free used heap spce.
      inline void clear()
      {
        for (int i = 0; i < numArgs; ++i) delete [] (char * )mem[i];
        numArgs = 0;
      }

      /// Set the unmber of arguments to be zero.
      inline void set()
      {
        clear();
        numArgs = 0;
      }

      /**
       * Store the parameter passed as the only parameter to this function.
       *
       * @param t1 The first parameter for the kernel.
       */
      template <typename T1>
      void set(const T1 & t1)
      {
        set();
        storeParam(&t1, sizeof(t1));
      }

      /**
       * Store the parameters for later use by a kernel invocation.
       *
       * @param t1 The first parameter for the kernel.
       * @param t2 The second parameter for the kernel.
       */
      template <typename T1, typename T2>
      void set(const T1 & t1, const T2 & t2)
      {
        set(t1);
        storeParam(&t2, sizeof(t2));
      }

      /**
       * Store the parameters for later use by a kernel invocation.
       *
       * @param t1 The first parameter for the kernel.
       * @param t2 The second parameter for the kernel.
       * @param t3 The second parameter for the kernel.
       */
      template <typename T1, typename T2, typename T3>
      void set(const T1 & t1, const T2 & t2, const T3 & t3)
      {
        set(t1, t2);
        storeParam(&t3, sizeof(t3));
      }

      /**
       * Store the parameters for later use by a kernel invocation.
       *
       * @param t1 The first parameter for the kernel.
       * @param t2 The second parameter for the kernel.
       * @param t3 The second parameter for the kernel.
       * @param t4 The fourth parameter for the kernel.
       */
      template <typename T1, typename T2, typename T3, typename T4>
      void set(const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4)
      {
        set(t1, t2, t3);
        storeParam(&t4, sizeof(t4));
      }

      /**
       * Store the parameters for later use by a kernel invocation.
       *
       * @param t1 The first parameter for the kernel.
       * @param t2 The second parameter for the kernel.
       * @param t3 The second parameter for the kernel.
       * @param t4 The fourth parameter for the kernel.
       * @param t5 The fifth parameter for the kernel.
       */
      template <typename T1, typename T2, typename T3, typename T4, typename T5>
      void set(const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5)
      {
        set(t1, t2, t3, t4);
        storeParam(&t5, sizeof(t5));
      }

      /**
       * Store the parameters for later use by a kernel invocation.
       *
       * @param t1 The first parameter for the kernel.
       * @param t2 The second parameter for the kernel.
       * @param t3 The second parameter for the kernel.
       * @param t4 The fourth parameter for the kernel.
       * @param t5 The fifth parameter for the kernel.
       * @param t6 The sixth parameter for the kernel.
       */
      template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
      void set(const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5, const T6 & t6)
      {
        set(t1, t2, t3, t4, t5);
        storeParam(&t6, sizeof(t6));
      }

      /**
       * Store the parameters for later use by a kernel invocation.
       *
       * @param t1 The first parameter for the kernel.
       * @param t2 The second parameter for the kernel.
       * @param t3 The second parameter for the kernel.
       * @param t4 The fourth parameter for the kernel.
       * @param t5 The fifth parameter for the kernel.
       * @param t6 The sixth parameter for the kernel.
       * @param t7 The seventh parameter for the kernel.
       */
      template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
      void set(const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5, const T6 & t6, const T7 & t7)
      {
        set(t1, t2, t3, t4, t5, t6);
        storeParam(&t7, sizeof(t7));
      }

      /**
       * Store the parameters for later use by a kernel invocation.
       *
       * @param t1 The first parameter for the kernel.
       * @param t2 The second parameter for the kernel.
       * @param t3 The second parameter for the kernel.
       * @param t4 The fourth parameter for the kernel.
       * @param t5 The fifth parameter for the kernel.
       * @param t6 The sixth parameter for the kernel.
       * @param t7 The seventh parameter for the kernel.
       * @param t8 The eighth parameter for the kernel.
       */
      template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
      void set(const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5, const T6 & t6, const T7 & t7, const T8 & t8)
      {
        set(t1, t2, t3, t4, t5, t6, t7);
        storeParam(&t8, sizeof(t8));
      }

      /**
       * Store the parameters for later use by a kernel invocation.
       *
       * @param t1 The first parameter for the kernel.
       * @param t2 The second parameter for the kernel.
       * @param t3 The second parameter for the kernel.
       * @param t4 The fourth parameter for the kernel.
       * @param t5 The fifth parameter for the kernel.
       * @param t6 The sixth parameter for the kernel.
       * @param t7 The seventh parameter for the kernel.
       * @param t8 The eighth parameter for the kernel.
       * @param t9 The ninth parameter for the kernel.
       */
      template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
      void set(const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5, const T6 & t6, const T7 & t7, const T8 & t8, const T9 & t9)
      {
        set(t1, t2, t3, t4, t5, t6, t7, t8);
        storeParam(&t9, sizeof(t9));
      }

      /**
       * Store the parameters for later use by a kernel invocation.
       *
       * @param t1 The first parameter for the kernel.
       * @param t2 The second parameter for the kernel.
       * @param t3 The second parameter for the kernel.
       * @param t4 The fourth parameter for the kernel.
       * @param t5 The fifth parameter for the kernel.
       * @param t6 The sixth parameter for the kernel.
       * @param t7 The seventh parameter for the kernel.
       * @param t8 The eighth parameter for the kernel.
       * @param t9 The ninth parameter for the kernel.
       * @param t10 The tenth parameter for the kernel.
       */
      template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
      void set(const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5, const T6 & t6, const T7 & t7, const T8 & t8, const T9 & t9, const T10 & t10)
      {
        set(t1, t2, t3, t4, t5, t6, t7, t8, t9);
        storeParam(&t10, sizeof(t10));
      }

      /**
       * Store the parameters for later use by a kernel invocation.
       *
       * @param t1 The first parameter for the kernel.
       * @param t2 The second parameter for the kernel.
       * @param t3 The second parameter for the kernel.
       * @param t4 The fourth parameter for the kernel.
       * @param t5 The fifth parameter for the kernel.
       * @param t6 The sixth parameter for the kernel.
       * @param t7 The seventh parameter for the kernel.
       * @param t8 The eighth parameter for the kernel.
       * @param t9 The ninth parameter for the kernel.
       * @param t10 The tenth parameter for the kernel.
       * @param t11 The eleventh parameter for the kernel.
       */
      template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11>
      void set(const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5, const T6 & t6, const T7 & t7, const T8 & t8, const T9 & t9, const T10 & t10, const T11 & t11)
      {
        set(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10);
        storeParam(&t11, sizeof(t11));
      }

      /**
       * Store the parameters for later use by a kernel invocation.
       *
       * @param t1 The first parameter for the kernel.
       * @param t2 The second parameter for the kernel.
       * @param t3 The second parameter for the kernel.
       * @param t4 The fourth parameter for the kernel.
       * @param t5 The fifth parameter for the kernel.
       * @param t6 The sixth parameter for the kernel.
       * @param t7 The seventh parameter for the kernel.
       * @param t8 The eighth parameter for the kernel.
       * @param t9 The ninth parameter for the kernel.
       * @param t10 The tenth parameter for the kernel.
       * @param t11 The eleventh parameter for the kernel.
       * @param t12 The twelfth parameter for the kernel.
       */
      template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12>
      void set(const T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5, const T6 & t6, const T7 & t7, const T8 & t8, const T9 & t9, const T10 & t10, const T11 & t11, const T12 & t12)
      {
        set(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11);
        storeParam(&t12, sizeof(t12));
      }

      /**
       * Return the index<sup>th</sup> argument as one of type T.
       *
       * @param index The index of the element which is to be returned.
       * @return The index<sup>th</sup> parameter.
       */
      template <typename T>
      T & get(const int index)
      {
        return *reinterpret_cast<T * >(mem[index]);
      }

      /**
       * Return the index<sup>th</sup> argument as one of type T.
       *
       * @param index The index of the element which is to be returned.
       * @return The index<sup>th</sup> parameter.
       */
      template <typename T>
      const T & get(const int index) const
      {
        return *reinterpret_cast<const T * >(mem[index]);
      }
  };

}

#endif
