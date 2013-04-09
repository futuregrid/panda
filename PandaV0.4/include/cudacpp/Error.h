#ifndef __ERROR_H__
#define __ERROR_H__

namespace cudacpp
{
  class String;

  /**
   * This class wraps cudaError_t and provides a mechanism for obtaining both
   * the integral representation and string representation thereof.
   */
  class Error
  {
    protected:
      /// The native cuda error, coerced from cudaError_t.
      int error;
    public:
      /**
       * Creates a new error object.
       *
       * @param errorVal The error which this object shall represent.
       */
      Error(const int errorVal);

      /// @return The integral value of this error.
      int getErrorValue() const;

      /**
       * @return The string representation of this error, as returned
       *         by cudaGetErrorString().
       */
      String toString() const;
  };
}

#endif
