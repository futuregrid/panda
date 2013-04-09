#ifndef __CUDACPP_EVENT_H__
#define __CUDACPP_EVENT_H__

#include <driver_types.h>

namespace cudacpp
{
  class Stream;

  /**
   * The Event class encapsulates a cudaEvent_t object. The event object is
   * automatically created within the constructor, so it is important to set the
   * CUDA device before creating a stream. Unexpected behavior may result if an
   * event is created for one device and used on another.
   *
   * For more details, please consult the CUDA reference manual.
   */
  class Event
  {
    protected:
      /// The pointer to the cudaEvent_t.
      cudaEvent_t handle;
    public:
      /// The default constructor. Simply calls cudaStreamCreate.
      Event();
      /// Simply calls cudaEventDestroy on the native handle.
      ~Event();

      /**
       * Simply calls cudaEventRecord with the stream.
       *
       * @param stream The stream from which to record an event.
       */
      void record(Stream * stream);

      /// Simply calls cudaStreamSynchronize on the native handle.
      void sync();

      /**
       * Simply calls cudaEventQuery on the native handle.
       *
       * @return True iff cudaEventQuery(handle) == cudaSuccess, or in other
       * words, if all commands in the stream have executed successfully.
       */
      bool query();

      /**
       * Returns the number of elapsed seconds between the finish time of this
       * event and the finish time of end. Both events must have been recorded
       * and successfully executed.
       *
       * @param end The other event from which to measure.
       */
      double getElapsedSeconds(const Event * end);

      /**
       * @return The native event handle. Public as a side-effect of me hating
       *         the friend keyword.
       */
      cudaEvent_t & getHandle();
  };
}

#endif
