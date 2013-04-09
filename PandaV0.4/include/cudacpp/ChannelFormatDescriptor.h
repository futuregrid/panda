#ifndef __CUDACPP_CHANNELFORMATDESCRIPTOR_H__
#define __CUDACPP_CHANNELFORMATDESCRIPTOR_H__

namespace cudacpp
{
  /**
   * A wrapper for the cudaChannelFormatDesc structure.
   *
   * To use this class, one should call the static, templated, create function.
   * To know which arguments for the template are acceptable, please see the
   * toolkit header file "channel_descriptor.h".
   */
  class ChannelFormatDescriptor
  {
    protected:
      /// A pointer to a cudaChannelFormatDesc.
      void * handle;
    public:
      /// Mimics the cudaChannelFormatKind enum.
      enum
      {
        CHANNEL_FORMAT_SIGNED = 0,
        CHANNEL_FORMAT_UNSIGNED,
        CHANNEL_FORMAT_FLOAT,
        CHANNEL_FORMAT_NONE,
      };
      /// Default constructor. This is blocked, use create instead.
      ChannelFormatDescriptor();
    public:
      /**
       * Creates a new descriptor using the specialized template argument.
       *
       * @param T The type of data held within the channel.
       * @return A new descriptor, assuming a valid type was passed.
       */
      template <typename T> static ChannelFormatDescriptor * create();
      /// Default constructor.
      ~ChannelFormatDescriptor();

      /// @return handle->x.
      int getX() const;
      /// @return handle->y.
      int getY() const;
      /// @return handle->z.
      int getZ() const;
      /// @return The matching enum for handle->f.
      int getFormat() const;
      /// @return handle.
      void * getHandle();
      /// @return handle.
      const void * getHandle() const;
  };

}

#endif
