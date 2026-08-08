#pragma once

#include "ffx/core/mem/buf/device_buffer.h"
#include "ffx/soa/portable/portable_collection_common.h"

namespace ffx::soa {

  // generic SoA-based product in device memory
  template <concepts::device TDev, typename T>
  class PortableDeviceCollection {
    static_assert(not std::is_same_v<TDev, ffx_alpaka::DevHost>,
                  "Use PortableHostCollection<T> instead of PortableDeviceCollection<T, DevHost>");

  public:
    using Layout = T;
    using View = typename Layout::View;
    using ConstView = typename Layout::ConstView;
    using Descriptor = typename Layout::Descriptor;
    using ConstDescriptor = typename Layout::ConstDescriptor;
    using Buffer = ffx::device_buffer<TDev, std::byte[]>;
    using ConstBuffer = ffx::const_device_buffer<TDev, std::byte[]>;

    PortableDeviceCollection() = delete;

    template <std::integral Int>
    PortableDeviceCollection(TDev const& device, const Int size)
      requires(!requires { Layout::blocksNumber; })
        : buffer_{ffx::make_device_buffer<std::byte[]>(device, Layout::computeDataSize(detail::size_cast(size)))},
          layout_{buffer_->data(), detail::size_cast(size)},
          view_{layout_} {
      // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
      assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    }

    template <concepts::queue TQueue, std::integral Int>
      requires((!requires { Layout::blocksNumber; }))
    PortableDeviceCollection(TQueue const& queue, const Int size)
        : buffer_{ffx::make_device_buffer<std::byte[]>(queue, Layout::computeDataSize(detail::size_cast(size)))},
          layout_{buffer_->data(), detail::size_cast(size)},
          view_{layout_} {
      // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
      assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    }

    // constructor for a SoABlocks-layout, taking per-block sizes as variadic integral arguments
    template <std::integral... Ints>
    explicit PortableDeviceCollection(TDev const& device, const Ints... sizes)
      requires requires { Layout::blocksNumber; } && (sizeof...(Ints) == static_cast<std::size_t>(Layout::blocksNumber))
        : PortableDeviceCollection(device, std::to_array({detail::size_cast(sizes)...})) {}

    // constructor for a SoABlocks-layout, taking per-block sizes as variadic integral arguments
    template <concepts::queue TQueue, std::integral... Ints>
    explicit PortableDeviceCollection(TQueue const& queue, const Ints... sizes)
      requires requires { Layout::blocksNumber; } && (sizeof...(Ints) == static_cast<std::size_t>(Layout::blocksNumber))
        : PortableDeviceCollection(queue, std::to_array({detail::size_cast(sizes)...})) {}

    // constructor for a SoABlocks-layout, taking per-block sizes as a fixed-size array
    template <std::size_t N>
    explicit PortableDeviceCollection(TDev const& device, std::array<std::size_t, N> const& sizes)
      requires requires { Layout::blocksNumber; } && (N == static_cast<std::size_t>(Layout::blocksNumber))
        : buffer_{ffx::make_device_buffer<std::byte[]>(device, Layout::computeDataSize(sizes))},
          layout_{buffer_->data(), sizes},
          view_{layout_} {
      // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
      assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    }

    // constructor for a SoABlocks-layout, taking per-block sizes as a fixed-size array
    template <concepts::queue TQueue, std::size_t N>
    explicit PortableDeviceCollection(TQueue const& queue, std::array<std::size_t, N> const& sizes)
      requires requires { Layout::blocksNumber; } && (N == static_cast<std::size_t>(Layout::blocksNumber))
        : buffer_{ffx::make_device_buffer<std::byte[]>(queue, Layout::computeDataSize(sizes))},
          layout_{buffer_->data(), sizes},
          view_{layout_} {
      // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
      assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    }

    // non-copyable
    PortableDeviceCollection(PortableDeviceCollection const&) = delete;
    PortableDeviceCollection& operator=(PortableDeviceCollection const&) = delete;

    // movable
    PortableDeviceCollection(PortableDeviceCollection&&) = default;
    PortableDeviceCollection& operator=(PortableDeviceCollection&&) = default;

    // default destructor
    ~PortableDeviceCollection() = default;

    // access the View
    View& view() { return view_; }
    ConstView const& view() const { return view_; }
    ConstView const& const_view() const { return view_; }

    View& operator*() { return view_; }
    ConstView const& operator*() const { return view_; }

    View* operator->() { return &view_; }
    ConstView const* operator->() const { return &view_; }

    // access the Buffer
    Buffer buffer() { return *buffer_; }
    ConstBuffer buffer() const { return *buffer_; }
    ConstBuffer const_buffer() const { return *buffer_; }

    // erases the data in the Buffer by writing zeros (bytes containing '\0') to it
    template <concepts::queue TQueue>
    void zeroInitialise(TQueue&& queue) {
      alpaka::memset(std::forward<TQueue>(queue), *buffer_, 0x00);
    }

    // Copy column by column or block by block heterogeneously for device to host/device data transfer.
    template <concepts::queue TQueue>
    void deepCopy(TQueue& queue, ConstView const& view) {
      ConstDescriptor desc{view};
      Descriptor desc_{view_};
      detail::deepCopy(queue, desc_, desc);
    }

    // Either Layout::size_type for normal layouts or std::array<Layout::size_type, N> for SoABlocks layouts
    auto size() const { return layout_.metadata().size(); }

  private:
    // Data members
    std::optional<Buffer> buffer_;  //!
    Layout layout_;                 //
    View view_;                     //!
  };

}  // namespace ffx::soa