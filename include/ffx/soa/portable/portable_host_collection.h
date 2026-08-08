#pragma once

#include <optional>

#include "ffx/core/detail/concepts.h"
#include "ffx/core/mem/buf/host_buffer.h"
#include "ffx/soa/portable/portable_collection_common.h"

namespace ffx::soa {

  // generic SoA-based product in host memory
  template <typename T>
  class PortableHostCollection {
  public:
    using Layout = T;
    using View = typename Layout::View;
    using ConstView = typename Layout::ConstView;
    using Descriptor = typename Layout::Descriptor;
    using ConstDescriptor = typename Layout::ConstDescriptor;
    using Buffer = ffx::host_buffer<std::byte[]>;
    using ConstBuffer = ffx::const_host_buffer<std::byte[]>;

    PortableHostCollection() = delete;

    template <std::integral Int>
    PortableHostCollection(ffx_alpaka::DevHost const& host, const Int size)
      requires(!requires { Layout::blocksNumber; })
        // allocate pageable host memory
        : buffer_{ffx::make_host_buffer<std::byte[]>(Layout::computeDataSize(detail::size_cast(size)))},
          layout_{buffer_->data(), detail::size_cast(size)},
          view_{layout_} {
      // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
      assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    }

    template <concepts::queue TQueue, std::integral Int>
      requires((!requires { Layout::blocksNumber; }))
    PortableHostCollection(TQueue const& queue, const Int size)
        // allocate pinned host memory associated to the given work queue, accessible by the queue's device
        : buffer_{ffx::make_host_buffer<std::byte[]>(queue, Layout::computeDataSize(detail::size_cast(size)))},
          layout_{buffer_->data(), detail::size_cast(size)},
          view_{layout_} {
      // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
      assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    }

    // constructor for code that does not use alpaka explicitly, using the global "host" object returned by ffx::host()
    template <std::integral Int>
    PortableHostCollection(const Int size)
      requires(!requires { Layout::blocksNumber; })
        : PortableHostCollection(ffx::host(), size) {}

    // constructor for code that does not use alpaka explicitly, using the global "host" object returned by ffx::host()
    // constructor for a SoABlocks-layout, taking per-block sizes as variadic integral arguments
    template <std::integral... Ints>
    PortableHostCollection(const Ints... sizes)
      requires requires { Layout::blocksNumber; } && (sizeof...(Ints) == static_cast<std::size_t>(Layout::blocksNumber))
        : PortableHostCollection(ffx::host(), std::to_array({detail::size_cast(sizes)...})) {}

    // constructor for a SoABlocks-layout, taking per-block sizes as variadic integral arguments
    template <std::integral... Ints>
    explicit PortableHostCollection(ffx_alpaka::DevHost const& host, const Ints... sizes)
      requires requires { Layout::blocksNumber; } && (sizeof...(Ints) == static_cast<std::size_t>(Layout::blocksNumber))
        // allocate pageable host memory
        : PortableHostCollection(host, std::to_array({detail::size_cast(sizes)...})) {}

    // constructor for a SoABlocks-layout, taking per-block sizes as variadic integral arguments
    template <concepts::queue TQueue, std::integral... Ints>
    explicit PortableHostCollection(TQueue const& queue, const Ints... sizes)
      requires requires { Layout::blocksNumber; } && (sizeof...(Ints) == static_cast<std::size_t>(Layout::blocksNumber))
        // allocate pinned host memory associated to the given work queue, accessible by the queue's device
        : PortableHostCollection(queue, std::to_array({detail::size_cast(sizes)...})) {}

    // constructor for a SoABlocks-layout, taking per-block sizes as a fixed-size array
    template <std::size_t N>
    explicit PortableHostCollection(ffx_alpaka::DevHost const& host, std::array<std::size_t, N> const& sizes)
      requires requires { Layout::blocksNumber; } && (N == static_cast<std::size_t>(Layout::blocksNumber))
        // allocate pageable host memory
        : buffer_{ffx::make_host_buffer<std::byte[]>(Layout::computeDataSize(sizes))},
          layout_{buffer_->data(), sizes},
          view_{layout_} {
      // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
      assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    }

    // constructor for a SoABlocks-layout, taking per-block sizes as a fixed-size array
    template <concepts::queue TQueue, std::size_t N>
    explicit PortableHostCollection(TQueue const& queue, std::array<std::size_t, N> const& sizes)
      requires requires { Layout::blocksNumber; } && (N == static_cast<std::size_t>(Layout::blocksNumber))
        // allocate pinned host memory associated to the given work queue, accessible by the queue's device
        : buffer_{ffx::make_host_buffer<std::byte[]>(queue, Layout::computeDataSize(sizes))},
          layout_{buffer_->data(), sizes},
          view_{layout_} {
      // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
      assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout::alignment == 0);
    }

    // non-copyable
    PortableHostCollection(PortableHostCollection const&) = delete;
    PortableHostCollection& operator=(PortableHostCollection const&) = delete;

    // movable
    PortableHostCollection(PortableHostCollection&&) = default;
    PortableHostCollection& operator=(PortableHostCollection&&) = default;

    // default destructor
    ~PortableHostCollection() = default;

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
    void zeroInitialise() {
      std::memset(std::data(*buffer_), 0x00, alpaka::getExtentProduct(*buffer_) * sizeof(std::byte));
    }

    template <concepts::queue TQueue>
    void zeroInitialise(TQueue&& queue) {
      alpaka::memset(std::forward<TQueue>(queue), *buffer_, 0x00);
    }

    // Copy column by column the content of the given ConstView into this PortableHostCollection.
    // The view must point to data in host memory.
    void deepCopy(ConstView const& view) { layout_.deepCopy(view); }

    // Copy column by column or block by block heterogeneously for device to host data transfer.
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