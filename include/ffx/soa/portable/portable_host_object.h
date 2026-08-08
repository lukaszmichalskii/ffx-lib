#pragma once

#include "ffx/core/mem/buf/host_buffer.h"

namespace ffx::soa {

  // generic object in host memory
  template <typename T>
  class PortableHostObject {
  public:
    using Product = T;
    using Buffer = ffx::host_buffer<Product>;
    using ConstBuffer = ffx::const_host_buffer<Product>;

    static_assert(std::is_trivially_destructible_v<Product>);

    PortableHostObject() = delete;

    // Note that in contrast to the variadic template overload, this
    // constructor does not initialize the contained object
    PortableHostObject(ffx_alpaka::DevHost const& host)
        // allocate pageable host memory
        : buffer_{ffx::make_host_buffer<Product>()}, product_{buffer_->data()} {
      assert(reinterpret_cast<uintptr_t>(product_) % alignof(Product) == 0);
    }

    template <typename... Args>
    PortableHostObject(ffx_alpaka::DevHost const& host, Args&&... args)
        // allocate pageable host memory
        : buffer_{ffx::make_host_buffer<Product>()},
          product_{new(buffer_->data()) Product(std::forward<Args>(args)...)} {
      assert(reinterpret_cast<uintptr_t>(product_) % alignof(Product) == 0);
    }

    // Note that in contrast to the variadic template overload, this
    // constructor does not initialize the contained object
    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    PortableHostObject(TQueue const& queue)
        // allocate pinned host memory associated to the given work queue, accessible by the queue's device
        : buffer_{ffx::make_host_buffer<Product>(queue)}, product_{buffer_->data()} {
      assert(reinterpret_cast<uintptr_t>(product_) % alignof(Product) == 0);
    }

    template <typename TQueue, typename... Args, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    PortableHostObject(TQueue const& queue, Args&&... args)
        // allocate pinned host memory associated to the given work queue, accessible by the queue's device
        : buffer_{ffx::make_host_buffer<Product>(queue)},
          product_{new(buffer_->data()) Product(std::forward<Args>(args)...)} {
      assert(reinterpret_cast<uintptr_t>(product_) % alignof(Product) == 0);
    }

    // non-copyable
    PortableHostObject(PortableHostObject const&) = delete;
    PortableHostObject& operator=(PortableHostObject const&) = delete;

    // movable
    PortableHostObject(PortableHostObject&&) = default;
    PortableHostObject& operator=(PortableHostObject&&) = default;

    // default destructor
    ~PortableHostObject() = default;

    // access the product
    Product& value() { return *product_; }
    Product const& value() const { return *product_; }
    Product const& const_value() const { return *product_; }

    Product* data() { return product_; }
    Product const* data() const { return product_; }
    Product const* const_data() const { return product_; }

    Product& operator*() { return *product_; }
    Product const& operator*() const { return *product_; }

    Product* operator->() { return product_; }
    Product const* operator->() const { return product_; }

    // access the buffer
    Buffer buffer() { return *buffer_; }
    ConstBuffer buffer() const { return *buffer_; }
    ConstBuffer const_buffer() const { return *buffer_; }

    // erases the data in the Buffer by writing zeros (bytes containing '\0') to it
    void zeroInitialise() {
      std::memset(std::data(*buffer_), 0x00, alpaka::getExtentProduct(*buffer_) * sizeof(std::byte));
    }

    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    void zeroInitialise(TQueue&& queue) {
      alpaka::memset(std::forward<TQueue>(queue), *buffer_, 0x00);
    }

    // part of the ROOT read streamer
    static void ROOTReadStreamer(PortableHostObject* newObj, Product& product) {
      // destroy the default-constructed object
      newObj->~PortableHostObject();
      // use the global "host" object returned by ffx::host()
      new (newObj) PortableHostObject(ffx::host());
      // copy the data from the on-file object to the new one
      std::memcpy(newObj->product_, &product, sizeof(Product));
    }

  private:
    std::optional<Buffer> buffer_;  //!
    Product* product_ = nullptr;
  };

}  // namespace ffx::soa
