#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <nvtx3/nvToolsExt.h>

namespace nvtx_tools {

  namespace colors {
    inline constexpr uint32_t Black = 0x00000000;
    inline constexpr uint32_t Red = 0x00ff0000;
    inline constexpr uint32_t DarkGreen = 0x00009900;
    inline constexpr uint32_t Green = 0x0000ff00;
    inline constexpr uint32_t LightGreen = 0x00ccffcc;
    inline constexpr uint32_t Blue = 0x000000ff;
    inline constexpr uint32_t Amber = 0x00ffbf00;
    inline constexpr uint32_t LightAmber = 0x00fff2cc;
    inline constexpr uint32_t White = 0x00ffffff;
  }  // namespace colors

  using DomainHandle = nvtxDomainHandle_t;
  using RangeId = nvtxRangeId_t;
  inline constexpr RangeId InvalidRangeId = 0xfffffffffffffffful;
  inline constexpr DomainHandle DefaultDomain = nullptr;

  [[nodiscard]] inline DomainHandle createDomain(const char* name) { return nvtxDomainCreateA(name); }

  inline void destroyDomain(DomainHandle domain) {
    if (domain != nullptr) {
      nvtxDomainDestroy(domain);
    }
  }

  // --- Instantaneous Markers ---
  inline void mark(const char* message, DomainHandle domain = DefaultDomain) {
    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;
    nvtxDomainMarkEx(domain, &eventAttrib);
  }

  inline void markColor(const char* message, uint32_t color, DomainHandle domain = DefaultDomain) {
    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;
    nvtxDomainMarkEx(domain, &eventAttrib);
  }

  // --- Explicit Start / End Ranges ---
  [[nodiscard]] inline RangeId rangeStart(const char* message, DomainHandle domain = DefaultDomain) {
    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;
    return nvtxDomainRangeStartEx(domain, &eventAttrib);
  }

  [[nodiscard]] inline RangeId rangeStartColor(const char* message,
                                               uint32_t color,
                                               DomainHandle domain = DefaultDomain) {
    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;
    return nvtxDomainRangeStartEx(domain, &eventAttrib);
  }

  inline void rangeEnd(RangeId id, DomainHandle domain = DefaultDomain) {
    if (id != InvalidRangeId) {
      nvtxDomainRangeEnd(domain, id);
    }
  }

  // --- Push / Pop Ranges ---
  inline void rangePush(const char* message, DomainHandle domain = DefaultDomain) {
    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;
    nvtxDomainRangePushEx(domain, &eventAttrib);
  }

  inline void rangePushColor(const char* message, uint32_t color, DomainHandle domain = DefaultDomain) {
    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;
    nvtxDomainRangePushEx(domain, &eventAttrib);
  }

  inline void rangePop(DomainHandle domain = DefaultDomain) { nvtxDomainRangePop(domain); }

  // --- Scoped RAII Push/Pop Range ---
  class ScopedRange {
  public:
    // Without explicit domain
    explicit ScopedRange(const char* name) : domain_(DefaultDomain) { rangePush(name, domain_); }
    ScopedRange(const char* name, uint32_t color) : domain_(DefaultDomain) { rangePushColor(name, color, domain_); }

    // With explicit domain
    ScopedRange(DomainHandle domain, const char* name) : domain_(domain) { rangePush(name, domain_); }
    ScopedRange(DomainHandle domain, const char* name, uint32_t color) : domain_(domain) {
      rangePushColor(name, color, domain_);
    }

    ~ScopedRange() { rangePop(domain_); }

    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
    ScopedRange(ScopedRange&&) = delete;
    ScopedRange& operator=(ScopedRange&&) = delete;

  private:
    DomainHandle domain_;
  };

}  // namespace nvtx_tools

#else  // Fallback / Dummy when Alpaka CUDA GPU is disabled
namespace nvtx_tools {

  namespace colors {
    inline constexpr uint32_t Black = 0;
    inline constexpr uint32_t Red = 0;
    inline constexpr uint32_t DarkGreen = 0;
    inline constexpr uint32_t Green = 0;
    inline constexpr uint32_t LightGreen = 0;
    inline constexpr uint32_t Blue = 0;
    inline constexpr uint32_t Amber = 0;
    inline constexpr uint32_t LightAmber = 0;
    inline constexpr uint32_t White = 0;
  }  // namespace colors

  using DomainHandle = void*;
  using RangeId = uint64_t;
  inline constexpr RangeId InvalidRangeId = 0xfffffffffffffffful;
  inline constexpr DomainHandle DefaultDomain = nullptr;

  inline DomainHandle createDomain(const char*) { return nullptr; }
  inline void destroyDomain(DomainHandle) {}

  inline void mark(const char*, DomainHandle = DefaultDomain) {}
  inline void markColor(const char*, uint32_t, DomainHandle = DefaultDomain) {}

  [[nodiscard]] inline RangeId rangeStart(const char*, DomainHandle = DefaultDomain) { return InvalidRangeId; }
  [[nodiscard]] inline RangeId rangeStartColor(const char*, uint32_t, DomainHandle = DefaultDomain) {
    return InvalidRangeId;
  }
  inline void rangeEnd(RangeId, DomainHandle = DefaultDomain) {}

  inline void rangePush(const char*, DomainHandle = DefaultDomain) {}
  inline void rangePushColor(const char*, uint32_t, DomainHandle = DefaultDomain) {}
  inline void rangePop(DomainHandle = DefaultDomain) {}

  class ScopedRange {
  public:
    explicit ScopedRange(const char*) {}
    ScopedRange(const char*, uint32_t) {}
    ScopedRange(DomainHandle, const char*) {}
    ScopedRange(DomainHandle, const char*, uint32_t) {}
    ~ScopedRange() = default;
    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
  };

}  // namespace nvtx_tools

#endif