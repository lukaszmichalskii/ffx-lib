#pragma once

namespace ffx::framework {

  // forward declaration
  class SharedMemory;

  class TokenTag {
  public:
    static constexpr std::size_t id(const std::string_view name) noexcept {
      std::size_t hash = 14695981039346656037ULL;
      for (const auto c : name) {
        hash ^= static_cast<std::size_t>(c);
        hash *= 1099511628211ULL;
      }
      return hash;
    }
  };

  template <typename T>
  class Token {
  public:
    using value_type = T;

    constexpr Token() = delete;
    constexpr explicit Token(const std::string_view tag) noexcept : id_(TokenTag::id(tag)){};

    [[nodiscard]] constexpr std::size_t id() const noexcept { return id_; }
    [[nodiscard]] constexpr bool valid() const noexcept { return id_ != 0; }

    constexpr auto operator<=>(const Token&) const = default;

  private:
    std::size_t id_{0};

    friend class SharedMemory;
  };

}  // namespace ffx::framework