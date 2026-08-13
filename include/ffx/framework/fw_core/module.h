#pragma once

#include <unordered_set>

#include "ffx/core/detail/concepts.h"
#include "ffx/framework/fw_core/context.h"
#include "ffx/framework/utilities/token.h"

namespace ffx::framework {

  // forward declaration
  template <concepts::queue TQueue>
  class Scheduler;

  template <concepts::queue TQueue>
  class Module {
  public:
    virtual ~Module() = default;
    virtual void init(TQueue& queue) {};
    virtual void dispatch(const Context<TQueue>& context) const = 0;

  protected:
    template <typename T>
    Token<T> consumes(const std::string_view name) {
      auto token = Token<T>{name};
      consume_.insert(token.id());
      return token;
    }

    template <typename T>
    Token<T> produces(const std::string_view name) {
      auto token = Token<T>{name};
      produce_.insert(token.id());
      return token;
    }

  private:
    friend class Scheduler<TQueue>;
    const std::unordered_set<std::size_t>& consume() const noexcept { return consume_; }
    const std::unordered_set<std::size_t>& produce() const noexcept { return produce_; }

    std::unordered_set<std::size_t> consume_;
    std::unordered_set<std::size_t> produce_;
  };

}  // namespace ffx::framework
