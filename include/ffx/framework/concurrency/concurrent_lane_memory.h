#pragma once

#include <format>
#include <memory>

#include "ffx/framework/utilities/token.h"

namespace ffx::framework::concurrency {

  class ConcurrentLaneMemory {
  public:
    template <typename T, typename... Args>
    void put(const std::size_t batch_id, Token<T> token, Args&&... args) {
      auto pointer = std::make_shared<T>(std::forward<Args>(args)...);
      memory_[batch_id][token.id()] = std::move(pointer);
    }

    template <typename T>
    void put(const std::size_t batch_id, Token<T> token, std::shared_ptr<const T> data) {
      memory_[batch_id][token.id()] = std::move(data);
    }

    template <typename T>
    std::shared_ptr<const T> get(const std::size_t batch_id, Token<T> token) const {
      const auto batch_iter = memory_.find(batch_id);
      if (batch_iter != memory_.end()) {
        const auto token_iter = batch_iter->second.find(token.id());
        if (token_iter != batch_iter->second.end()) {
          return std::static_pointer_cast<const T>(token_iter->second);
        }
      }
      throw std::runtime_error(std::format("Product not found for batch_id: {}, token: {}", batch_id, token.id()));
    }

    void erase(const std::size_t batch_id) { memory_.erase(batch_id); }

    void clear() noexcept { memory_.clear(); }

  private:
    using map_t = std::unordered_map<std::size_t, std::unordered_map<std::size_t, std::shared_ptr<const void>>>;

    map_t memory_;
  };

}  // namespace ffx::framework::concurrency
