#pragma once

#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ffx/core/detail/concepts.h"
#include "ffx/framework/fw_core/module.h"

namespace ffx::framework {

  template <concepts::queue TQueue>
  class Scheduler {
  public:
    using Device = alpaka::Dev<TQueue>;

    Scheduler() = default;

    void add_module(std::shared_ptr<Module<TQueue>> module) {
      modules_.push_back(std::move(module));
      is_pipeline_ready_ = false;
    }

    void build_pipeline(TQueue& queue) {
      if (modules_.empty())
        return;

      const std::size_t number_of_modules = modules_.size();

      // token.id() → index of the module that produces it
      std::unordered_map<std::size_t, std::size_t> token_producer;
      for (auto index = 0zu; index < number_of_modules; ++index) {
        for (auto token_id : modules_[index]->produce()) {
          token_producer.emplace(token_id, index);
        }
      }

      // directed acyclic graph (DAG) adjacency list and in-degree map.
      std::vector<std::unordered_set<std::size_t>> adjacency_list(number_of_modules);
      std::vector<std::size_t> in_degree(number_of_modules, 0);

      for (auto consumer_index = 0zu; consumer_index < number_of_modules; ++consumer_index) {
        for (auto consumer_token_id : modules_[consumer_index]->consume()) {
          if (auto it = token_producer.find(consumer_token_id); it != token_producer.end()) {
            const auto producer_index = it->second;
            // prevent self-loops and duplicate edges.
            if (producer_index != consumer_index && adjacency_list[producer_index].insert(consumer_index).second) {
              ++in_degree[consumer_index];
            }
          }
        }
      }

      // Kahn's algorithm for topological sort.
      std::queue<std::size_t> q;
      for (auto index = 0zu; index < number_of_modules; ++index) {
        if (in_degree[index] == 0)
          q.push(index);
      }

      topological_pipeline_.clear();
      topological_pipeline_.reserve(number_of_modules);

      while (!q.empty()) {
        auto module_index = q.front();
        q.pop();

        topological_pipeline_.push_back(modules_[module_index]);
        for (const auto neighbor_index : adjacency_list[module_index]) {
          if (--in_degree[neighbor_index] == 0)
            q.push(neighbor_index);
        }
      }

      if (topological_pipeline_.size() != number_of_modules)
        throw std::runtime_error("Cyclic dependency detected in scheduled module graph");

      for (auto& module : topological_pipeline_)
        module->init(queue);

      is_pipeline_ready_ = true;
    }

    void dispatch(const Context<TQueue>& context) {
      if (!is_pipeline_ready_)
        build_pipeline(context.queue());

      for (const auto& module : topological_pipeline_)
        module->dispatch(context);
    }

  private:
    bool is_pipeline_ready_{false};
    std::vector<std::shared_ptr<Module<TQueue>>> modules_;
    std::vector<std::shared_ptr<Module<TQueue>>> topological_pipeline_;
  };

}  // namespace ffx::framework
