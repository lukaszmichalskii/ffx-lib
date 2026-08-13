#pragma once

#include <cxxabi.h>
#include <cctype>
#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ffw::framework {

  namespace detail {

    // Safely replace occurrences without risk of infinite loops
    inline void replace_all(std::string& str, std::string_view from, std::string_view to) {
      if (from.empty())
        return;
      std::string::size_type pos = 0;
      while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
      }
    }

    // Remove template arguments like allocator/less cleanly
    inline void remove_template_parameter(std::string& name, std::string_view to_remove) {
      std::string::size_type index = 0;
      while ((index = name.find(to_remove)) != std::string::npos) {
        int depth = 1;
        std::string::size_type inx = index + to_remove.size();

        while (inx < name.size()) {
          if (name[inx] == '<') {
            ++depth;
          } else if (name[inx] == '>') {
            --depth;
            if (depth == 0) {
              name.erase(index, inx + 1 - index);
              if (index < name.size() && name[index] == ' ' && (index == 0 || name[index - 1] != '>')) {
                name.erase(index, 1);
              }
              break;
            }
          }
          ++inx;
        }
        if (depth != 0)
          break;
      }
    }

    // Normalize 'const' position
    inline void normalize_const_position(std::string& name) {
      constexpr std::string_view target = " const";
      std::string::size_type index = 0;

      while ((index = name.find(target)) != std::string::npos) {
        name.erase(index, target.size());
        int depth = 0;

        for (std::string::size_type inx = index; inx > 0; --inx) {
          const char c = name[inx - 1];
          if (c == '>') {
            ++depth;
          } else if (depth > 0) {
            if (c == '<')
              --depth;
          } else if (c == '<' || c == ',') {
            name.insert(inx, "const ");
            break;
          }
        }
      }
    }

    // Strip trailing 'u', 'l', 'UL', 'ULL', etc. before a delimiter (',', '>')
    inline void strip_integer_literal_suffixes(std::string& name) {
      for (std::size_t i = 0; i < name.size(); ++i) {
        if (std::isdigit(static_cast<unsigned char>(name[i]))) {
          std::size_t pos_end = i + 1;
          while (pos_end < name.size() && std::isdigit(static_cast<unsigned char>(name[pos_end]))) {
            ++pos_end;
          }

          std::size_t suffix_start = pos_end;
          while (pos_end < name.size() &&
                 (name[pos_end] == 'u' || name[pos_end] == 'U' || name[pos_end] == 'l' || name[pos_end] == 'L')) {
            ++pos_end;
          }

          if (suffix_start < pos_end && (pos_end == name.size() || name[pos_end] == ',' || name[pos_end] == '>')) {
            name.erase(suffix_start, pos_end - suffix_start);
          }
          i = pos_end;
        }
      }
    }

  }  // namespace detail

  inline std::string type_demangle(char const* mangled_name) {
    if (!mangled_name)
      return {};

    int status = 0;
    std::unique_ptr<char, void (*)(void*)> demangled_ptr(abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status),
                                                         std::free);

    if (status != 0 || !demangled_ptr) {
      throw std::runtime_error(std::format("Demangling error for symbol: '{}'", mangled_name));
    }

    std::string name(demangled_ptr.get());

    detail::replace_all(name, ", ", ",");
    detail::replace_all(name, " [", "[");
    detail::replace_all(name, "std::__1::", "std::");
    detail::replace_all(name, "std::__cxx11::", "std::");

    detail::remove_template_parameter(name, ",std::allocator<");
    detail::remove_template_parameter(name, ",std::less<");

    detail::normalize_const_position(name);
    detail::replace_all(name, ">>", "> >");

    detail::strip_integer_literal_suffixes(name);

    return name;
  }

}  // namespace ffw::framework