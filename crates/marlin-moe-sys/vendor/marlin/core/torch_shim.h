// Minimal stand-in for the few torch headers / macros that the vendored
// Marlin sources reach for. Defined so the kernels can compile in our
// build without pulling libtorch.
#pragma once

#include <stdexcept>
#include <sstream>
#include <string>

namespace marlin_moe_impl_detail {

inline void append_to(std::ostringstream&) {}

template <typename T, typename... Rest>
inline void append_to(std::ostringstream& oss, const T& v, const Rest&... rest) {
  oss << v;
  append_to(oss, rest...);
}

template <typename... Args>
[[noreturn]] inline void fail(const char* file, int line, const char* expr, Args&&... args) {
  std::ostringstream oss;
  oss << "marlin-moe check failed at " << file << ":" << line << ": " << expr;
  if constexpr (sizeof...(args) > 0) {
    oss << " — ";
    append_to(oss, std::forward<Args>(args)...);
  }
  throw std::runtime_error(oss.str());
}

}  // namespace marlin_moe_impl_detail

#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...) \
  do { \
    if (!(cond)) { \
      ::marlin_moe_impl_detail::fail(__FILE__, __LINE__, #cond, ##__VA_ARGS__); \
    } \
  } while (0)
#endif

#ifndef STD_TORCH_CHECK
#define STD_TORCH_CHECK(cond, ...) TORCH_CHECK(cond, ##__VA_ARGS__)
#endif

#ifndef TORCH_LIBRARY_IMPL_EXPAND
// Strip away PyTorch op registration; we expose a plain extern "C" ABI instead.
#define TORCH_LIBRARY_IMPL_EXPAND(NS, DEV, m) \
  static inline void _marlin_moe_impl_unused_registration(int)
#endif
