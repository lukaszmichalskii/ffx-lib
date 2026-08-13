# PR: ffx::framework – Lane Architecture Enforcement & Bug Fix Pass

## Summary

This PR applies a comprehensive architectural review and targeted bug-fix pass
across `ring_buffer`, `thread_pool`, `pipeline`, `scheduler`, `queue_pool`,
`data_stream`, and `context`.  All changes are fully backward-compatible except
for the `Pipeline` constructor signature (see migration below).

---

## Bug Fixes

### 1 · `ring_buffer.h` — Wrong memory ordering on stop signal  *(correctness)*

**File:** `include/ffx/framework/concurrency/ring_buffer.h`

```diff
- if (stop_tok.stop_requested() || stopped_.load(std::memory_order_relaxed)) {
+ if (stop_tok.stop_requested() || stopped_.load(std::memory_order_acquire)) {
```

`unblock_waiters()` stores `stopped_ = true` with `release` semantics.  The
corresponding load in `pop_wait()` was `relaxed`, which carries no
synchronisation guarantee.  On weakly-ordered hardware (Arm, POWER) the
consumer could spin forever, never observing the store.  Changed to `acquire`
so the release/acquire pair is correctly formed.

---

### 2 · `thread_pool.h` — `rr_index` function-local static shared across all pools  *(correctness)*

**File:** `include/ffx/framework/concurrency/thread_pool.h`

```diff
  template <typename Fn>
  void enqueue(Fn&& fn) {
-   static std::atomic<std::size_t> rr_index{0};
-   enqueue_on(rr_index.fetch_add(1, std::memory_order_relaxed), ...);
+   enqueue_on(rr_index_.fetch_add(1, std::memory_order_relaxed), ...);
  }

+ std::atomic<std::size_t> rr_index_{0};  // member, not static
```

A function-local `static` is shared across every `ThreadPool` instance in the
process.  When two pools of different sizes coexist the counter wraps at the
wrong modulus, producing a skewed and non-repeatable lane distribution.
Promoted to a member variable.

Also removed the now-redundant `if (*task)` null-check in `worker_loop`: tasks
are always constructed as non-empty lambdas by `enqueue_on`; the defensive
guard obscured the design intent.

---

### 3 · `pipeline.h` — Six bugs  *(critical correctness + performance)*

**File:** `include/ffx/framework/fw_core/pipeline.h`

#### 3a — Lane collapse: default `number_of_queues = 1`  *(critical)*

```diff
- explicit Pipeline(const Device& device,
-                   std::size_t number_of_threads = hardware_concurrency(),
-                   std::size_t number_of_queues  = 1)
-     : thread_pool_(number_of_threads),
-       queue_pool_(device, number_of_queues),
-       schedulers_(number_of_queues) {}

+ explicit Pipeline(const Device& device,
+                   std::size_t num_lanes = hardware_concurrency())
+     : memory_pool_(), queue_pool_(device, num_lanes),
+       schedulers_(num_lanes), thread_pool_(num_lanes) {}
```

With the old defaults (`number_of_threads = N, number_of_queues = 1`) the
batch assignment `worker_id = batch_index % queue_pool_.size()` always
produced `worker_id = 0`.  All N worker threads and a single lane were
available but only lane 0 was ever used — zero parallelism.

The Lane architecture requires a strict 1:1 binding of Thread[i] ↔ Queue[i] ↔
Scheduler[i].  A single `num_lanes` parameter makes this invariant impossible
to violate at construction time.

#### 3b — Redundant pipeline rebuild on every `dispatch()` call  *(performance)*

```diff
+ void build() {
+   if (is_built_) return;
    ...Kahn's sort + module->init()...
+   is_built_ = true;
+ }
```

`build()` was called unconditionally at the start of every `dispatch()`
invocation, re-running the topological sort and `module->init()` for every
batch stream.  A guarded `is_built_` flag makes `build()` idempotent;
subsequent calls are no-ops unless `add_module()` invalidates the graph.

#### 3c — `std::forward` inside a `while` loop  *(UB)*

```diff
- while (auto batch = std::forward<TDataProvider>(data_provider).get()) {
+ while (auto batch = data_provider.get()) {
```

`std::forward` casts to an rvalue on every iteration.  After the first call,
`data_provider` is in a moved-from state; subsequent iterations are undefined
behaviour.  Plain lvalue call is correct for a streaming data source.

#### 3d — Wrong explicit template argument in string `dispatch` overload  *(compile error)*

```diff
- dispatch<T>(std::move(file_stream));   // T resolved as TDataProvider → concept fails
+ dispatch(std::move(file_stream));      // deduces TDataProvider = DataStream<T> ✓
```

#### 3e — Member destruction order: lifetime hazard  *(correctness)*

```diff
  private:
-   SharedMemory                   memory_pool_;
-   ThreadPool                     thread_pool_;   // destroyed 4th
-   QueuePool<TQueue>              queue_pool_;    // destroyed 3rd ← BUG
-   std::vector<Scheduler<TQueue>> schedulers_;   // destroyed 2nd ← BUG
+   SharedMemory                   memory_pool_;
+   QueuePool<TQueue>              queue_pool_;
+   std::vector<Scheduler<TQueue>> schedulers_;
+   bool                           is_built_{false};
+   ThreadPool                     thread_pool_;   // ← destroyed FIRST
```

Members are destroyed in reverse declaration order.  With the old order,
`queue_pool_` and `schedulers_` were freed before `~ThreadPool()` joined the
worker threads.  Workers accessing `queue_pool_.queue(lane_id)` or
`schedulers_[lane_id]` after those objects were destroyed was a use-after-free.
`thread_pool_` is now declared last so its destructor (which calls
`shutdown()` → `wait()` → `join()`) runs first, guaranteeing all workers have
exited before any referenced resource is freed.

#### 3f — Unused dead member  *(cleanup)*

```diff
- mutable std::vector<std::mutex> slot_mutexes_;
```

Declared but never referenced.  Removed.

---

### 4 · `scheduler.h` — Missing standard-library includes  *(compile correctness)*

**File:** `include/ffx/framework/fw_core/scheduler.h`

```diff
+ #include <format>
+ #include <queue>
+ #include <stdexcept>
+ #include <unordered_map>
+ #include <unordered_set>
```

`build_pipeline()` used `std::queue`, `std::unordered_map`,
`std::unordered_set`, `std::runtime_error`, and `std::format` but relied on
those being transitively included by `module.h`.  Transitive includes are not
guaranteed to be stable; explicit includes added.

---

### 5 · `queue_pool.h` — Missing `const` overload  *(API completeness)*

**File:** `include/ffx/framework/concurrency/queue_pool.h`

```diff
+ const TQueue& queue(std::size_t index) const noexcept {
+     return queue_pool_[index % queue_pool_.size()];
+ }
```

Without a `const` overload, `queue()` could not be called through a `const
QueuePool&` reference, preventing use in const member functions of dependents.
An empty-pool `assert` is also added to guard the modulo-by-zero case.

---

### 6 · `data_stream.h` — Wrong `madvise` hint constant  *(correctness)*

**File:** `include/ffx/framework/fw_core/data_stream.h`

```diff
- madvise(const_cast<uint8_t*>(byte_data_), size_, POSIX_MADV_SEQUENTIAL);
+ (void)madvise(const_cast<uint8_t*>(byte_data_), size_, MADV_SEQUENTIAL);
```

`POSIX_MADV_SEQUENTIAL` is the constant for `posix_madvise()`.  The Linux
`madvise()` syscall requires `MADV_SEQUENTIAL`.  While both happen to be `2`
on glibc today, mixing constants across APIs is undefined behaviour.  The
`(void)` cast silences the `-Wunused-result` warning that was masked before.

A move constructor was also added so `DataStream<T>` objects can be forwarded
into `Pipeline::dispatch()`.

---

### 7 · `context.h` — Clang false-positive `-Wdangling-field`  *(warning elimination)*

**File:** `include/ffx/framework/fw_core/context.h`

```diff
  struct Meta {
    const std::size_t batch_id;
    const std::size_t batch_size;
+   constexpr Meta(std::size_t bid, std::size_t bsz) noexcept
+       : batch_id(bid), batch_size(bsz) {}
  };
```

The HIP/clang compiler emitted `-Wdangling-field` when `Meta` was
aggregate-initialised in `Context`'s member-initialiser list, misidentifying
the `const std::size_t` fields as pointer-like captures of constructor
parameters.  Providing an explicit constructor makes the value-copy semantics
unambiguous to the analyser and eliminates the warning across all backends.

---

## Migration Guide

### `Pipeline` constructor

```cpp
// Before (old API, now removed):
Pipeline<Queue> pipeline(device, kNumberOfThreads, kNumberOfQueues);

// After (new API – num_lanes enforces the 1:1 lane invariant):
Pipeline<Queue> pipeline(device, kNumberOfThreads);
// kNumberOfThreads == kNumberOfQueues was always required by the lane model;
// the new API makes this explicit.
```

---

## Build Verification

All targets compile with **zero errors and zero warnings** under C++23:

| Backend | Compiler      | Status |
|---------|---------------|--------|
| Serial  | GCC-15        | ✅ clean |
| CUDA    | nvcc + GCC-15 | ✅ clean |
| HIP     | clang/ROCm    | ✅ clean |
