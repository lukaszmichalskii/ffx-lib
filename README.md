<h3><b>Foundations for Framework eXtensions (ffx)</b></h3>

**ffx** is a header-only single-source performance-portability library and dependency-less C++ machine learning inference code synthesizer. Designed to simplify high-performance heterogeneous parallel computing research and rapid prototyping, it bridges high-level model descriptions with zero-overhead native C++ execution. 

| Component        | Description                                                                                                                                                                  |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ffx`            | A header-only library for accelerator development based on [alpaka](https://github.com/alpaka-group/alpaka)                                                                  |
| `ffx::algorithm` | A component with common parallel algorithms e.g. sort, reduce, scan                                                                                                          |
| `ffx::nn`        | A runtime for [ffx-compiler](python/ffx/compiler) implementing common portable kernels and operators for machine learning inference                                          |
| `ffx::framework` | A shared runtime utilities, application abstractions, and common execution helpers for application development                                                               |
| `ffx-compiler`   | A python based ahead-of-time (AOT) compilation stack for graph lowering, IR optimization, and C++ code synthesis from ffx::nn into portable, heterogeneous ML inference code |

Copyright © 2026 Lukasz Michalski
