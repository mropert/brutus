# Docs

Shared by every work stream:

- [working_process.md](working_process.md): conventions, the verification loop, measurement conditions, tools and their
  pitfalls, GPU and Vulkan findings. Read it before starting any work.
- [benchmark.md](benchmark.md): the `--benchmark` scene and how to reproduce it in another engine.
- [util/README.md](../util/README.md): the measurement scripts.

One file per feature stream, with its state, plan, experiments and findings:

- [shadows.md](shadows.md): cascaded shadow maps, caster culling, filtering.
- [draw_performance.md](draw_performance.md): the draw pass (culling, LODs, draw commands and bins, overdraw).

A new stream gets its own file, following the same layout (done so far, current state, plan, experiments, findings).
