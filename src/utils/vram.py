"""VRAM-Utilities (deprecated).

Die Funktionen aus diesem Modul werden nicht mehr benötigt:
- get_tensor_parallel_size: TP wird jetzt per-Modell automatisch
  vom Scheduler berechnet (Auto-TP basierend auf per_gpu_vram_gb).
- calculate_router_allocations: Ersetzt durch Per-GPU VRAM-Tracking
  im Scheduler (GPUState).
"""
