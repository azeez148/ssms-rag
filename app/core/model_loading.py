import threading

# transformers builds models on the "meta" device through a process-global
# patch, so two models loading at once in different threads (the startup
# warm-up and the first request, say) can leave one with no weights:
# "Cannot copy out of meta tensor" on first use, and the broken instance stays
# cached. Every local model load goes through this one lock.
MODEL_LOAD_LOCK = threading.RLock()
