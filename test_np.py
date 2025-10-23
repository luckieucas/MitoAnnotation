import importlib.util, sys
spec = importlib.util.find_spec("numpy")
print("loader:", spec.loader)
print("origin:", spec.origin)  # namespace 时为 None
print("locations:", list(spec.submodule_search_locations or []))
print("sys.path[0:8]:", sys.path[:8])
