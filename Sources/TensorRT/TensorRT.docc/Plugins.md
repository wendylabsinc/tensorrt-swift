# Plugins

Many real-world TensorRT plans rely on plugins (either built-in plugin layers or custom `.so`s).

If deserialization fails for an engine that works elsewhere, plugin registration/loading is a common
cause.

## Built-in plugins

Call ``TensorRTSystem/initializePlugins()`` once at process start (or before deserializing/building):

```swift
import TensorRT

try TensorRTSystem.initializePlugins()
```

## Custom plugins (.so)

If you have custom plugin libraries, load them before deserializing:

```swift
import TensorRT

try TensorRTSystem.loadPluginLibrary("/path/to/libmyplugins.so")
```

You can also configure plugin initialization/loading during deserialization using
``EngineLoadConfiguration``.
