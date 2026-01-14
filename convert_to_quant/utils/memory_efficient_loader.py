"""
Unified safetensors loader with optional memory-efficient mode.

Provides a consistent interface for tensor loading regardless of mode.
"""
import gc
import mmap
import json
import struct
import torch
from safetensors import safe_open
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

class UnifiedSafetensorsLoader:
    """Unified safetensors loader supporting both preload and streaming modes.

    In standard mode (low_memory=False):
        - Loads all tensors upfront (fast, uses more RAM)
        - Tensors remain in memory until explicitly deleted

    In low-memory mode (low_memory=True):
        - Loads tensors on-demand via get_tensor()
        - Caller should delete tensors after processing

    In fast-load mode (fast_load=True):
        - Uses mmap and concurrent futures for parallel batch loading
        - Much faster for large files with many tensors

    Usage:
        with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
            for key in loader.keys():
                tensor = loader.get_tensor(key)
                # ... process tensor ...
                loader.mark_processed(key)  # Frees memory in low_memory mode
    """

    def __init__(self, filename: str, low_memory: bool = False, fast_load: bool = True):
        """Initialize the loader.

        Args:
            filename: Path to safetensors file
            low_memory: If True, use streaming mode; if False, preload all tensors
            fast_load: If True and not low_memory, use parallel mmap loading (default: True)
        """
        self.filename = filename
        self.low_memory = low_memory
        self.fast_load = fast_load and not low_memory  # fast_load only applies to preload mode
        self._tensors: Dict[str, torch.Tensor] = {}
        self._all_keys = []
        self._file = None
        self._header = None
        self._header_size = None

        if low_memory:
            # Streaming mode: read header only, keep file open
            self._header, self._header_size = self._read_header()
            self._file = open(filename, "rb")
            self._all_keys = [k for k in self._header.keys() if k != "__metadata__"]
            print(f"Low-memory mode: found {len(self._all_keys)} tensors (streaming)")
        elif self.fast_load:
            # Fast-load mode: use mmap and concurrent loading
            self._header, self._header_size = self._read_header()
            self._all_keys = [k for k in self._header.keys() if k != "__metadata__"]
            print(f"Fast-load mode: loading {len(self._all_keys)} tensors with mmap...")
            self._load_tensors_fast()
        else:
            # Standard mode: preload all tensors sequentially
            with safe_open(filename, framework="pt", device="cpu") as f:
                self._all_keys = list(f.keys())
                print(f"Loading {len(self._all_keys)} tensors from source file...")
                from tqdm import tqdm
                for key in tqdm(self._all_keys, desc="Loading tensors"):
                    self._tensors[key] = f.get_tensor(key)

    def _load_tensors_fast(self, max_workers: int = 8):
        """Load all tensors in parallel using mmap."""
        from tqdm import tqdm
        
        with open(self.filename, "rb") as f:
            # Memory-map the file for fast random access
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            def load_tensor(key: str) -> tuple:
                """Load a single tensor from mmap."""
                metadata = self._header[key]
                offset_start, offset_end = metadata["data_offsets"]
                data_offset = self._header_size + 8 + offset_start
                data_length = offset_end - offset_start
                
                if data_length > 0:
                    # Read directly from mmap
                    tensor_bytes = bytearray(mm[data_offset:data_offset + data_length])
                    tensor = self._deserialize_tensor(tensor_bytes, metadata)
                else:
                    tensor = self._deserialize_tensor(None, metadata)
                
                return key, tensor
            
            # Load tensors in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(load_tensor, key): key for key in self._all_keys}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Loading tensors"):
                    key, tensor = future.result()
                    self._tensors[key] = tensor
            
            mm.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close file handle and release resources."""
        if self._file:
            self._file.close()
            self._file = None
        self._tensors.clear()

    def keys(self):
        """Return list of all tensor keys."""
        return self._all_keys

    def get_shape(self, key: str) -> tuple:
        """Get tensor shape without loading tensor data.

        In low-memory mode, reads from header.
        In standard mode, returns shape from loaded tensor.
        """
        if self.low_memory:
            if key not in self._header:
                raise KeyError(f"Tensor '{key}' not found in file")
            return tuple(self._header[key]["shape"])
        else:
            return tuple(self._tensors[key].shape)

    def get_ndim(self, key: str) -> int:
        """Get tensor ndim without loading tensor data."""
        return len(self.get_shape(key))

    def get_tensor(self, key: str) -> torch.Tensor:
        """Get a tensor by key.

        In standard mode, returns from cache.
        In low-memory mode, loads from file on-demand.
        """
        if not self.low_memory:
            # Standard mode: return from preloaded cache
            return self._tensors[key]

        # Low-memory mode: load on-demand
        if key not in self._header:
            raise KeyError(f"Tensor '{key}' not found in file")

        metadata = self._header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start != offset_end:
            self._file.seek(self._header_size + 8 + offset_start)
            # Use bytearray to create a writable buffer, avoiding PyTorch warning
            # about non-writable tensors from read-only bytes.
            tensor_bytes = bytearray(offset_end - offset_start)
            self._file.readinto(tensor_bytes)
        else:
            tensor_bytes = None

        return self._deserialize_tensor(tensor_bytes, metadata)

    def mark_processed(self, key: str):
        """Mark a tensor as processed, freeing memory if in low-memory mode.

        In standard mode, optionally deletes from cache.
        In low-memory mode, this is a no-op (tensor was never cached).
        """
        if not self.low_memory and key in self._tensors:
            del self._tensors[key]
            gc.collect()

    def _read_header(self):
        """Read and parse the safetensors header."""
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata) -> torch.Tensor:
        """Deserialize raw bytes into a torch tensor."""
        dtype_str = metadata["dtype"]
        shape = metadata["shape"]
        dtype = self._get_torch_dtype(dtype_str)

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        if dtype_str in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, dtype_str, shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str: str) -> torch.dtype:
        """Map safetensors dtype string to torch dtype."""
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn

        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype

    @staticmethod
    def _convert_float8(byte_tensor: torch.Tensor, dtype_str: str, shape: list) -> torch.Tensor:
        """Convert bytes to float8 tensor."""
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str}")


# Backward compatibility alias
MemoryEfficientSafeOpen = UnifiedSafetensorsLoader
