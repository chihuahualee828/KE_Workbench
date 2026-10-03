# EasyEdit precision on Apple Silicon

Editing compatibility code lives in `app/ml/editing/mps_compat.py`. The EasyEdit
checkout is not patched. CUDA and CPU operations retain their original precision.

Before calling EasyEdit on MPS, the workbench changes `mom2_dtype: float64` (or
`double`) to `float32`. Cache listing, preparation, import, and edit preflight use
the same effective precision and filenames. Existing FP64 cache files are not
rewritten or relabeled.

Cache lookup and upload matching use the expected filename, not the stored
matrix's dtype. When the effective precision is FP32, FP64-named caches are not
listed or selected, and uploads must have the expected FP32 filename. Archive
validation checks structure, sample count, and matrix shape, but not dtype.
The workbench does not automatically convert uploaded FP64 cache files.

MEMIT also requests FP64 internally using `.double()`, independently of its
hyperparameters. During EasyEdit calls only, a thread-local PyTorch
`TorchFunctionMode` converts these MPS casts to FP32. It also converts FP64 cached
tensors to FP32 before they are transferred to MPS. This mode leaves CPU-only
casts alone and is removed even if an edit fails. It changes no global torch
functions or EasyEdit source files.
The interception message is printed only after an FP64 cast or transfer is
successfully redirected, once per operation.

The model loader uses `dtype="auto"` on CPU to infer precision from the model
config/checkpoint. Before moving to MPS it converts only FP64 parameters and
buffers to FP32; FP16, BF16, FP32, and integer tensors retain their loaded dtype.
No separate safetensors scan or second checkpoint load is needed. Automatic dtype
selection follows Transformers' rules, rather than preserving every individual
tensor's on-disk dtype in a mixed-precision checkpoint.

Model passes use the loaded precision. Statistics requested as FP64, covariance
inversion of FP64 caches, and MEMIT's explicit FP64 matrix solve use FP32 on GPU.
Normal CPU storage/serialization of caches and patches is unchanged. The
workbench does not enable CPU kernel fallback; an
unsupported MPS operation reports an error unless the user independently enables
PyTorch fallback. FP32 may produce different edits from the original FP64
algorithm, particularly for ill-conditioned covariance matrices.

The integration was checked against the device-aware EasyEdit checkout at
revision `14cea82`. Its normal optional dependencies are still required to use the
full EasyEdit application. After updating EasyEdit, rerun the compatibility tests:

```bash
cd backend
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest tests/test_device.py tests/test_mps_integration.py tests/test_mps_easyedit_numerics.py -v
```

The native hardware tests use tiny local models without downloads. Numerical tests
load the checkout's actual function definitions in isolation to avoid unrelated
multimodal/training imports. They check ROME/R-ROME/MEMIT gradients, KN operations,
scoring, FP32 statistics, and the actual device/dtype of MEMIT solves. They do not
measure real-model editing quality or replace full EasyEdit application testing.
Loader tests cover FP16, BF16, FP32, and FP64 checkpoints with forward/backward
passes on MPS. They do not establish editing quality for every lower-precision model.
