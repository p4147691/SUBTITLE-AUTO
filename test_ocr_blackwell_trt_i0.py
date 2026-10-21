from __future__ import annotations

import json
import queue
import subprocess
import textwrap
import threading
import time
from pathlib import Path

import screen_ocr

ROOT = Path(__file__).resolve().parent
BLACKWELL_PYTHON = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"

COMMON = r'''
import json
import traceback
import numpy as np
import screen_ocr
screen_ocr._prepare_tensorrt_runtime()
from paddle.pir.core import _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE
from paddle.tensorrt.util import TensorRTConstantManager
from paddle.tensorrt import converter as converter_module

events = []
source_meta = {}
current_constant = {"name": None}

def flags_of(value, dtype=None):
    try:
        arr = np.asarray(value, dtype=dtype)
        return {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "strides": list(arr.strides),
            "c_contiguous": bool(arr.flags["C_CONTIGUOUS"]),
            "owndata": bool(arr.flags["OWNDATA"]),
        }
    except Exception as exc:
        return {"inspect_error": f"{type(exc).__name__}:{exc}"}

def describe_weight(value):
    try:
        arr = np.asarray(value)
        flat = arr.reshape(-1)
        return {
            "python_type": f"{type(value).__module__}.{type(value).__name__}",
            "dtype": str(arr.dtype),
            "dtype_str": arr.dtype.str,
            "dtype_char": arr.dtype.char,
            "itemsize": int(arr.dtype.itemsize),
            "shape": list(arr.shape),
            "size": int(arr.size),
            "strides": list(arr.strides),
            "c_contiguous": bool(arr.flags["C_CONTIGUOUS"]),
            "owndata": bool(arr.flags["OWNDATA"]),
            "sample": [str(item) for item in flat[:8]],
        }
    except Exception as exc:
        return {
            "python_type": f"{type(value).__module__}.{type(value).__name__}",
            "inspect_error": f"{type(exc).__name__}:{exc}",
        }

def remember_source(name, value):
    dtype = getattr(value, "dtype", None)
    source_meta[name] = {
        "pir_dtype": str(dtype),
        "pir_dtype_name": str(getattr(dtype, "name", None)),
        "pir_shape": list(getattr(value, "shape", []) or []),
        "value_id": str(getattr(value, "id", None)),
    }

def emit(kind, **payload):
    print(
        "TRT_I0_DIAG=" + json.dumps({"kind": kind, **payload}, ensure_ascii=False, default=str),
        flush=True,
    )

_original_get_constant_value = TensorRTConstantManager.get_constant_value
def traced_get_constant_value(self, name):
    result = _original_get_constant_value(self, name)
    current_constant["name"] = name
    return result

TensorRTConstantManager.get_constant_value = traced_get_constant_value

_real_trt = converter_module.trt
class _TrtProxy:
    def __getattr__(self, name):
        return getattr(_real_trt, name)

    def Weights(self, value):
        name = current_constant.get("name")
        actual = describe_weight(value)
        try:
            return _real_trt.Weights(value)
        except Exception as exc:
            emit(
                "weights_failure",
                name=name,
                source=source_meta.get(name),
                actual=actual,
                error=f"{type(exc).__name__}:{exc}",
            )
            raise

converter_module.trt = _TrtProxy()
'''

BASELINE = COMMON + r'''
def traced(self, name, tensor_data, value):
    remember_source(name, value)
    out_dtype = np.dtype(_PADDLE_PIR_DTYPE_2_NUMPY_DTYPE[value.dtype])
    if out_dtype == np.dtype("float64"):
        out_dtype = np.dtype("float32")
    if out_dtype == np.dtype("int64"):
        out_dtype = np.dtype("int32")
    before = flags_of(tensor_data)
    try:
        constant_array = np.array(tensor_data, dtype=out_dtype)
    except Exception as exc:
        emit("set_constant_failed", name=name, target_dtype=str(out_dtype), before=before,
             error=f"{type(exc).__name__}:{exc}")
        raise
    after = flags_of(constant_array)
    item = {"name": name, "target_dtype": str(out_dtype), "before": before, "after": after}
    events.append(item)
    if len(events) > 24:
        del events[:-24]
    self.constant_dict.update({name: constant_array})

TensorRTConstantManager.set_constant_value = traced

from paddleocr import PaddleOCR
try:
    ocr = PaddleOCR(
        device="gpu:0",
        text_detection_model_name="PP-OCRv6_tiny_det",
        text_recognition_model_name="PP-OCRv6_small_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        use_tensorrt=True,
        precision="fp16",
    )
    emit("baseline_success", recent=events[-12:])
except Exception as exc:
    emit("baseline_failure", error=f"{type(exc).__name__}:{exc}", recent=events[-24:])
    traceback.print_exc()
    raise
'''

PATCHED = COMMON + r'''
def traced(self, name, tensor_data, value):
    remember_source(name, value)
    out_dtype = np.dtype(_PADDLE_PIR_DTYPE_2_NUMPY_DTYPE[value.dtype])
    if out_dtype == np.dtype("float64"):
        out_dtype = np.dtype("float32")
    if out_dtype == np.dtype("int64"):
        out_dtype = np.dtype("int32")
    before = flags_of(tensor_data)
    normalized_intc = False
    try:
        initial = np.array(tensor_data, dtype=out_dtype)
        if initial.dtype.num == np.dtype(np.intc).num:
            constant_array = np.empty(initial.shape, dtype=np.dtype("int32"))
            constant_array[...] = initial
            normalized_intc = True
        else:
            constant_array = np.ascontiguousarray(initial).copy(order="C")
    except Exception as exc:
        emit("patched_set_constant_failed", name=name, target_dtype=str(out_dtype), before=before,
             error=f"{type(exc).__name__}:{exc}")
        raise
    after = flags_of(constant_array)
    item = {
        "name": name,
        "target_dtype": str(out_dtype),
        "before": before,
        "after": after,
        "initial_dtype_char": initial.dtype.char,
        "initial_dtype_num": int(initial.dtype.num),
        "normalized_intc": normalized_intc,
    }
    events.append(item)
    if len(events) > 24:
        del events[:-24]
    self.constant_dict.update({name: constant_array})

TensorRTConstantManager.set_constant_value = traced

import inspect
from paddle.tensorrt.impls import manipulation as manipulation_module

def _safe_trt_tensor(value):
    try:
        dtype = str(value.dtype)
    except Exception as exc:
        dtype = f"<dtype_error:{type(exc).__name__}:{exc}>"
    try:
        shape = repr(value.shape)
    except Exception as exc:
        shape = f"<shape_error:{type(exc).__name__}:{exc}>"
    try:
        name = str(value.name)
    except Exception as exc:
        name = f"<name_error:{type(exc).__name__}:{exc}>"
    try:
        is_shape_tensor = bool(value.is_shape_tensor)
    except Exception as exc:
        is_shape_tensor = f"<shape_tensor_error:{type(exc).__name__}:{exc}>"
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "is_shape_tensor": is_shape_tensor,
    }

def _safe_paddle_source(source, depth=0):
    try:
        defining_op = source.get_defining_op()
        defining_name = defining_op.name() if defining_op is not None else None
    except Exception as exc:
        defining_op = None
        defining_name = f"<def_error:{type(exc).__name__}:{exc}>"
    try:
        dtype = source.dtype
        dtype_text = str(dtype)
        dtype_name = str(getattr(dtype, "name", None))
    except Exception as exc:
        dtype_text = f"<dtype_error:{type(exc).__name__}:{exc}>"
        dtype_name = None
    try:
        shape = repr(source.shape)
    except Exception as exc:
        shape = f"<shape_error:{type(exc).__name__}:{exc}>"
    children = []
    if defining_op is not None and depth < 1:
        try:
            for operand in defining_op.operands():
                children.append(_safe_paddle_source(operand.source(), depth + 1))
        except Exception as exc:
            children.append({"inspect_error": f"{type(exc).__name__}:{exc}"})
    return {
        "id": str(getattr(source, "id", None)),
        "dtype": dtype_text,
        "dtype_name": dtype_name,
        "shape": shape,
        "defining_op": defining_name,
        "children": children,
    }

def _safe_trt_list(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [_safe_trt_tensor(item) for item in value]

def _stack_context(frame):
    local = frame.f_locals
    paddle_op = local.get("paddle_op")
    operands = []
    if paddle_op is not None:
        try:
            for operand in paddle_op.operands():
                operands.append(_safe_paddle_source(operand.source()))
        except Exception as exc:
            operands.append({"inspect_error": f"{type(exc).__name__}:{exc}"})
    try:
        result_shape = repr(paddle_op.results()[0].shape) if paddle_op is not None else None
    except Exception as exc:
        result_shape = f"<result_shape_error:{type(exc).__name__}:{exc}>"
    try:
        result_dtype = str(paddle_op.results()[0].dtype) if paddle_op is not None else None
    except Exception as exc:
        result_dtype = f"<result_dtype_error:{type(exc).__name__}:{exc}>"
    return {
        "paddle_op": paddle_op.name() if paddle_op is not None else None,
        "axis": local.get("axis"),
        "index": local.get("i"),
        "input_rank": local.get("input_rank"),
        "output_rank": local.get("output_rank"),
        "result_shape": result_shape,
        "result_dtype": result_dtype,
        "operands": operands,
        "input_tensors": _safe_trt_list(local.get("input_tensors")),
        "reshaped_inputs": _safe_trt_list(local.get("inputs")),
        "shape_tensor": _safe_trt_tensor(local.get("shape_tensor")) if local.get("shape_tensor") is not None else None,
        "shape_tensor_vec": _safe_trt_list(local.get("shape_tensor_vec")),
        "after_shape_tensor": _safe_trt_tensor(local.get("after_shape_tensor")) if local.get("after_shape_tensor") is not None else None,
        "output_tensor": _safe_trt_tensor(local.get("output_tensor")) if local.get("output_tensor") is not None else None,
    }

_original_get_shape_tensor_element = manipulation_module.get_shape_tensor_element
def _diag_get_shape_tensor_element(network, x, index, is_scalar=False, name=None):
    frame = inspect.currentframe().f_back
    stack_call = frame is not None and frame.f_code.co_name == "stack_converter"
    input_before = _safe_trt_tensor(x)
    normalized_shape_int64 = False
    if stack_call and str(getattr(x, "dtype", "")) == "DataType.INT64":
        cast_name = [name[0], "stack_shape_int32"] if isinstance(name, list) and name else name
        x = manipulation_module.trt_cast(
            network, x, manipulation_module.trt.int32, name=cast_name
        )
        normalized_shape_int64 = True
    result = _original_get_shape_tensor_element(
        network, x, index, is_scalar=is_scalar, name=name
    )
    if stack_call:
        emit(
            "stack_shape_input",
            branch="get_shape_tensor_element",
            requested_index=index,
            helper_name=name,
            input_before=input_before,
            input_after=_safe_trt_tensor(x),
            output=_safe_trt_tensor(result),
            normalized_shape_int64=normalized_shape_int64,
            context=_stack_context(frame),
        )
    return result

_original_add_1d_constant_layer = manipulation_module.add_1D_constant_layer
def _diag_add_1d_constant_layer(
    network, data, dtype=np.int32, is_scalar=False, name=None
):
    frame = inspect.currentframe().f_back
    normalized_stack_inputs = []
    if frame is not None and frame.f_code.co_name == "stack_converter":
        local = frame.f_locals
        paddle_op = local.get("paddle_op")
        stack_inputs = local.get("inputs")
        try:
            result_dtype = str(paddle_op.results()[0].dtype)
        except Exception:
            result_dtype = None
        target_trt_dtype = None
        if result_dtype == "paddle.int64":
            target_trt_dtype = manipulation_module.trt.int64
        elif result_dtype == "paddle.int32":
            target_trt_dtype = manipulation_module.trt.int32
        if target_trt_dtype is not None and isinstance(stack_inputs, list):
            for idx, tensor in enumerate(stack_inputs):
                if tensor.dtype != target_trt_dtype:
                    before = _safe_trt_tensor(tensor)
                    tensor = manipulation_module.trt_cast(
                        network,
                        tensor,
                        target_trt_dtype,
                        name=[paddle_op.name(), f"stack_data_cast_{idx}"],
                    )
                    stack_inputs[idx] = tensor
                    normalized_stack_inputs.append(
                        {"index": idx, "before": before, "after": _safe_trt_tensor(tensor)}
                    )
    result = _original_add_1d_constant_layer(
        network, data, dtype=dtype, is_scalar=is_scalar, name=name
    )
    if frame is not None and frame.f_code.co_name == "stack_converter":
        emit(
            "stack_shape_input",
            branch="add_1D_constant_layer",
            data=str(data),
            requested_numpy_dtype=str(np.dtype(dtype)),
            requested_numpy_dtype_char=np.dtype(dtype).char,
            requested_numpy_dtype_num=int(np.dtype(dtype).num),
            helper_name=name,
            output=_safe_trt_tensor(result),
            normalized_stack_inputs=normalized_stack_inputs,
            context=_stack_context(frame),
        )
    return result

_original_resize_to_1d = manipulation_module.resize_to_1d
def _diag_resize_to_1d(network, shape_tensor, name=None):
    frame = inspect.currentframe().f_back
    if frame is not None and frame.f_code.co_name == "stack_converter":
        emit(
            "stack_resize_input",
            name=name,
            tensor=_safe_trt_tensor(shape_tensor),
            context=_stack_context(frame),
        )
    return _original_resize_to_1d(network, shape_tensor, name=name)

manipulation_module.get_shape_tensor_element = _diag_get_shape_tensor_element
manipulation_module.add_1D_constant_layer = _diag_add_1d_constant_layer
manipulation_module.resize_to_1d = _diag_resize_to_1d

from paddleocr._models import TextRecognition
try:
    ocr = TextRecognition(
        model_name="PP-OCRv6_small_rec",
        device="gpu:0",
        enable_mkldnn=False,
        use_tensorrt=True,
        precision="fp16",
    )
    frame = np.zeros((48, 320, 3), dtype=np.uint8)
    outputs = list(ocr.predict(frame) or [])
    emit(
        "patched_success",
        one_frame_ocr=True,
        output_count=len(outputs),
        recent=events[-12:],
    )
except Exception as exc:
    emit("patched_failure", error=f"{type(exc).__name__}:{exc}", recent=events[-24:])
    traceback.print_exc()
    raise
'''

def run_case(label: str, code: str):
    assert BLACKWELL_PYTHON.is_file(), BLACKWELL_PYTHON
    screen_ocr._prepare_tensorrt_runtime()
    proc = subprocess.Popen(
        [str(BLACKWELL_PYTHON), "-u", "-c", code],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        bufsize=1,
    )
    assert proc.stdout is not None
    messages: queue.Queue = queue.Queue()
    captured: list[str] = []

    def pump():
        try:
            for line in proc.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    deadline = time.monotonic() + 600.0
    stream_done = False
    while True:
        try:
            item = messages.get(timeout=1.0)
            if item is None:
                stream_done = True
            else:
                captured.append(item)
                print(f"[{label}] {item}", end="", flush=True)
        except queue.Empty:
            pass

        if proc.poll() is not None and stream_done:
            break
        if time.monotonic() >= deadline:
            proc.kill()
            proc.wait()
            raise subprocess.TimeoutExpired(
                proc.args, 600, output="".join(captured)
            )

    thread.join(timeout=2.0)
    output = "".join(captured)
    result = subprocess.CompletedProcess(
        proc.args, proc.returncode, stdout=output, stderr=""
    )
    print(f"=== {label} rc={result.returncode} ===", flush=True)
    return result

def test_blackwell_tensorrt_i0_baseline_and_contiguous_patch():
    patched = run_case("patched_stack_homogeneous_fp16", PATCHED)
    assert patched.returncode == 0, patched.stdout + patched.stderr
    assert '"kind": "patched_success"' in patched.stdout, patched.stdout
    assert '"one_frame_ocr": true' in patched.stdout, patched.stdout
    assert '"normalized_stack_inputs": [' in patched.stdout, patched.stdout


import unittest


class BlackwellTrtI0Tests(unittest.TestCase):
    def test_baseline_and_contiguous_patch(self):
        test_blackwell_tensorrt_i0_baseline_and_contiguous_patch()
