from __future__ import annotations

import json
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv-ocr-blackwell" / "Scripts" / "python.exe"
DIR = ROOT / "_trt_rec_convbn_probe"
LABELS = ["pool428","conv429","bn430","swish431","conv432","bn433"]

CHILD = r'''
import json, sys
from pathlib import Path
import cv2, numpy as np, paddle
import screen_ocr

ROOT=Path.cwd()
MODE=sys.argv[1]
DIR=ROOT/"_trt_rec_convbn_probe"
DIR.mkdir(parents=True,exist_ok=True)
MODEL=ROOT/".paddlex"/"official_models"/"PP-OCRv6_small_rec"/"inference"
PMODEL=DIR/"probe_model"/"inference"
FMODEL=DIR/"fp16_probe"/"inference"
MEDIA=ROOT/"_ocr_existing_srt_real_video"/"existing.mp4"
LABELS=["pool428","conv429","bn430","swish431","conv432","bn433"]

screen_ocr._prepare_tensorrt_runtime()
screen_ocr._install_tensorrt_compat()
from paddlex.inference.models.text_recognition.processors import OCRReisizeNormImg,ToBatch

cap=cv2.VideoCapture(str(MEDIA))
cap.set(cv2.CAP_PROP_POS_MSEC,3000.0)
ok,frame=cap.read(); cap.release()
if not ok: raise RuntimeError("FRAME_READ_FAILED")
crop=frame[494:570,568:1534].copy()
batch=ToBatch()(OCRReisizeNormImg(rec_image_shape=[3,48,320])([crop]))[0]
print("FOCUS_INPUT="+json.dumps({"mode":MODE,"shape":list(batch.shape)}),flush=True)

def run_saved(prefix):
    paddle.enable_static()
    exe=paddle.static.Executor(paddle.CUDAPlace(0))
    with paddle.pir_utils.IrGuard():
        p,feeds,fetches=paddle.static.io.load_inference_model(str(prefix),executor=exe)
        out=exe.run(p,feed={feeds[0]:batch},fetch_list=fetches,return_numpy=True)
    out=list(out or [])
    if len(out)!=len(LABELS): raise RuntimeError(f"FOCUS_OUTPUT_COUNT:{len(out)}")
    print("FOCUS_STATIC_OUTPUTS="+json.dumps({"mode":MODE,"count":len(out)}),flush=True)
    return out

if MODE=="fp32":
    paddle.enable_static()
    exe=paddle.static.Executor(paddle.CUDAPlace(0))
    with paddle.pir_utils.IrGuard():
        p,_,_=paddle.static.io.load_inference_model(str(MODEL),executor=exe)
        ops=list(p.global_block().ops)
        idxs=[428,429,430,431,432,433]
        vals=[ops[i].results()[0] for i in idxs]
        print("FOCUS_BOUNDARIES="+json.dumps([
            {"label":lab,"op_index":i,"op":ops[i].name(),"shape":list(v.shape)}
            for lab,i,v in zip(LABELS,idxs,vals)
        ],ensure_ascii=False),flush=True)
        inputs=[r for op in p.global_block().ops if op.name() in {"pd_op.data","pd_op.feed"} for r in op.results()]
        PMODEL.parent.mkdir(parents=True,exist_ok=True)
        paddle.static.save_inference_model(str(PMODEL),inputs,vals,exe,program=p)
    outputs=run_saved(PMODEL)

elif MODE=="fp16":
    from paddle.tensorrt.export import Input,PrecisionMode,TensorRTConfig,convert
    cfg=TensorRTConfig(
        inputs=[Input(warmup_data=(batch,batch,batch))],
        save_model_dir=str(FMODEL),
        precision_mode=PrecisionMode.FP16,
        disable_logging=False,
    )
    FMODEL.parent.mkdir(parents=True,exist_ok=True)
    print("FOCUS_TRT_BUILD_START="+str(FMODEL),flush=True)
    convert(str(PMODEL),cfg)
    print("FOCUS_TRT_BUILD_DONE",flush=True)
    outputs=run_saved(FMODEL)
else:
    raise RuntimeError("UNKNOWN_MODE")

arrs={k:np.asarray(v) for k,v in zip(LABELS,outputs)}
np.savez_compressed(DIR/f"{MODE}.npz",**arrs)
print("FOCUS_RESULT="+json.dumps({
    "mode":MODE,
    "items":{k:{
        "shape":list(v.shape),
        "mean":float(np.mean(v.astype(np.float64))),
        "std":float(np.std(v.astype(np.float64))),
        "min":float(np.min(v)),
        "max":float(np.max(v)),
    } for k,v in arrs.items()}
},ensure_ascii=False),flush=True)
'''

def run(mode):
    p=subprocess.Popen([str(PY),"-u","-c",CHILD,mode],cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1)
    assert p.stdout is not None
    for line in p.stdout:
        print(f"[{mode}] {line}",end="",flush=True)
    return p.wait()

def compare():
    a=np.load(DIR/"fp32.npz"); b=np.load(DIR/"fp16.npz")
    rows={}; first=None
    for k in LABELS:
        x=np.asarray(a[k],dtype=np.float64); y=np.asarray(b[k],dtype=np.float64)
        d=x-y
        rmse=float(np.sqrt(np.mean(d*d)))
        std=float(np.std(x))
        nrmse=rmse/max(std,1e-12)
        row={"max_abs_diff":float(np.max(np.abs(d))),"mean_abs_diff":float(np.mean(np.abs(d))),"rmse":rmse,"fp32_std":std,"nrmse":nrmse,"material":bool(nrmse>=0.05)}
        rows[k]=row
        if row["material"] and first is None: first=k
    print("FOCUS_SUMMARY="+json.dumps({"first_material_divergence":first,"boundaries":rows},ensure_ascii=False,sort_keys=True),flush=True)

def main():
    DIR.mkdir(parents=True,exist_ok=True)
    if run("fp32")!=0:return 1
    if run("fp16")!=0:return 2
    compare(); return 0

if __name__=="__main__":
    raise SystemExit(main())
