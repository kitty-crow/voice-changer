import json
import math
import os
import sys
import traceback
from dataclasses import asdict
from typing import Union

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File, Form

from downloader.SampleDownloader import getSampleInfos
from restapi.mods.FileUploader import upload_file, concat_file_chunks
from voice_changer.VoiceChangerManager import VoiceChangerManager

from const import MODEL_DIR, UPLOAD_DIR
from voice_changer.utils.LoadModelParams import LoadModelParamFile, LoadModelParams


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def _sanitize_json(value):
    """Make historical runtime state safe for Starlette's strict JSON encoder."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    return value


def _json_response(value, status_code: int = 200):
    encoded = jsonable_encoder(value)
    return JSONResponse(content=_sanitize_json(encoded), status_code=status_code)


class MMVC_Rest_Fileuploader:
    def __init__(self, voiceChangerManager: VoiceChangerManager):
        self.voiceChangerManager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/info", self.get_info, methods=["GET"])
        self.router.add_api_route("/performance", self.get_performance, methods=["GET"])
        self.router.add_api_route("/upload_file", self.post_upload_file, methods=["POST"])
        self.router.add_api_route("/concat_uploaded_file", self.post_concat_uploaded_file, methods=["POST"])
        self.router.add_api_route("/update_settings", self.post_update_settings, methods=["POST"])
        self.router.add_api_route("/load_model", self.post_load_model, methods=["POST"])
        self.router.add_api_route("/onnx", self.get_onnx, methods=["GET"])
        self.router.add_api_route("/merge_model", self.post_merge_models, methods=["POST"])
        self.router.add_api_route("/update_model_default", self.post_update_model_default, methods=["POST"])
        self.router.add_api_route("/update_model_info", self.post_update_model_info, methods=["POST"])
        self.router.add_api_route("/upload_model_assets", self.post_upload_model_assets, methods=["POST"])

    def _get_core_info(self):
        """Build enough server state for the UI even if an optional info source fails."""
        manager = self.voiceChangerManager
        data = asdict(manager.settings)

        # These two fields are essential to the main UI. Re-read them rather
        # than allowing an unrelated optional status probe to hide them.
        gpus = manager.gpus
        if len(gpus) == 0:
            try:
                gpus = manager._get_gpuInfos()
            except Exception:
                print("[Voice Changer] GPU info fallback failed:")
                traceback.print_exc()
                gpus = []
        data["gpus"] = gpus
        data["modelSlots"] = manager.modelSlotManager.getAllSlotInfo(reload=True)

        try:
            data["sampleModels"] = getSampleInfos(manager.params.sample_mode)
        except Exception:
            print("[Voice Changer] sample catalogue info failed:")
            traceback.print_exc()
            data["sampleModels"] = []

        data["python"] = sys.version
        data["voiceChangerParams"] = asdict(manager.params)
        data["status"] = "OK"

        try:
            data.update(manager.serverDevice.get_info())
        except Exception:
            print("[Voice Changer] server audio info failed:")
            traceback.print_exc()
            data["serverAudioInputDevices"] = []
            data["serverAudioOutputDevices"] = []

        if manager.voiceChanger is not None:
            try:
                data.update(manager.voiceChanger.get_info())
            except Exception:
                # Active-model telemetry is useful but must never make the
                # complete server information endpoint unavailable.
                print("[Voice Changer] active voice changer info failed:")
                traceback.print_exc()

        return data

    def post_upload_file(self, file: UploadFile = File(...), filename: str = Form(...)):
        try:
            res = upload_file(UPLOAD_DIR, file, filename)
            return _json_response(res)
        except Exception as e:
            print("[Voice Changer] post_upload_file ex:", e)

    def post_concat_uploaded_file(self, filename: str = Form(...), filenameChunkNum: int = Form(...)):
        try:
            res = concat_file_chunks(UPLOAD_DIR, filename, filenameChunkNum, UPLOAD_DIR)
            return _json_response(res)
        except Exception as e:
            print("[Voice Changer] post_concat_uploaded_file ex:", e)

    def get_info(self):
        try:
            info = self.voiceChangerManager.get_info()
            return _json_response(info)
        except Exception:
            print("[Voice Changer] get_info failed; falling back to core server state:")
            traceback.print_exc()

        try:
            return _json_response(self._get_core_info())
        except Exception:
            print("[Voice Changer] core get_info fallback failed:")
            traceback.print_exc()
            return _json_response(
                {
                    "status": "ERROR",
                    "message": "Failed to build server information.",
                },
                status_code=500,
            )

    def get_performance(self):
        try:
            info = self.voiceChangerManager.get_performance()
            return _json_response(info)
        except Exception as e:
            print("[Voice Changer] get_performance ex:", e)

    def post_update_settings(self, key: str = Form(...), val: Union[int, str, float] = Form(...)):
        try:
            print("[Voice Changer] update configuration:", key, val)
            info = self.voiceChangerManager.update_settings(key, val)
            return _json_response(info)
        except Exception as e:
            print("[Voice Changer] post_update_settings ex:", e)
            traceback.print_exc()

    def post_load_model(
        self,
        slot: int = Form(...),
        isHalf: bool = Form(...),
        params: str = Form(...),
    ):
        try:
            paramDict = json.loads(params)
            print("paramDict", paramDict)
            loadModelparams = LoadModelParams(**paramDict)
            loadModelparams.files = [LoadModelParamFile(**x) for x in paramDict["files"]]

            info = self.voiceChangerManager.loadModel(loadModelparams)
            return _json_response(info)
        except Exception as e:
            print("[Voice Changer] post_load_model ex:", e)
            traceback.print_exc()

    def get_onnx(self):
        try:
            info = self.voiceChangerManager.export2onnx()
            return _json_response(info)
        except Exception as e:
            print("[Voice Changer] get_onnx ex:", e)

    def post_merge_models(self, request: str = Form(...)):
        try:
            print(request)
            info = self.voiceChangerManager.merge_models(request)
            return _json_response(info)
        except Exception as e:
            print("[Voice Changer] post_merge_models ex:", e)
            traceback.print_exc()

    def post_update_model_default(self):
        try:
            info = self.voiceChangerManager.update_model_default()
            return _json_response(info)
        except Exception as e:
            print("[Voice Changer] post_update_model_default ex:", e)
            traceback.print_exc()

    def post_update_model_info(self, newData: str = Form(...)):
        try:
            info = self.voiceChangerManager.update_model_info(newData)
            return _json_response(info)
        except Exception as e:
            print("[Voice Changer] post_update_model_info ex:", e)

    def post_upload_model_assets(self, params: str = Form(...)):
        try:
            info = self.voiceChangerManager.upload_model_assets(params)
            return _json_response(info)
        except Exception as e:
            print("[Voice Changer] post_update_model_info ex:", e)
