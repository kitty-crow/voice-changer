import { useMemo, useRef, useState } from "react"
import { VoiceChangerServerSetting, ServerInfo, ServerSettingKey, OnnxExporterInfo, MergeModelRequest, VoiceChangerType, DefaultServerSetting, F0Detector } from "../const"
import { VoiceChangerClient } from "../VoiceChangerClient"

export const ModelAssetName = {
    iconFile: "iconFile"
} as const
export type ModelAssetName = typeof ModelAssetName[keyof typeof ModelAssetName]

export const ModelFileKind = {
    "mmvcv13Config": "mmvcv13Config",
    "mmvcv13Model": "mmvcv13Model",
    "mmvcv15Config": "mmvcv15Config",
    "mmvcv15Model": "mmvcv15Model",
    "mmvcv15Correspondence": "mmvcv15Correspondence",

    "soVitsSvc40Config": "soVitsSvc40Config",
    "soVitsSvc40Model": "soVitsSvc40Model",
    "soVitsSvc40Cluster": "soVitsSvc40Cluster",

    "rvcModel": "rvcModel",
    "rvcIndex": "rvcIndex",

    "ddspSvcModel": "ddspSvcModel",
    "ddspSvcModelConfig": "ddspSvcModelConfig",
    "ddspSvcDiffusion": "ddspSvcDiffusion",
    "ddspSvcDiffusionConfig": "ddspSvcDiffusionConfig",

    "diffusionSVCModel": "diffusionSVCModel",

    "beatriceModel": "beatriceModel",

} as const
export type ModelFileKind = typeof ModelFileKind[keyof typeof ModelFileKind]

export type ModelFile = {
    file: File,
    kind: ModelFileKind
    dir: string
}

export type ModelUploadSetting = {
    voiceChangerType: VoiceChangerType,
    slot: number
    isSampleMode: boolean
    sampleId: string | null

    files: ModelFile[]
    params: any
}
export type ModelFileForServer = Omit<ModelFile, "file"> & {
    name: string,
    kind: ModelFileKind
}
export type ModelUploadSettingForServer = Omit<ModelUploadSetting, "files"> & {
    files: ModelFileForServer[]
}

type AssetUploadSetting = {
    slot: number
    name: ModelAssetName
    file: string
}

export type UseServerSettingProps = {
    voiceChangerClient: VoiceChangerClient | null
}

export type ServerSettingState = {
    serverSetting: ServerInfo
    updateServerSettings: (setting: ServerInfo) => Promise<void>
    reloadServerInfo: () => Promise<void>;

    uploadModel: (setting: ModelUploadSetting) => Promise<void>
    uploadProgress: number
    isUploading: boolean

    getOnnx: () => Promise<OnnxExporterInfo>
    mergeModel: (request: MergeModelRequest) => Promise<ServerInfo>
    updateModelDefault: () => Promise<ServerInfo>
    updateModelInfo: (slot: number, key: string, val: string) => Promise<ServerInfo>
    uploadAssets: (slot: number, name: ModelAssetName, file: File) => Promise<void>
}

const SERVER_PREFERENCES_STORAGE_KEY = "kittycrow.voice-changer.server-preferences.v1"

type ServerPreferences = {
    modelSlotIndex?: number
    f0Detector?: F0Detector
    gpu?: number
    serverInputDeviceId?: number
    serverOutputDeviceId?: number
    serverMonitorDeviceId?: number
}

const readServerPreferences = (): ServerPreferences => {
    if (typeof window === "undefined") return {}
    try {
        const raw = window.localStorage.getItem(SERVER_PREFERENCES_STORAGE_KEY)
        if (!raw) return {}
        return JSON.parse(raw) as ServerPreferences
    } catch (e) {
        console.warn("[voice-changer] failed to read local server preferences", e)
        return {}
    }
}

const writeServerPreferences = (setting: ServerInfo) => {
    if (typeof window === "undefined") return
    try {
        const normalisedModelSlotIndex = setting.modelSlotIndex >= 0 ? setting.modelSlotIndex % 1000 : setting.modelSlotIndex
        const prefs: ServerPreferences = {
            modelSlotIndex: normalisedModelSlotIndex,
            f0Detector: setting.f0Detector,
            gpu: setting.gpu,
            serverInputDeviceId: setting.serverInputDeviceId,
            serverOutputDeviceId: setting.serverOutputDeviceId,
            serverMonitorDeviceId: setting.serverMonitorDeviceId,
        }
        window.localStorage.setItem(SERVER_PREFERENCES_STORAGE_KEY, JSON.stringify(prefs))
    } catch (e) {
        console.warn("[voice-changer] failed to persist local server preferences", e)
    }
}

const restoreServerPreferences = (setting: ServerInfo): ServerInfo => {
    const prefs = readServerPreferences()
    const restored = { ...setting }

    if (
        typeof prefs.modelSlotIndex === "number" &&
        prefs.modelSlotIndex >= 0 &&
        setting.modelSlots?.some((slot) => slot.slotIndex === prefs.modelSlotIndex && Boolean(slot.modelFile))
    ) {
        restored.modelSlotIndex = prefs.modelSlotIndex
    }

    if (typeof prefs.f0Detector === "string" && prefs.f0Detector.length > 0) {
        restored.f0Detector = prefs.f0Detector
    }

    if (
        typeof prefs.gpu === "number" &&
        (prefs.gpu === -1 || setting.gpus?.some((gpu) => gpu.id === prefs.gpu))
    ) {
        restored.gpu = prefs.gpu
    }

    if (
        typeof prefs.serverInputDeviceId === "number" &&
        (prefs.serverInputDeviceId === -1 || setting.serverAudioInputDevices?.some((device) => device.index === prefs.serverInputDeviceId))
    ) {
        restored.serverInputDeviceId = prefs.serverInputDeviceId
    }

    if (
        typeof prefs.serverOutputDeviceId === "number" &&
        (prefs.serverOutputDeviceId === -1 || setting.serverAudioOutputDevices?.some((device) => device.index === prefs.serverOutputDeviceId))
    ) {
        restored.serverOutputDeviceId = prefs.serverOutputDeviceId
    }

    if (
        typeof prefs.serverMonitorDeviceId === "number" &&
        (prefs.serverMonitorDeviceId === -1 || setting.serverAudioOutputDevices?.some((device) => device.index === prefs.serverMonitorDeviceId))
    ) {
        restored.serverMonitorDeviceId = prefs.serverMonitorDeviceId
    }

    return restored
}

const changedServerSettingKeys = (current: ServerInfo, next: ServerInfo) => {
    return Object.values(ServerSettingKey).filter((key) => {
        const k = key as keyof VoiceChangerServerSetting
        return current[k] != next[k]
    }) as (keyof VoiceChangerServerSetting)[]
}

export const useServerSetting = (props: UseServerSettingProps): ServerSettingState => {
    const [serverSetting, setServerSetting] = useState<ServerInfo>(DefaultServerSetting)
    const serverSettingRef = useRef<ServerInfo>(DefaultServerSetting)
    const updateSerialRef = useRef<Promise<void>>(Promise.resolve())
    const updateVersionRef = useRef<number>(0)
    const preferencesRestoredRef = useRef<boolean>(false)

    const setServerSettingState = (setting: ServerInfo) => {
        serverSettingRef.current = setting
        setServerSetting(setting)
    }

    //////////////
    // 設定
    /////////////
    const updateServerSettings = useMemo(() => {
        return async (setting: ServerInfo) => {
            if (!props.voiceChangerClient) return

            const current = serverSettingRef.current
            const changedKeys = changedServerSettingKeys(current, setting)
            if (changedKeys.length === 0) return

            // Controlled sliders/selects must react immediately. The historical
            // implementation waited for a full HTTP round trip before changing
            // React state, which made range inputs appear frozen and allowed
            // stale responses to snap controls backwards while dragging.
            const version = ++updateVersionRef.current
            setServerSettingState(setting)
            writeServerPreferences(setting)

            const operation = async () => {
                let response: ServerInfo | null = null
                try {
                    for (const k of changedKeys) {
                        response = await props.voiceChangerClient!.updateServerSettings(k, "" + setting[k])
                    }

                    // Updates are serialised, but the user may already have made
                    // a newer optimistic change while this request was in flight.
                    // Only the newest operation is allowed to reconcile the UI.
                    if (response && version === updateVersionRef.current) {
                        setServerSettingState(response)
                        writeServerPreferences(response)
                    }
                } catch (e) {
                    console.error("[voice-changer] failed to update server setting", e)
                    if (version === updateVersionRef.current) {
                        try {
                            const authoritative = await props.voiceChangerClient!.getServerSettings()
                            setServerSettingState(authoritative)
                            writeServerPreferences(authoritative)
                        } catch (reloadError) {
                            console.error("[voice-changer] failed to recover server settings", reloadError)
                        }
                    }
                    throw e
                }
            }

            const queued = updateSerialRef.current.then(operation, operation)
            updateSerialRef.current = queued.catch(() => undefined)
            await queued
        }
    }, [props.voiceChangerClient])



    //////////////
    // 操作
    /////////////
    const [uploadProgress, setUploadProgress] = useState<number>(0)
    const [isUploading, setIsUploading] = useState<boolean>(false)

    // (e) モデルアップロード
    const _uploadFile2 = useMemo(() => {
        return async (file: File, onprogress: (progress: number, end: boolean) => void, dir: string = "") => {
            if (!props.voiceChangerClient) return
            const num = await props.voiceChangerClient.uploadFile2(dir, file, onprogress)
            const res = await props.voiceChangerClient.concatUploadedFile(dir + file.name, num)
            console.log("uploaded", num, res)
        }
    }, [props.voiceChangerClient])

    // 新しいアップローダ
    const uploadModel = useMemo(() => {
        return async (setting: ModelUploadSetting) => {
            if (!props.voiceChangerClient) {
                return
            }

            setUploadProgress(0)
            setIsUploading(true)


            if (setting.isSampleMode == false) {
                const progRate = 1 / setting.files.length
                for (let i = 0; i < setting.files.length; i++) {
                    const progOffset = 100 * i * progRate
                    await _uploadFile2(setting.files[i].file, (progress: number, _end: boolean) => {
                        setUploadProgress(progress * progRate + progOffset)
                    }, setting.files[i].dir)
                }
            }
            const params: ModelUploadSettingForServer = {
                ...setting, files: setting.files.map((f) => { return { name: f.file.name, kind: f.kind, dir: f.dir } })
            }

            const loadPromise = props.voiceChangerClient.loadModel(
                0,
                false,
                JSON.stringify(params),
            )
            await loadPromise

            setUploadProgress(0)
            setIsUploading(false)
            reloadServerInfo()

        }
    }, [props.voiceChangerClient])

    const uploadAssets = useMemo(() => {
        return async (slot: number, name: ModelAssetName, file: File) => {
            if (!props.voiceChangerClient) return

            await _uploadFile2(file, (progress: number, _end: boolean) => {
                console.log(progress, _end)
            })
            const assetUploadSetting: AssetUploadSetting = {
                slot,
                name,
                file: file.name
            }
            await props.voiceChangerClient.uploadAssets(JSON.stringify(assetUploadSetting))
            reloadServerInfo()
        }
    }, [props.voiceChangerClient])



    const reloadServerInfo = useMemo(() => {
        return async () => {
            if (!props.voiceChangerClient) return

            const res = await props.voiceChangerClient.getServerSettings()

            if (!preferencesRestoredRef.current) {
                preferencesRestoredRef.current = true
                const restored = restoreServerPreferences(res)
                const changedKeys = changedServerSettingKeys(res, restored)

                if (changedKeys.length > 0) {
                    setServerSettingState(res)
                    await updateServerSettings(restored)
                    return
                }
            }

            setServerSettingState(res)
            writeServerPreferences(res)
        }
    }, [props.voiceChangerClient, updateServerSettings])


    const getOnnx = async () => {
        return props.voiceChangerClient!.getOnnx()
    }

    const mergeModel = async (request: MergeModelRequest) => {
        const serverInfo = await props.voiceChangerClient!.mergeModel(request)
        setServerSettingState(serverInfo)
        writeServerPreferences(serverInfo)
        return serverInfo
    }

    const updateModelDefault = async () => {
        const serverInfo = await props.voiceChangerClient!.updateModelDefault()
        setServerSettingState(serverInfo)
        writeServerPreferences(serverInfo)
        return serverInfo
    }
    const updateModelInfo = async (slot: number, key: string, val: string) => {
        const serverInfo = await props.voiceChangerClient!.updateModelInfo(slot, key, val)
        setServerSettingState(serverInfo)
        writeServerPreferences(serverInfo)
        return serverInfo
    }

    return {
        serverSetting,
        updateServerSettings,
        reloadServerInfo,

        uploadModel,
        uploadProgress,
        isUploading,
        getOnnx,
        mergeModel,
        updateModelDefault,
        updateModelInfo,
        uploadAssets
    }
}