import React, { useContext, useEffect, useRef, useState } from "react";
import { ReactNode } from "react";
import { useAppState } from "../../001_provider/001_AppStateProvider";
import { StateControlCheckbox, useStateControlCheckbox } from "../../hooks/useStateControlCheckbox";

export const OpenServerControlCheckbox = "open-server-control-checkbox";
export const OpenModelSettingCheckbox = "open-model-setting-checkbox";
export const OpenDeviceSettingCheckbox = "open-device-setting-checkbox";
export const OpenQualityControlCheckbox = "open-quality-control-checkbox";
export const OpenSpeakerSettingCheckbox = "open-speaker-setting-checkbox";
export const OpenConverterSettingCheckbox = "open-converter-setting-checkbox";
export const OpenAdvancedSettingCheckbox = "open-advanced-setting-checkbox";
export const OpenLabCheckbox = "open-lab-checkbox";

export const OpenLicenseDialogCheckbox = "open-license-dialog-checkbox";
export const OpenWaitingDialogCheckbox = "open-waiting-dialog-checkbox";
export const OpenStartingNoticeDialogCheckbox = "open-starting-notice-dialog-checkbox";
export const OpenModelSlotManagerDialogCheckbox = "open-model-slot-manager-dialog-checkbox";
export const OpenMergeLabDialogCheckbox = "open-merge-lab-dialog-checkbox";
export const OpenAdvancedSettingDialogCheckbox = "open-advanced-setting-dialog-checkbox";
export const OpenGetServerInformationDialogCheckbox = "open-get-server-information-dialog-checkbox";
export const OpenGetClientInformationDialogCheckbox = "open-get-client-information-dialog-checkbox";
export const OpenEnablePassThroughDialogCheckbox = "open-enable-pass-through-dialog-checkbox";

export const OpenTextInputDialogCheckbox = "open-text-input-dialog-checkbox";
export const OpenShowLicenseDialogCheckbox = "open-show-license-dialog-checkbox";

const AUDIO_INPUT_STORAGE_KEY = "kittycrow.voice-changer.audio-input.v1";
const AUDIO_OUTPUT_STORAGE_KEY = "kittycrow.voice-changer.audio-output.v1";
const AUDIO_MONITOR_STORAGE_KEY = "kittycrow.voice-changer.audio-monitor.v1";

const readStoredString = (key: string, fallback: string) => {
    try {
        return window.localStorage.getItem(key) || fallback;
    } catch (e) {
        console.warn(`[voice-changer] failed to read ${key}`, e);
        return fallback;
    }
};

const writeStoredString = (key: string, value: string) => {
    try {
        window.localStorage.setItem(key, value);
    } catch (e) {
        console.warn(`[voice-changer] failed to persist ${key}`, e);
    }
};

type Props = {
    children: ReactNode;
};

export type StateControls = {
    openServerControlCheckbox: StateControlCheckbox;
    openModelSettingCheckbox: StateControlCheckbox;
    openDeviceSettingCheckbox: StateControlCheckbox;
    openQualityControlCheckbox: StateControlCheckbox;
    openSpeakerSettingCheckbox: StateControlCheckbox;
    openConverterSettingCheckbox: StateControlCheckbox;
    openAdvancedSettingCheckbox: StateControlCheckbox;
    openLabCheckbox: StateControlCheckbox;

    showWaitingCheckbox: StateControlCheckbox;
    showStartingNoticeCheckbox: StateControlCheckbox;
    showModelSlotManagerCheckbox: StateControlCheckbox;

    showMergeLabCheckbox: StateControlCheckbox;
    showAdvancedSettingCheckbox: StateControlCheckbox;
    showGetServerInformationCheckbox: StateControlCheckbox;
    showGetClientInformationCheckbox: StateControlCheckbox;
    showEnablePassThroughDialogCheckbox: StateControlCheckbox;
    showTextInputCheckbox: StateControlCheckbox;
    showLicenseCheckbox: StateControlCheckbox;
};

type GuiStateAndMethod = {
    stateControls: StateControls;
    isConverting: boolean;
    isAnalyzing: boolean;
    showPyTorchModelUpload: boolean;
    setIsConverting: (val: boolean) => void;
    setIsAnalyzing: (val: boolean) => void;
    setShowPyTorchModelUpload: (val: boolean) => void;

    inputAudioDeviceInfo: MediaDeviceInfo[];
    outputAudioDeviceInfo: MediaDeviceInfo[];
    audioInputForGUI: string;
    audioOutputForGUI: string;
    audioMonitorForGUI: string;
    fileInputEchoback: boolean | undefined;
    shareScreenEnabled: boolean;
    audioOutputForAnalyzer: string;
    setInputAudioDeviceInfo: (val: MediaDeviceInfo[]) => void;
    setOutputAudioDeviceInfo: (val: MediaDeviceInfo[]) => void;
    setAudioInputForGUI: (val: string) => void;
    setAudioOutputForGUI: (val: string) => void;
    setAudioMonitorForGUI: (val: string) => void;
    setFileInputEchoback: (val: boolean) => void;
    setShareScreenEnabled: (val: boolean) => void;
    setAudioOutputForAnalyzer: (val: string) => void;

    modelSlotNum: number;
    setModelSlotNum: (val: number) => void;

    textInputResolve: TextInputResolveType | null;
    setTextInputResolve: (val: TextInputResolveType | null) => void;
};

const GuiStateContext = React.createContext<GuiStateAndMethod | null>(null);
export const useGuiState = (): GuiStateAndMethod => {
    const state = useContext(GuiStateContext);
    if (!state) {
        throw new Error("useGuiState must be used within GuiStateProvider");
    }
    return state;
};

type TextInputResolveType = {
    resolve: ((value: string | PromiseLike<string>) => void) | null;
};

export const GuiStateProvider = ({ children }: Props) => {
    const { initialized, setting, setVoiceChangerClientSetting } = useAppState();
    const [isConverting, setIsConverting] = useState<boolean>(false);
    const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
    const [modelSlotNum, setModelSlotNum] = useState<number>(0);

    const [showPyTorchModelUpload, setShowPyTorchModelUpload] = useState<boolean>(false);

    const storedInputRef = useRef<string>(readStoredString(AUDIO_INPUT_STORAGE_KEY, "none"));
    const storedOutputRef = useRef<string>(readStoredString(AUDIO_OUTPUT_STORAGE_KEY, "none"));
    const storedMonitorRef = useRef<string>(readStoredString(AUDIO_MONITOR_STORAGE_KEY, "none"));
    const inputPreferenceHydratedRef = useRef<boolean>(false);
    const outputPreferenceHydratedRef = useRef<boolean>(false);

    const [inputAudioDeviceInfo, setInputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([]);
    const [outputAudioDeviceInfo, setOutputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([]);
    const [audioInputForGUI, setAudioInputForGUI] = useState<string>(storedInputRef.current);
    const [audioOutputForGUI, setAudioOutputForGUI] = useState<string>(storedOutputRef.current);
    const [audioMonitorForGUI, setAudioMonitorForGUI] = useState<string>(storedMonitorRef.current);
    const [fileInputEchoback, setFileInputEchoback] = useState<boolean>(false); //最初のmuteが有効になるように。undefined <-- ??? falseしておけばよさそう。undefinedだとwarningがでる。
    const [shareScreenEnabled, setShareScreenEnabled] = useState<boolean>(false);
    const [audioOutputForAnalyzer, setAudioOutputForAnalyzer] = useState<string>("default");

    const [textInputResolve, setTextInputResolve] = useState<TextInputResolveType | null>(null);

    const reloadDeviceInfo = async () => {
        try {
            const ms = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
            ms.getTracks().forEach((x) => {
                x.stop();
            });
        } catch (e) {
            console.warn("Enumerate device error::", e);
        }
        const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();

        const audioInputs = mediaDeviceInfos.filter((x) => {
            return x.kind == "audioinput";
        });
        audioInputs.push({
            deviceId: "none",
            groupId: "none",
            kind: "audioinput",
            label: "none",
            toJSON: () => {},
        });
        audioInputs.push({
            deviceId: "file",
            groupId: "file",
            kind: "audioinput",
            label: "file",
            toJSON: () => {},
        });
        audioInputs.push({
            deviceId: "screen",
            groupId: "screen",
            kind: "audioinput",
            label: "system(only win)",
            toJSON: () => {},
        });
        const audioOutputs = mediaDeviceInfos.filter((x) => {
            return x.kind == "audiooutput";
        });
        audioOutputs.push({
            deviceId: "none",
            groupId: "none",
            kind: "audiooutput",
            label: "none",
            toJSON: () => {},
        });
        return [audioInputs, audioOutputs];
    };
    useEffect(() => {
        const audioInitialize = async () => {
            const audioInfo = await reloadDeviceInfo();
            setInputAudioDeviceInfo(audioInfo[0]);
            setOutputAudioDeviceInfo(audioInfo[1]);
        };
        audioInitialize();
    }, []);

    // LocalStorage is the durable preference source for browser devices. The
    // historical client also uses IndexedDB internally, but these preferences
    // should survive cleanly and synchronously across ordinary page sessions.
    useEffect(() => {
        if (!initialized || inputAudioDeviceInfo.length === 0 || inputPreferenceHydratedRef.current) {
            return;
        }

        const stored = storedInputRef.current;
        const storedIsAvailable = inputAudioDeviceInfo.some((device) => device.deviceId === stored);
        const clientInput = typeof setting.voiceChangerClientSetting.audioInput === "string" ? setting.voiceChangerClientSetting.audioInput : "none";
        const clientInputIsAvailable = inputAudioDeviceInfo.some((device) => device.deviceId === clientInput);
        const preferred = storedIsAvailable ? stored : clientInputIsAvailable ? clientInput : "none";

        storedInputRef.current = preferred;
        setAudioInputForGUI(preferred);
        writeStoredString(AUDIO_INPUT_STORAGE_KEY, preferred);

        if (preferred !== clientInput && preferred !== "file" && preferred !== "screen") {
            setVoiceChangerClientSetting({ ...setting.voiceChangerClientSetting, audioInput: preferred });
        }

        inputPreferenceHydratedRef.current = true;
    }, [initialized, inputAudioDeviceInfo, setting.voiceChangerClientSetting.audioInput]);

    useEffect(() => {
        if (outputAudioDeviceInfo.length === 0 || outputPreferenceHydratedRef.current) {
            return;
        }

        const output = outputAudioDeviceInfo.some((device) => device.deviceId === storedOutputRef.current) ? storedOutputRef.current : "none";
        const monitor = outputAudioDeviceInfo.some((device) => device.deviceId === storedMonitorRef.current) ? storedMonitorRef.current : "none";

        storedOutputRef.current = output;
        storedMonitorRef.current = monitor;
        setAudioOutputForGUI(output);
        setAudioMonitorForGUI(monitor);
        writeStoredString(AUDIO_OUTPUT_STORAGE_KEY, output);
        writeStoredString(AUDIO_MONITOR_STORAGE_KEY, monitor);
        outputPreferenceHydratedRef.current = true;
    }, [outputAudioDeviceInfo]);

    useEffect(() => {
        if (!inputPreferenceHydratedRef.current) return;
        storedInputRef.current = audioInputForGUI;
        writeStoredString(AUDIO_INPUT_STORAGE_KEY, audioInputForGUI);
    }, [audioInputForGUI]);

    useEffect(() => {
        if (!outputPreferenceHydratedRef.current) return;
        storedOutputRef.current = audioOutputForGUI;
        writeStoredString(AUDIO_OUTPUT_STORAGE_KEY, audioOutputForGUI);
    }, [audioOutputForGUI]);

    useEffect(() => {
        if (!outputPreferenceHydratedRef.current) return;
        storedMonitorRef.current = audioMonitorForGUI;
        writeStoredString(AUDIO_MONITOR_STORAGE_KEY, audioMonitorForGUI);
    }, [audioMonitorForGUI]);

    // (1) Controller Switch
    const openServerControlCheckbox = useStateControlCheckbox(OpenServerControlCheckbox);
    const openModelSettingCheckbox = useStateControlCheckbox(OpenModelSettingCheckbox);
    const openDeviceSettingCheckbox = useStateControlCheckbox(OpenDeviceSettingCheckbox);
    const openQualityControlCheckbox = useStateControlCheckbox(OpenQualityControlCheckbox);
    const openSpeakerSettingCheckbox = useStateControlCheckbox(OpenSpeakerSettingCheckbox);
    const openConverterSettingCheckbox = useStateControlCheckbox(OpenConverterSettingCheckbox);
    const openAdvancedSettingCheckbox = useStateControlCheckbox(OpenAdvancedSettingCheckbox);
    const openLabCheckbox = useStateControlCheckbox(OpenLabCheckbox);

    const showWaitingCheckbox = useStateControlCheckbox(OpenWaitingDialogCheckbox);
    const showStartingNoticeCheckbox = useStateControlCheckbox(OpenStartingNoticeDialogCheckbox);
    const showModelSlotManagerCheckbox = useStateControlCheckbox(OpenModelSlotManagerDialogCheckbox);
    const showMergeLabCheckbox = useStateControlCheckbox(OpenMergeLabDialogCheckbox);
    const showAdvancedSettingCheckbox = useStateControlCheckbox(OpenAdvancedSettingDialogCheckbox);
    const showGetServerInformationCheckbox = useStateControlCheckbox(OpenGetServerInformationDialogCheckbox);
    const showGetClientInformationCheckbox = useStateControlCheckbox(OpenGetClientInformationDialogCheckbox);
    const showEnablePassThroughDialogCheckbox = useStateControlCheckbox(OpenEnablePassThroughDialogCheckbox);

    const showTextInputCheckbox = useStateControlCheckbox(OpenTextInputDialogCheckbox);
    const showLicenseCheckbox = useStateControlCheckbox(OpenShowLicenseDialogCheckbox);

    useEffect(() => {
        openServerControlCheckbox.updateState(true);
        openModelSettingCheckbox.updateState(false);
        openDeviceSettingCheckbox.updateState(true);
        openSpeakerSettingCheckbox.updateState(true);
        openConverterSettingCheckbox.updateState(true);
        openQualityControlCheckbox.updateState(false);
        openLabCheckbox.updateState(false);
        openAdvancedSettingCheckbox.updateState(false);

        showWaitingCheckbox.updateState(false);
        showStartingNoticeCheckbox.updateState(false);
        showModelSlotManagerCheckbox.updateState(false);
        showMergeLabCheckbox.updateState(false);
        showAdvancedSettingCheckbox.updateState(false);
        showGetServerInformationCheckbox.updateState(false);
        showGetClientInformationCheckbox.updateState(false);
        showEnablePassThroughDialogCheckbox.updateState(false);

        showTextInputCheckbox.updateState(false);
        showLicenseCheckbox.updateState(false);
    }, []);

    const providerValue = {
        stateControls: {
            openServerControlCheckbox,
            openModelSettingCheckbox,
            openDeviceSettingCheckbox,
            openQualityControlCheckbox,
            openSpeakerSettingCheckbox,
            openConverterSettingCheckbox,
            openAdvancedSettingCheckbox,
            openLabCheckbox,

            showWaitingCheckbox,
            showStartingNoticeCheckbox,
            showModelSlotManagerCheckbox,

            showMergeLabCheckbox,
            showAdvancedSettingCheckbox,
            showGetServerInformationCheckbox,
            showGetClientInformationCheckbox,
            showEnablePassThroughDialogCheckbox,

            showTextInputCheckbox,
            showLicenseCheckbox,
        },
        isConverting,
        setIsConverting,
        isAnalyzing,
        setIsAnalyzing,
        showPyTorchModelUpload,
        setShowPyTorchModelUpload,

        reloadDeviceInfo,
        inputAudioDeviceInfo,
        outputAudioDeviceInfo,
        audioInputForGUI,
        audioOutputForGUI,
        audioMonitorForGUI,
        fileInputEchoback,
        shareScreenEnabled,
        audioOutputForAnalyzer,
        setInputAudioDeviceInfo,
        setOutputAudioDeviceInfo,
        setAudioInputForGUI,
        setAudioOutputForGUI,
        setAudioMonitorForGUI,
        setFileInputEchoback,
        setShareScreenEnabled,
        setAudioOutputForAnalyzer,

        modelSlotNum,
        setModelSlotNum,

        textInputResolve,
        setTextInputResolve,
    };
    return <GuiStateContext.Provider value={providerValue}>{children}</GuiStateContext.Provider>;
};