import { ClientState } from "@dannadori/voice-changer-client-js";
import React, { useContext, useEffect, useMemo, useRef, useState } from "react";
import { ReactNode } from "react";
import { useVCClient } from "../001_globalHooks/001_useVCClient";
import { useAppRoot } from "./001_AppRootProvider";
import { useMessageBuilder } from "../hooks/useMessageBuilder";

type Props = {
    children: ReactNode;
};

type ServerInfo = ClientState["serverSetting"]["serverSetting"];
type ServerSettingState = ClientState["serverSetting"];

type AppStateValue = ClientState & {
    audioContext: AudioContext;
    initializedRef: React.MutableRefObject<boolean>;
};

const SERVER_PREFERENCES_STORAGE_KEY = "kittycrow.voice-changer.server-preferences.v1";

type StoredServerPreferences = Partial<Pick<ServerInfo, "modelSlotIndex" | "f0Detector" | "gpu" | "serverInputDeviceId" | "serverOutputDeviceId" | "serverMonitorDeviceId">>;

const readServerPreferences = (): StoredServerPreferences => {
    try {
        const raw = window.localStorage.getItem(SERVER_PREFERENCES_STORAGE_KEY);
        return raw ? (JSON.parse(raw) as StoredServerPreferences) : {};
    } catch (e) {
        console.warn("[voice-changer] failed to read local server preferences", e);
        return {};
    }
};

const writeServerPreferences = (setting: ServerInfo) => {
    try {
        const preferences: StoredServerPreferences = {
            modelSlotIndex: setting.modelSlotIndex >= 0 ? setting.modelSlotIndex % 1000 : setting.modelSlotIndex,
            f0Detector: setting.f0Detector,
            gpu: setting.gpu,
            serverInputDeviceId: setting.serverInputDeviceId,
            serverOutputDeviceId: setting.serverOutputDeviceId,
            serverMonitorDeviceId: setting.serverMonitorDeviceId,
        };
        window.localStorage.setItem(SERVER_PREFERENCES_STORAGE_KEY, JSON.stringify(preferences));
    } catch (e) {
        console.warn("[voice-changer] failed to persist local server preferences", e);
    }
};

const restoreServerPreferences = (server: ServerInfo): ServerInfo => {
    const saved = readServerPreferences();
    const restored = { ...server };

    if (typeof saved.modelSlotIndex === "number" && saved.modelSlotIndex >= 0 && server.modelSlots?.some((slot) => slot.slotIndex === saved.modelSlotIndex && Boolean((slot as any).modelFile))) {
        restored.modelSlotIndex = saved.modelSlotIndex;
    }
    if (typeof saved.f0Detector === "string" && saved.f0Detector.length > 0) {
        restored.f0Detector = saved.f0Detector;
    }
    if (typeof saved.gpu === "number" && (saved.gpu === -1 || server.gpus?.some((gpu) => gpu.id === saved.gpu))) {
        restored.gpu = saved.gpu;
    }
    if (typeof saved.serverInputDeviceId === "number" && (saved.serverInputDeviceId === -1 || server.serverAudioInputDevices?.some((device) => device.index === saved.serverInputDeviceId))) {
        restored.serverInputDeviceId = saved.serverInputDeviceId;
    }
    if (typeof saved.serverOutputDeviceId === "number" && (saved.serverOutputDeviceId === -1 || server.serverAudioOutputDevices?.some((device) => device.index === saved.serverOutputDeviceId))) {
        restored.serverOutputDeviceId = saved.serverOutputDeviceId;
    }
    if (typeof saved.serverMonitorDeviceId === "number" && (saved.serverMonitorDeviceId === -1 || server.serverAudioOutputDevices?.some((device) => device.index === saved.serverMonitorDeviceId))) {
        restored.serverMonitorDeviceId = saved.serverMonitorDeviceId;
    }

    return restored;
};

const serverPreferencesDiffer = (current: ServerInfo, next: ServerInfo) => {
    return current.modelSlotIndex !== next.modelSlotIndex || current.f0Detector !== next.f0Detector || current.gpu !== next.gpu || current.serverInputDeviceId !== next.serverInputDeviceId || current.serverOutputDeviceId !== next.serverOutputDeviceId || current.serverMonitorDeviceId !== next.serverMonitorDeviceId;
};

const AppStateContext = React.createContext<AppStateValue | null>(null);
export const useAppState = (): AppStateValue => {
    const state = useContext(AppStateContext);
    if (!state) {
        throw new Error("useAppState must be used within AppStateProvider");
    }
    return state;
};

export const AppStateProvider = ({ children }: Props) => {
    const appRoot = useAppRoot();
    const clientState = useVCClient({ audioContext: appRoot.audioContextState.audioContext });
    const messageBuilderState = useMessageBuilder();

    useEffect(() => {
        messageBuilderState.setMessage(__filename, "ioError", {
            ja: "エラーが頻発しています。対象としているフレームワークのモデルがロードされているか確認してください。",
            en: "Frequent errors occur. Please check if the model of the framework being targeted is loaded.",
        });
    }, []);

    const initializedRef = useRef<boolean>(false);
    useEffect(() => {
        if (clientState.clientState.initialized) {
            initializedRef.current = true;
            clientState.clientState.getInfo();
        }
    }, [clientState.clientState.initialized]);

    useEffect(() => {
        if (clientState.clientState.ioErrorCount > 100) {
            alert(messageBuilderState.getMessage(__filename, "ioError"));
            clientState.clientState.resetIoErrorCount();
        }
    }, [clientState.clientState.ioErrorCount]);

    const baseServerSetting = clientState.clientState.serverSetting;
    const [visibleServerInfo, setVisibleServerInfo] = useState<ServerInfo>(baseServerSetting.serverSetting);
    const pendingServerUpdatesRef = useRef<number>(0);
    const serialServerUpdateRef = useRef<Promise<void>>(Promise.resolve());
    const preferencesRestoredRef = useRef<boolean>(false);

    // Mirror authoritative server state whenever no optimistic update is in flight.
    useEffect(() => {
        if (pendingServerUpdatesRef.current === 0) {
            setVisibleServerInfo(baseServerSetting.serverSetting);
        }
    }, [baseServerSetting.serverSetting]);

    const updateServerSettings = useMemo(() => {
        return async (next: ServerInfo) => {
            // React range inputs are controlled. Update their visible value before
            // waiting on the backend, otherwise a drag is immediately forced back
            // to the previous server value and feels completely frozen.
            setVisibleServerInfo(next);
            writeServerPreferences(next);
            pendingServerUpdatesRef.current += 1;

            const operation = async () => {
                try {
                    await baseServerSetting.updateServerSettings(next);
                } finally {
                    pendingServerUpdatesRef.current -= 1;
                    if (pendingServerUpdatesRef.current === 0) {
                        await baseServerSetting.reloadServerInfo();
                    }
                }
            };

            const queued = serialServerUpdateRef.current.then(operation, operation);
            serialServerUpdateRef.current = queued.catch(() => undefined);
            await queued;
        };
    }, [baseServerSetting.updateServerSettings, baseServerSetting.reloadServerInfo]);

    // Restore durable browser preferences once the first real /info payload has
    // arrived. Validate hardware/model identifiers first so disconnected devices
    // never leave the UI stuck on an impossible selection.
    useEffect(() => {
        if (!clientState.clientState.initialized || preferencesRestoredRef.current || !baseServerSetting.serverSetting.modelSlots) {
            return;
        }

        preferencesRestoredRef.current = true;
        const restored = restoreServerPreferences(baseServerSetting.serverSetting);
        setVisibleServerInfo(restored);

        if (serverPreferencesDiffer(baseServerSetting.serverSetting, restored)) {
            updateServerSettings(restored).catch((e) => console.error("[voice-changer] failed to restore server preferences", e));
        } else {
            writeServerPreferences(restored);
        }
    }, [clientState.clientState.initialized, baseServerSetting.serverSetting.modelSlots, updateServerSettings]);

    const wrappedServerSetting = useMemo<ServerSettingState>(() => {
        return {
            ...baseServerSetting,
            serverSetting: visibleServerInfo,
            updateServerSettings,
        };
    }, [baseServerSetting, visibleServerInfo, updateServerSettings]);

    const providerValue: AppStateValue = {
        audioContext: appRoot.audioContextState.audioContext!,
        ...clientState.clientState,
        serverSetting: wrappedServerSetting,
        initializedRef,
    };

    return <AppStateContext.Provider value={providerValue}>{children}</AppStateContext.Provider>;
};
