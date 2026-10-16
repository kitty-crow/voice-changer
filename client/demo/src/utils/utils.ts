export const trimfileName = (name: string, length: number) => {
    const trimmedName = name.replace(/^.*[\\\/]/, '')
    if (trimmedName.length > length) {
        return trimmedName.substring(0, length) + "..."
    } else {
        return trimmedName
    }
}

export const checkExtention = (filename: string, acceptExtentions: string[]) => {
    const ext = filename.split('.').pop();
    if (!ext) {
        return false
    }
    return acceptExtentions.includes(ext)
}

/**
 * Choose an audio output when the browser permits it. Safari can reject this
 * during page initialisation because no user gesture has taken place yet.
 * Audio output selection must not crash the entire application.
 */
export const trySetAudioSink = (audio: HTMLAudioElement | null, deviceId: string): void => {
    if (!audio) return;

    const selectableAudio = audio as HTMLAudioElement & { setSinkId?: (id: string) => Promise<void> };
    if (typeof selectableAudio.setSinkId !== "function") return;

    try {
        void Promise.resolve(selectableAudio.setSinkId(deviceId)).catch((error) => {
            console.warn("Audio output selection was denied; keeping the current output:", error);
        });
    } catch (error) {
        console.warn("Audio output selection was denied; keeping the current output:", error);
    }
};
