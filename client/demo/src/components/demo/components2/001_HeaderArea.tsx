import React, { useEffect, useMemo, useState } from "react";
import { INDEXEDDB_KEY_AUDIO_OUTPUT, isDesktopApp } from "../../../const";
import { useAppRoot } from "../../../001_provider/001_AppRootProvider";
import { useAppState } from "../../../001_provider/001_AppStateProvider";
import { useIndexedDB } from "@dannadori/voice-changer-client-js";
import { useMessageBuilder } from "../../../hooks/useMessageBuilder";

export type HeaderAreaProps = {
    mainTitle: string;
    subTitle: string;
};

type Theme = "light" | "dark";

const themeKey = "voice-changer.theme";

export const HeaderArea = (props: HeaderAreaProps) => {
    const { appGuiSettingState } = useAppRoot();
    const messageBuilderState = useMessageBuilder();
    const { clearSetting } = useAppState();
    const { removeItem } = useIndexedDB({ clientType: null });
    const [theme, setTheme] = useState<Theme>(() => (document.documentElement.dataset.theme === "light" ? "light" : "dark"));

    useMemo(() => {
        messageBuilderState.setMessage(__filename, "github", { ja: "github", en: "github" });
        messageBuilderState.setMessage(__filename, "manual", { ja: "マニュアル", en: "manual" });
        messageBuilderState.setMessage(__filename, "screenCapture", { ja: "録画ツール", en: "Record Screen" });
        messageBuilderState.setMessage(__filename, "support", { ja: "支援", en: "Donation" });
    }, []);

    useEffect(() => {
        document.documentElement.dataset.theme = theme;
        try {
            localStorage.setItem(themeKey, theme);
        } catch (_) {}
        const meta = document.querySelector('meta[name="theme-color"]');
        if (meta) {
            meta.setAttribute("content", theme === "dark" ? "#0a1020" : "#edf2f8");
        }
    }, [theme]);

    const githubLink = useMemo(() => {
        return isDesktopApp() ? (
            // @ts-ignore
            <button className="link tooltip vc-icon-button" type="button" onClick={() => window.electronAPI.openBrowser("https://github.com/kitty-crow/voice-changer")} aria-label="GitHub">
                <img src="./assets/icons/github.svg" alt="" aria-hidden="true" />
                <span className="tooltip-text">{messageBuilderState.getMessage(__filename, "github")}</span>
            </button>
        ) : (
            <a className="link tooltip vc-icon-button" href="https://github.com/kitty-crow/voice-changer" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                <img src="./assets/icons/github.svg" alt="" aria-hidden="true" />
                <span className="tooltip-text">{messageBuilderState.getMessage(__filename, "github")}</span>
            </a>
        );
    }, []);

    const manualLink = useMemo(() => {
        const href = "https://github.com/w-okada/voice-changer/blob/master/tutorials/tutorial_rvc_ja_latest.md";
        return isDesktopApp() ? (
            // @ts-ignore
            <button className="link tooltip vc-icon-button" type="button" onClick={() => window.electronAPI.openBrowser(href)} aria-label="Manual">
                <img src="./assets/icons/help-circle.svg" alt="" aria-hidden="true" />
                <span className="tooltip-text tooltip-text-100px">{messageBuilderState.getMessage(__filename, "manual")}</span>
            </button>
        ) : (
            <a className="link tooltip vc-icon-button" href={href} target="_blank" rel="noopener noreferrer" aria-label="Manual">
                <img src="./assets/icons/help-circle.svg" alt="" aria-hidden="true" />
                <span className="tooltip-text tooltip-text-100px">{messageBuilderState.getMessage(__filename, "manual")}</span>
            </a>
        );
    }, []);

    const toolLink = useMemo(() => {
        const href = "https://w-okada.github.io/screen-recorder-ts/";
        const openTool = () => {
            if (isDesktopApp()) {
                // @ts-ignore
                window.electronAPI.openBrowser(href);
            } else {
                window.open(href, "_blank", "noreferrer");
            }
        };
        return (
            <button className="link tooltip vc-icon-button" type="button" onClick={openTool} aria-label="Record screen">
                <img src="./assets/icons/tool.svg" alt="" aria-hidden="true" />
                <span className="tooltip-text tooltip-text-100px">{messageBuilderState.getMessage(__filename, "screenCapture")}</span>
            </button>
        );
    }, []);

    const coffeeLink = useMemo(() => {
        const href = "https://www.buymeacoffee.com/wokad";
        return isDesktopApp() ? (
            // @ts-ignore
            <button className="link tooltip vc-donation-button" type="button" onClick={() => window.electronAPI.openBrowser(href)} aria-label="Donation">
                <img className="donate-img" src="./assets/buymeacoffee.png" alt="" aria-hidden="true" />
                <span className="tooltip-text tooltip-text-100px">{messageBuilderState.getMessage(__filename, "support")}</span>
            </button>
        ) : (
            <a className="link tooltip vc-donation-button" href={href} target="_blank" rel="noopener noreferrer" aria-label="Donation">
                <img className="donate-img" src="./assets/buymeacoffee.png" alt="" aria-hidden="true" />
                <span className="tooltip-text tooltip-text-100px">{messageBuilderState.getMessage(__filename, "support")}</span>
            </a>
        );
    }, []);

    const onClearSettingClicked = async () => {
        await clearSetting();
        await removeItem(INDEXEDDB_KEY_AUDIO_OUTPUT);
        location.reload();
    };

    return (
        <header className="headerArea vc-site-header">
            <div className="title1 vc-brand-lockup">
                <span className="title">{props.mainTitle}</span>
                <span className="title-version">{props.subTitle}</span>
                <span className="title-version-number">{appGuiSettingState.version}</span>
                {appGuiSettingState.edition ? <span className="title-version-number vc-edition-chip">{appGuiSettingState.edition}</span> : null}
            </div>
            <nav className="icons vc-header-actions" aria-label="Application actions">
                <span className="belongings vc-header-links">
                    {githubLink}
                    {manualLink}
                    {toolLink}
                    {coffeeLink}
                </span>
                <span className="belongings vc-header-utilities">
                    <button className="vc-theme-toggle" type="button" data-theme-toggle aria-pressed={theme === "dark"} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
                        <span className="vc-theme-toggle__dot" aria-hidden="true" />
                        <span>{theme === "dark" ? "Light" : "Dark"}</span>
                    </button>
                    <button className="belongings-button vc-clear-button" type="button" onClick={onClearSettingClicked}>
                        clear setting
                    </button>
                </span>
            </nav>
        </header>
    );
};
