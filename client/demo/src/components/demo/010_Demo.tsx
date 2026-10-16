import React from "react";
import { useAppRoot } from "../../001_provider/001_AppRootProvider";
import { GuiStateProvider } from "./001_GuiStateProvider";
import { Dialogs } from "./900_Dialogs";
import { ModelSlotControl } from "./b00_ModelSlotControl";
import { Dialogs2 } from "./910_Dialogs2";

export const Demo = () => {
    const { appGuiSettingState } = useAppRoot();
    const rawVersion = appGuiSettingState.version || "?";
    const version = rawVersion.startsWith("v") ? rawVersion : `v${rawVersion}`;
    const edition = (appGuiSettingState.edition || "").trim();

    return (
        <GuiStateProvider>
            <div className="main-body vc-app-shell">
                <a className="vc-skip-link" href="#main-content">
                    Skip to controls
                </a>
                <Dialogs2 />
                <Dialogs />
                <ModelSlotControl />
                <footer className="vc-footer">
                    <span className="vc-footer__product">
                        Voice Changer <span className="vc-footer__version">{version}</span>
                        {edition ? <span className="vc-footer__edition">{edition}</span> : null}
                    </span>
                    <span className="vc-footer__links">
                        <a href="https://kittycrow.dev" target="_blank" rel="noopener noreferrer">
                            Kitty Crow
                        </a>
                        <span aria-hidden="true">·</span>
                        <a href="https://github.com/kitty-crow/voice-changer" target="_blank" rel="noopener noreferrer">
                            Source
                        </a>
                    </span>
                </footer>
            </div>
        </GuiStateProvider>
    );
};
