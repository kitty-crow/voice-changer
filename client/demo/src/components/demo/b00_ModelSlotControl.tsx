import React, { useMemo } from "react";
import { useAppRoot } from "../../001_provider/001_AppRootProvider";
import { generateComponent } from "./002_ComponentGenerator";

const sectionMeta: Record<string, { eyebrow: string; title: string; description: string }> = {
    modelSlotArea: {
        eyebrow: "Voice library",
        title: "Models",
        description: "Choose, sort and manage the voice models available to this server.",
    },
    characterArea: {
        eyebrow: "Live conversion",
        title: "Voice",
        description: "Control conversion, tuning, gain and model-specific voice parameters.",
    },
    configArea: {
        eyebrow: "Signal path",
        title: "Audio & engine",
        description: "Configure quality, buffers, devices, recording and advanced actions.",
    },
};

export const ModelSlotControl = () => {
    const { appGuiSettingState } = useAppRoot();
    const componentSettings = appGuiSettingState.appGuiSetting.front.modelSlotControl;

    return useMemo(() => {
        if (!componentSettings || componentSettings.length == 0) {
            return <></>;
        }

        const headers = componentSettings
            .filter((x) => x.name === "headerArea")
            .map((x, index) => <React.Fragment key={`${x.name}_${index}`}>{generateComponent(x.name, x.options)}</React.Fragment>);

        const sections = componentSettings
            .filter((x) => x.name !== "headerArea")
            .map((x, index) => {
                const meta = sectionMeta[x.name];
                const id = `vc-section-${x.name}-${index}`;
                const component = generateComponent(x.name, x.options);

                return (
                    <section key={`${x.name}_${index}`} className={`vc-section vc-section--${x.name}`} aria-labelledby={meta ? id : undefined}>
                        {meta ? (
                            <div className="vc-section-heading">
                                <div>
                                    <p className="vc-eyebrow">{meta.eyebrow}</p>
                                    <h2 id={id}>{meta.title}</h2>
                                </div>
                                <p className="vc-section-description">{meta.description}</p>
                            </div>
                        ) : null}
                        <div className="vc-section-body">{component}</div>
                    </section>
                );
            });

        return (
            <>
                <div className="vc-header-region">{headers}</div>
                <main id="main-content" className="vc-main">
                    <div className="partition vc-dashboard">
                        <div className="partition-content vc-dashboard__grid">{sections}</div>
                    </div>
                </main>
            </>
        );
    }, [componentSettings]);
};
