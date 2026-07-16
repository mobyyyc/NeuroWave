import { ImageResponse } from "next/og";

export const alt = "NeuroWave, audio to editable synthesis";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const marks = [28, 58, 18, 74, 34, 62, 22, 48, 78, 26, 56, 34, 68, 20, 42, 60, 30, 72];

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          backgroundColor: "#f6f5f1",
          color: "#171716",
          display: "flex",
          flexDirection: "column",
          height: "100%",
          padding: "58px 66px",
          width: "100%",
        }}
      >
        <div style={{ alignItems: "center", display: "flex", justifyContent: "space-between" }}>
          <div style={{ display: "flex", fontFamily: "serif", fontSize: 42, letterSpacing: "-2px" }}>
            Neuro<span style={{ color: "#46605a" }}>Wave</span>
          </div>
          <div style={{ color: "#46605a", display: "flex", fontSize: 16, fontWeight: 700, letterSpacing: "2px" }}>
            AUDIO TO EDITABLE SYNTHESIS
          </div>
        </div>

        <div style={{ borderTop: "2px solid #171716", display: "flex", marginTop: 42, paddingTop: 38 }}>
          <div style={{ display: "flex", flexDirection: "column", width: "67%" }}>
            <div style={{ display: "flex", fontFamily: "serif", fontSize: 92, letterSpacing: "-5px", lineHeight: 0.88 }}>
              Hear a sound.
            </div>
            <div style={{ color: "#46605a", display: "flex", fontFamily: "serif", fontSize: 92, fontStyle: "italic", letterSpacing: "-5px", lineHeight: 0.88 }}>
              Make it yours.
            </div>
          </div>
          <div style={{ alignItems: "flex-end", display: "flex", flex: 1, justifyContent: "flex-end", paddingBottom: 12 }}>
            <div style={{ backgroundColor: "#ebe9e2", display: "flex", height: 214, padding: "26px 22px", width: 255 }}>
              <div style={{ alignItems: "center", display: "flex", gap: 7, height: "100%" }}>
                {marks.map((height, index) => (
                  <div key={index} style={{ backgroundColor: index > 10 ? "#46605a" : "#171716", height, width: 5 }} />
                ))}
              </div>
            </div>
          </div>
        </div>

        <div style={{ borderTop: "1px solid #cfcdc4", display: "flex", gap: 24, marginTop: "auto", paddingTop: 20 }}>
          <span style={{ color: "#5f5e59", display: "flex", fontSize: 16, fontWeight: 700, letterSpacing: "1.5px" }}>LOCAL PROCESSING</span>
          <span style={{ color: "#a7a49b", display: "flex", fontSize: 16 }}>/</span>
          <span style={{ color: "#5f5e59", display: "flex", fontSize: 16, fontWeight: 700, letterSpacing: "1.5px" }}>WINDOWS DESKTOP</span>
          <span style={{ color: "#a7a49b", display: "flex", fontSize: 16 }}>/</span>
          <span style={{ color: "#5f5e59", display: "flex", fontSize: 16, fontWeight: 700, letterSpacing: "1.5px" }}>EDITABLE RESULTS</span>
        </div>
      </div>
    ),
    size,
  );
}
