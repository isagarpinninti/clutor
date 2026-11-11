import type { ModelInfo } from "../bridge";
import { fmtBytes } from "../store";

/** Shown before a model that does not fit in memory is opened: what will happen, and a way out to a faster model. */
export function FitDialog({ model, text, onRun, onFaster, onClose }: { model: ModelInfo; text: string; onRun: () => void; onFaster: () => void; onClose: () => void }) {
  return (
    <div className="overlay" role="presentation" onClick={onClose}>
      <div className="dialog fit-dialog" role="dialog" aria-label="This model does not fit in memory" onClick={(e) => e.stopPropagation()}>
        <h2>{model.name} does not fit in memory</h2>
        <p>{text}</p>
        <p className="check-note">File size {fmtBytes(model.bytes)}. A mixture-of-experts model can still run by reading its experts from the disk, so it may work, slowly. A dense model will mostly wait on the disk.</p>
        <div className="actions">
          <button type="button" className="btn" onClick={onFaster}>Choose a faster model</button>
          <button type="button" className="btn ghost" onClick={onRun}>Run anyway</button>
        </div>
      </div>
    </div>
  );
}
