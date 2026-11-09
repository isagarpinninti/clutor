import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { buildLibrary, pickImage, pickLlm, pickVision, scanFiles, type Classified, type Entry, type ImagePipeline, type Library, type Choice, type VisionModel } from "@core/models/library";
import { bridge, type ModelInfo } from "./bridge";
import { readSpeed } from "./store";

const GiB = 2 ** 30;

/** The model as the rest of the app knows it (no file extension in the name, no quantisation split out). */
export const toModelInfo = (c: Classified): ModelInfo => ({ path: c.path, name: c.name.replace(/\.(gguf|safetensors)$/i, ""), bytes: c.bytes });

/** What the library decided for each task on this computer, with the reason in words. */
export interface Auto {
  chat: Choice<Classified>;
  code: Choice<Classified>;
  vision: Choice<VisionModel>;
  image: Choice<ImagePipeline>;
}

const classified = new Map<string, Classified>();   // by path and size: a file is read once per run of the app

/** Reads every model file in the folders (the headers only) and works out what each is. */
export async function loadLibrary(folders: string[]): Promise<Classified[]> {
  const files = await bridge.scanFiles(folders);
  const todo: Entry[] = files.filter((f) => !classified.has(`${f.path}|${f.bytes}`));
  const done = await scanFiles(todo, (e) => ({ size: e.bytes, read: (o, n) => bridge.inspect.read(e.path, o, n) }));
  done.forEach((c, i) => classified.set(`${(todo[i] as Entry).path}|${(todo[i] as Entry).bytes}`, c));
  return files.map((f) => classified.get(`${f.path}|${f.bytes}`) as Classified);
}

/** The folder understood: its models, and which one to use for chat, code, pictures and image generation. Rescans when the folders change. */
export function useLibrary(folders: string[], ramBytes: number | undefined): { library: Library; auto: Auto; scanning: boolean; rescan: () => void } {
  const [files, setFiles] = useState<Classified[]>([]);
  const [scanning, setScanning] = useState(false);
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);   // set again on mount: React's development mode mounts, unmounts and mounts again
  const rescan = useCallback((): void => {
    setScanning(true);
    loadLibrary(folders).then((f) => { if (alive.current) setFiles(f); }, () => { if (alive.current) setFiles([]); }).finally(() => { if (alive.current) setScanning(false); });
  }, [folders]);
  useEffect(rescan, [rescan]);
  const library = useMemo(() => buildLibrary(files), [files]);
  const auto = useMemo((): Auto => {
    const machine = { ramBytes: ramBytes ?? 8 * GiB, measuredTps: (n: string) => readSpeed(n.replace(/\.(gguf|safetensors)$/i, "")) };
    return { chat: pickLlm(library, "chat", machine), code: pickLlm(library, "code", machine), vision: pickVision(library, machine), image: pickImage(library, machine) };
  }, [library, ramBytes]);
  return { library, auto, scanning, rescan };
}
