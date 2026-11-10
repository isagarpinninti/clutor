import type { ProjectFs } from "@core/agent/workspace";
import type { AgentHost } from "../bridge";

/** The project folder as the agent sees it: reads go to the shell, which checks every path again. */
export function hostProject(host: AgentHost, root: string): ProjectFs {
  return {
    list: () => host.list(root),
    read: (path) => host.read(root, path),
  };
}
