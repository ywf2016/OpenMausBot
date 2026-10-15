import { chmodSync, existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, expect, it } from "vitest";
import { ensureDirs } from "../../config.ts";
import type { ProviderInstance } from "../../contracts.ts";
import { removeTempDir } from "../../testing/cleanup.ts";
import { recordEvents } from "../../testing/events.ts";
import { acpNativeIncomingLogMessage } from "./core.ts";
import { KimiAgentDriver } from "./kimi.ts";

const cli = join(dirname(fileURLToPath(import.meta.url)), "../../testing/fake-acp-cli.ts");
const directories: string[] = [];
const instances: ProviderInstance[] = [];
afterEach(async () => {
  for (const instance of instances.splice(0)) await instance.dispose();
  for (const directory of directories.splice(0)) await removeTempDir(directory);
});
async function fixture(driver = KimiAgentDriver, environment: Record<string, string> = {}) {
  ensureDirs(); chmodSync(cli, 0o755);
  const home = mkdtempSync(join(tmpdir(), "omb-acp-selection-")); directories.push(home);
  const dump = join(home, "dump.json"); const launches = join(home, "launches.txt");
  const instance = await driver.create({ instanceId: "selection", displayName: "Scope fixture", enabled: true, config: { cli, fullAuto: true }, environment: { HOME: home, USERPROFILE: home, FAKE_ACP_DUMP: dump, FAKE_ACP_LAUNCH_COUNT_FILE: launches, ...environment } });
  instances.push(instance);
  return { instance, recorder: recordEvents(instance.adapter), dump, launches };
}

it("omits late MCP catalog replies and native diagnostics that could carry credentials", () => {
  const message = { jsonrpc: "2.0", id: 7, result: { result: { servers: [{ env: { ORDINARY: "private-fixture-value" } }] } } };
  expect(JSON.stringify(acpNativeIncomingLogMessage(message, new Set([7])))).not.toContain("private-fixture-value");
  expect(acpNativeIncomingLogMessage({ id: 7, error: { message: "private-fixture-value" } }, new Set([7]))).toEqual({ id: 7, jsonrpc: undefined, error: "[MCP catalog error omitted]" });
  expect(acpNativeIncomingLogMessage({ id: 8, result: "ordinary" }, new Set([7]))).toEqual({ id: 8, result: "ordinary" });
});

it("gates eligible ACP servers and retires a pooled process when selection changes", async () => {
  const f = await fixture(KimiAgentDriver);
  const turn = { threadId: "scoped", text: "Fixture", approvalMode: "full" as const, toolScope: { allow: ["native:*", "mcp:notes:read"] }, integrations: {
    agents: { command: "node", args: ["agents"], env: {} },
    custom: { notes: { type: "http" as const, url: "https://example.test/mcp", headers: { authorization: "Bearer synthetic" } } },
  } };
  await f.instance.adapter.sendTurn(turn);
  await f.recorder.until((event) => event.type === "turn.completed");
  const servers = JSON.parse(readFileSync(`${f.dump}.mcp.json`, "utf8"));
  expect(servers.map((server: { name: string }) => server.name)).toEqual(["notes"]);
  const env = Object.fromEntries(servers[0].env.map((entry: { name: string; value: string }) => [entry.name, entry.value]));
  expect(JSON.parse(env.OMB_GATE_TOOL_SCOPE)).toEqual({ allow: ["native:*", "mcp:notes:read"] });
  expect(servers[0].args).not.toContain("Bearer synthetic");
  const before = Number(readFileSync(f.launches, "utf8"));
  f.recorder.events.length = 0;
  await f.instance.adapter.sendTurn({ ...turn, toolScope: { allow: ["native:*", "mcp:notes:write"] } });
  await f.recorder.until((event) => event.type === "turn.completed");
  expect(Number(readFileSync(f.launches, "utf8"))).toBe(before + 1);
});

it("refuses unsupported ACP native restrictions before a prompt even in Full access", async () => {
  const f = await fixture(KimiAgentDriver);
  await expect(f.instance.adapter.sendTurn({ threadId: "unsupported", text: "Must not run", approvalMode: "full", toolScope: { allow: [] } })).rejects.toThrow(/native tool selection.*not supported/i);
  expect(existsSync(`${f.dump}.mcp.json`)).toBe(false);
});
