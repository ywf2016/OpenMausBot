// ACP driver contract tests, run against the scripted fake ACP CLI in
// server/testing/fake-acp-cli.ts. Covers the shared acp/core.ts runtime via
// its kept harness shims (kimi / cursor / qwen):
// normalize the ACP handshake into canonical events, keep argv/env hygiene,
// broker permission asks, and settle interrupts/crashes cleanly.
//
// The fake CLI is a shebang script Windows cannot exec directly —
// resolveCliSpawn turns it into `node <script>`, so these run everywhere.
import { randomUUID } from "node:crypto";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ensureDirs, NATIVE_DIR } from "../../config.ts";
import { BUILT_IN_DATA_SYSTEM_PROMPT } from "../../data/instructions.ts";
import type { ProviderInstance } from "../../contracts.ts";
import { TurnNotStartedError } from "../../contracts.ts";
import { recordEvents, type EventRecorder } from "../../testing/events.ts";
import { promptSplitFingerprints, writePromptSplitReceipt } from "../prompt-split.ts";
import { createAcpDriver, DEFAULT_ACP_PROMPT_IDLE_MS, skipSubscriptionAuthForLocalInject, type AcpSupport } from "./core.ts";
import { KimiAgentDriver } from "./kimi.ts";
import { CursorAgentDriver } from "./cursor.ts";
import { QwenAgentDriver } from "./qwen.ts";
import { removeTempDir } from "../../testing/cleanup.ts";
import { startFakeHttpMcp } from "../../testing/fake-http-mcp-server.ts";
import * as procs from "../../procs.ts";
import * as quietStatus from "./quiet-status.ts";

const FAKE_CLI = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "testing", "fake-acp-cli.ts");

/** A harness that exists only in tests: it exercises the opt-in session-config
 *  model hook so PR 1 can prove the core capability without shipping a visible
 *  engine. Real harnesses live in their own file. */
const SELECT_MODEL_SUPPORT: AcpSupport = {
  driverKind: "selectModelTest",
  displayName: "Select Model Test",
  models: { default: "m-one", options: [{ id: "m-one", label: "One" }, { id: "m-two", label: "Two" }] },
  defaultCli: "fake-select-model",
  nativeSource: "test.acp",
  loginNote: "never reached",
  selectModel: { configId: "model" },
  spawnArgs: () => [],
  pickAuthMethod: () => null,
  authFailure: "continue",
  isAuthenticated: () => true,
};
const SelectModelDriver = createAcpDriver(SELECT_MODEL_SUPPORT);

/** Image input is an explicit per-harness opt-in. Keep these transport tests
 * independent from production harnesses whose native image support has not
 * been verified. */
const NativeImageDriver = createAcpDriver({
  ...SELECT_MODEL_SUPPORT,
  driverKind: "nativeImageTest",
  images: true,
});

/** Proves transformEnv can vary with the instance config, which is how the
 *  opencode driver picks its permission policy from `fullAuto`. */
const EnvPolicyDriver = createAcpDriver({
  ...SELECT_MODEL_SUPPORT,
  driverKind: "envPolicyTest",
  selectModel: undefined,
  transformEnv: (env, config) => {
    env.TEST_POLICY = config.fullAuto ? "auto" : "ask";
  },
});

/** Proves snapshot() awaits an async isAuthenticated, which is how the
 *  opencode driver answers from a discovered catalog. */
const AsyncAuthDriver = createAcpDriver({
  ...SELECT_MODEL_SUPPORT,
  driverKind: "asyncAuthTest",
  selectModel: undefined,
  isAuthenticated: async () => true,
});

const ClassifiedErrorDriver = createAcpDriver({
  ...SELECT_MODEL_SUPPORT,
  driverKind: "classifiedErrorTest",
  selectModel: undefined,
  classifyError: (error) =>
    error && typeof error === "object" && (error as { code?: unknown }).code === -32000
      ? "invalid_credentials"
      : undefined,
});

const CONTROL_PLANE_FIXTURE = {
  OMB_CLOUD_READY_TOKEN: "ready-should-not-leak", OMB_CLOUD_BOOTSTRAP: "bootstrap-should-not-leak",
  OMB_LICENSE_KEY: "license-should-not-leak", OMB_INSTALLATION_CREDENTIAL: "fleet-should-not-leak",
};

describe("skipSubscriptionAuthForLocalInject", () => {
  it("is true only for a host:: inject id", () => {
    expect(skipSubscriptionAuthForLocalInject("omlx::MiniMax-M3-4bit")).toBe(true);
    expect(skipSubscriptionAuthForLocalInject("unsloth::orcarouter/Qwen3.8-27B-Uncensored-GGUF")).toBe(true);
    expect(skipSubscriptionAuthForLocalInject("grok-4.6")).toBe(false);
    expect(skipSubscriptionAuthForLocalInject(undefined)).toBe(false);
  });
});

describe("ACP decodeConfig", () => {
  it("resolves a dynamic model catalog when a support provides one", async () => {
    const support: AcpSupport = {
      driverKind: "dynamic-test",
      displayName: "Dynamic Test",
      models: { default: "fallback", options: [{ id: "fallback", label: "Fallback" }] },
      defaultCli: FAKE_CLI,
      nativeSource: "dynamic-test.acp",
      loginNote: "not authenticated",
      spawnArgs: () => [],
      pickAuthMethod: () => null,
      authFailure: "continue",
      isAuthenticated: () => true,
      resolveModels: async () => ({
        default: "dynamic-model",
        options: [{ id: "dynamic-model", label: "Dynamic model" }],
      }),
    };
    const driver = createAcpDriver(support);
    const instance = await driver.create({
      instanceId: "dynamic-test",
      displayName: "Dynamic Test",
      environment: {},
      enabled: true,
      config: driver.defaultConfig(),
    });
    expect(instance.models).toEqual({
      default: "dynamic-model",
      options: [{ id: "dynamic-model", label: "Dynamic model" }],
    });
    await instance.dispose();
  });
  it("kimi defaults to the kimi binary and declares cross-platform setup", () => {
    expect(KimiAgentDriver.decodeConfig(undefined)).toEqual({ cli: "kimi", fullAuto: false, workspace: undefined });
    expect(KimiAgentDriver.install?.command).toMatchObject({
      darwin: expect.stringContaining("install.sh"),
      linux: expect.stringContaining("install.sh"),
      win32: expect.stringContaining("install.ps1"),
    });
    expect(KimiAgentDriver.install?.signInCommand).toBe("kimi login");
  });
  it("cursor defaults to its unambiguous binary and declares cross-platform setup", () => {
    expect(CursorAgentDriver.decodeConfig(undefined)).toEqual({
      cli: "cursor-agent",
      fullAuto: false,
      workspace: undefined,
    });
    expect(CursorAgentDriver.install?.command).toMatchObject({
      darwin: expect.stringContaining("cursor.com/install"),
      linux: expect.stringContaining("cursor.com/install"),
      win32: expect.stringContaining("cursor.com/install"),
    });
    expect(CursorAgentDriver.install?.signInCommand).toBe("cursor-agent login");
  });
  it("fullAuto only when explicitly true", () => {
    expect(KimiAgentDriver.decodeConfig({ fullAuto: "yes" }).fullAuto).toBe(false);
    expect(KimiAgentDriver.decodeConfig({ fullAuto: true }).fullAuto).toBe(true);
  });

  it("advertises per-bot local CUA but rejects legacy full-auto turns without a mode", async () => {
    const fullAuto = await KimiAgentDriver.create({
      instanceId: "kimi-full-auto",
      displayName: "Kimi Full Auto",
      environment: {},
      enabled: true,
      config: { cli: FAKE_CLI, fullAuto: true },
    });
    expect(fullAuto.adapter.capabilities.localComputerMcp).toBe(true);
    await expect(
      fullAuto.adapter.sendTurn({
        threadId: "t-full-auto-local",
        text: "click",
        integrations: {
          localComputer: {
            command: "/cua-driver",
            args: ["mcp"],
            env: {},
            platform: "linux",
            scope: "local-computer",
          },
        },
      }),
    ).rejects.toThrow(/interactive provider approvals/);
    await fullAuto.dispose();
  });
});

describe("ACP turns (fake CLI)", () => {
  let instance: ProviderInstance;
  let recorder: EventRecorder;
  let scratch: string;

  const create = async (driver = KimiAgentDriver, mode?: string) => {
    if (mode) process.env.FAKE_ACP_MODE = mode;
    instance = await driver.create({
      instanceId: "acp-test",
      displayName: "ACP Test",
      environment: {},
      enabled: true,
      config: { cli: FAKE_CLI, fullAuto: false },
    });
    recorder = recordEvents(instance.adapter);
  };

  let savedAgyModel: string | undefined;
  beforeEach(() => {
    savedAgyModel = process.env.AGY_ACP_DEFAULT_MODEL;
    delete process.env.AGY_ACP_DEFAULT_MODEL;
    ensureDirs();
    chmodSync(FAKE_CLI, 0o755);
    scratch = mkdtempSync(join(tmpdir(), "omb-acp-test-"));
  });

  afterEach(async () => {
    delete process.env.FAKE_ACP_MODE;
    delete process.env.FAKE_ACP_DUMP;
    delete process.env.FAKE_ACP_RPC_DUMP;
    delete process.env.FAKE_ACP_RPC_APPEND_FILE;
    delete process.env.FAKE_ACP_RPC_FAILURE_FILE;
    delete process.env.FAKE_ACP_RPC_FAILURE_METHOD;
    delete process.env.FAKE_ACP_RPC_FAILURE_AFTER_OUTPUT;
    delete process.env.FAKE_ACP_RPC_FAILURE_AFTER_USAGE;
    delete process.env.FAKE_ACP_LOAD_ERROR;
    delete process.env.FAKE_ACP_ALLOW_ALWAYS;
    delete process.env.FAKE_ACP_PERMISSION_ANSWER;
    delete process.env.FAKE_ACP_PERMISSION_TOOL_CALL;
    delete process.env.FAKE_ACP_PERMISSION_OPTIONS;
    delete process.env.FAKE_ACP_QUESTION_OPTIONS;
    delete process.env.FAKE_ACP_HOLD_ON_CANCELLED;
    delete process.env.XAI_API_KEY;
    delete process.env.OPENCODE_API_KEY;
    delete process.env.CLINE_API_KEY;
    delete process.env.CURSOR_API_KEY;
    delete process.env.CURSOR_AUTH_TOKEN;
    delete process.env.BOX_TOKEN;
    delete process.env.OMB_TTS_KEY;
    if (savedAgyModel !== undefined) process.env.AGY_ACP_DEFAULT_MODEL = savedAgyModel;
    else delete process.env.AGY_ACP_DEFAULT_MODEL;
    for (const name of Object.keys(CONTROL_PLANE_FIXTURE)) delete process.env[name];
    delete process.env.FAKE_ACP_MODELS;
    delete process.env.FAKE_ACP_MODEL_STICKS;
    delete process.env.FAKE_ACP_USAGE_ROOT;
    delete process.env.FAKE_ACP_LOAD_NULL;
    delete process.env.FAKE_ACP_REJECT_LIVE_LOAD_FILE;
    delete process.env.FAKE_ACP_IMAGE_CAPABILITY;
    delete process.env.FAKE_ACP_TOOL_MS;
    delete process.env.FAKE_ACP_USAGE_UPDATES_FILE;
    delete process.env.FAKE_ACP_DUMP_PROMPT;
    delete process.env.OPENMAUS_ACP_PROMPT_IDLE_TIMEOUT_MS;
    delete process.env.OMB_ACP_QUIET_NOTICE_MS;
    delete process.env.OMB_ACP_QUIET_TICK_MS;
    delete process.env.FAKE_ACP_QUIET_MS;
    delete process.env.FAKE_ACP_QWEN_LOG;
    delete process.env.FAKE_ACP_LOG_EVERY_MS;
    delete process.env.QWEN_HOME;
    delete process.env.OMB_ACP_SESSION_IDLE_MS;
    delete process.env.OMB_ACP_SESSION_IDLE_MIN_MS;
    delete process.env.FAKE_ACP_LAUNCH_COUNT_FILE;
    recorder?.stop();
    await instance?.dispose();
    vi.restoreAllMocks();
    await removeTempDir(scratch);
  });

  it("names the resolved executable when spawning it fails", async () => {
    const missing = join(scratch, "resolved-managed-runtime");
    const driver = createAcpDriver({
      ...SELECT_MODEL_SUPPORT,
      selectModel: undefined,
      resolveCommand: async () => ({ command: missing }),
    });
    instance = await driver.create({
      instanceId: "resolved-spawn-error",
      displayName: "Resolved spawn error",
      environment: {},
      enabled: true,
      config: { cli: "managed-alias", fullAuto: false },
    });
    recorder = recordEvents(instance.adapter);

    await instance.adapter.sendTurn({ threadId: "t-resolved-spawn-error", text: "go" });
    await recorder.until((event) => event.type === "turn.completed");
    const error = recorder.events.find((event) => event.type === "runtime.error");
    expect(error?.message).toContain(missing);
    expect(error?.message).not.toContain("managed-alias");
  });

  it("pairs an unkeyed tool call's lifecycle events on one stable itemId", async () => {
    await create(KimiAgentDriver, "unkeyed-tool");
    const { turnId } = await instance.adapter.sendTurn({ threadId: "t-unkeyed-tool", text: "go" });
    await recorder.until((e) => e.type === "turn.completed" && e.turnId === turnId);
    const started = recorder.events.find((e) => e.type === "item.started" && e.itemType === "tool");
    const completed = recorder.events.find((e) => e.type === "item.completed" && e.itemType === "tool");
    // The agent sent no toolCallId: without a synthetic id the completion
    // would be dropped by the itemId guard and the tool chip would never
    // show its result.
    expect(typeof started?.itemId).toBe("string");
    expect(started?.itemId).toBeTruthy();
    expect(completed?.itemId).toBe(started?.itemId);
  });

  it("normalizes a full turn into the canonical event sequence", async () => {
    await create();
    const { turnId } = await instance.adapter.sendTurn({ threadId: "t-happy", text: "hi", model: "kimi-test" });
    await recorder.until((e) => e.type === "turn.completed");

    const types = recorder.events.map((e) => e.type);
    expect(types).toEqual([
      "turn.started",
      "session.started",
      "content.delta",
      "item.completed", // assistant_text before the tool, not summed on settle
      "item.started", // tool tc-1
      "item.completed", // tool tc-1 done
      "thread.token-usage.updated",
      "turn.completed",
    ]);
    expect(recorder.events.every((e) => e.turnId === turnId && e.provider === "kimiAgent")).toBe(true);
    expect(recorder.events.filter((event) => event.itemId === "tc-1")).toMatchObject([
      { type: "item.started", input: expect.stringContaining("/fixture/readme.md") },
      { type: "item.completed", output: expect.stringContaining("fixture file content") },
    ]);
    expect(JSON.stringify(recorder.events)).not.toContain("acp-input-secret");
    expect(JSON.stringify(recorder.events)).not.toContain("acp-output-secret");
    const usage = recorder.events.find((e) => e.type === "thread.token-usage.updated")!;
    expect(usage).toMatchObject({ input: 10, output: 5 });
    const text = recorder.events.find((e) => e.type === "item.completed" && (e as any).itemType === "assistant_text")!;
    expect((text as any).text).toBe("hello from fake acp");
    const done = recorder.events.at(-1)!;
    expect(done).toMatchObject({ type: "turn.completed", ok: true });
    expect(instance.adapter.hasSession("t-happy")).toBe(false);
  });

  it("sends images as negotiated ACP content blocks without copying bytes into diagnostics", async () => {
    const dump = join(scratch, "image-prompt.json");
    const imagePath = join(scratch, "tiny.webp");
    const bytes = Buffer.from("private-acp-image-bytes");
    const base64 = bytes.toString("base64");
    writeFileSync(imagePath, bytes);
    process.env.FAKE_ACP_DUMP = dump;
    process.env.FAKE_ACP_DUMP_PROMPT = "1";
    process.env.FAKE_ACP_IMAGE_CAPABILITY = "1";
    await create(NativeImageDriver);

    const { turnId } = await instance.adapter.sendTurn({
      threadId: "t-acp-native-image",
      text: "What is this?",
      images: [{ path: imagePath, mime: "image/webp", bytes: bytes.length }],
    });
    await recorder.until((event) => event.type === "turn.completed" && event.turnId === turnId);

    expect(JSON.parse(readFileSync(`${dump}.prompt.json`, "utf8"))).toEqual([
      { type: "text", text: "What is this?" },
      { type: "image", data: base64, mimeType: "image/webp" },
    ]);
    const nativeLog = readFileSync(join(NATIVE_DIR, "t-acp-native-image.ndjson"), "utf8");
    expect(nativeLog).not.toContain(base64);
    expect(nativeLog).toContain(`[image data: ${base64.length} base64 chars]`);
  });

  it("delivers the full prompt once per native session and rides volatile changes as notes", async () => {
    const dump = join(scratch, "acp-prompt-split.json");
    process.env.FAKE_ACP_DUMP = dump;
    process.env.FAKE_ACP_DUMP_PROMPT = "1";
    await create();
    // The receipt store is keyed by thread and session id, so a unique thread
    // keeps the run hermetic against earlier executions of this suite.
    const threadId = "t-acp-prompt-split-" + randomUUID();
    const promptOf = () =>
      (JSON.parse(readFileSync(dump + ".prompt.json", "utf8")) as Array<{ type: string; text: string }>)[0]?.text;
    const send = async (text: string, volatile: string) => {
      const { turnId } = await instance.adapter.sendTurn({
        threadId,
        text,
        system: "Standing rules.\n\n" + volatile,
        systemStable: "Standing rules.",
        systemVolatile: volatile,
      });
      await recorder.until((event) => event.type === "turn.completed" && event.turnId === turnId);
      return promptOf();
    };

    // The establishing turn carries the full prompt, exactly as before.
    expect(await send("first", "Memory: likes quiet hours."))
      .toBe("Standing rules.\n\nMemory: likes quiet hours.\n\nfirst");
    // The pooled session already carries it: later turns go through bare.
    expect(await send("second", "Memory: likes quiet hours.")).toBe("second");
    // A changed volatile half rides the next prompt as a labelled note.
    expect(await send("third", "Memory: moved to Toronto."))
      .toBe("Context from OpenMausBot updated since this conversation started; it replaces any earlier copy:\n\nMemory: moved to Toronto.\n\nthird");
    // A cleared volatile half is announced once, not silently dropped.
    expect(await send("fourth", "")).toContain("have been cleared");
    expect(await send("fifth", "")).toBe("fifth");
  });

  it("re-anchors the full prompt after eight bare turns on one native session", async () => {
    const dump = join(scratch, "acp-prompt-reanchor.json");
    process.env.FAKE_ACP_DUMP = dump;
    process.env.FAKE_ACP_DUMP_PROMPT = "1";
    await create();
    const threadId = "t-acp-prompt-reanchor-" + randomUUID();
    const promptOf = () =>
      (JSON.parse(readFileSync(dump + ".prompt.json", "utf8")) as Array<{ type: string; text: string }>)[0]?.text;
    const messages: string[] = [];
    for (let i = 0; i < 10; i++) {
      const { turnId } = await instance.adapter.sendTurn({
        threadId,
        text: "turn " + i,
        system: "Standing rules.\n\nMemory: likes quiet hours.",
        systemStable: "Standing rules.",
        systemVolatile: "Memory: likes quiet hours.",
      });
      await recorder.until((event) => event.type === "turn.completed" && event.turnId === turnId);
      messages.push(promptOf()!);
    }
    // Agents can compact their own history away between turns; the re-anchor
    // backstop returns the standing instructions within a bounded window.
    const full = "Standing rules.\n\nMemory: likes quiet hours.";
    expect(messages[0]).toBe(full + "\n\nturn 0");
    for (let i = 1; i <= 8; i++) expect(messages[i]).toBe("turn " + i);
    expect(messages[9]).toBe(full + "\n\nturn 9");
  });

  // One split-prompt thread: usage() sets the context sizes the fake reports
  // on the next turn, and send() returns the prompt text the agent received.
  const FULL = "Standing rules.\n\nMemory: likes quiet hours.";
  const usageThread = async (name: string) => {
    const dump = join(scratch, `acp-${name}.json`);
    const usageFile = join(scratch, `acp-${name}-usage.json`);
    process.env.FAKE_ACP_DUMP = dump;
    process.env.FAKE_ACP_DUMP_PROMPT = "1";
    process.env.FAKE_ACP_USAGE_UPDATES_FILE = usageFile;
    writeFileSync(usageFile, "[]");
    await create();
    const threadId = `t-acp-${name}-` + randomUUID();
    const usage = (...used: number[]) => writeFileSync(usageFile, JSON.stringify(used));
    const send = async (text: string, ok = true) => {
      const { turnId } = await instance.adapter.sendTurn({
        threadId,
        text,
        system: FULL,
        systemStable: "Standing rules.",
        systemVolatile: "Memory: likes quiet hours.",
      });
      expect(await recorder.until((event) => event.type === "turn.completed" && event.turnId === turnId)).toMatchObject({ ok });
      return (JSON.parse(readFileSync(dump + ".prompt.json", "utf8")) as Array<{ type: string; text: string }>)[0]?.text;
    };
    return { threadId, usage, send };
  };

  it("re-anchors the full prompt on the turn after the reported context collapses", async () => {
    const { usage, send } = await usageThread("compaction");
    expect(await send("turn 1")).toBe(FULL + "\n\nturn 1");
    usage(100000);
    expect(await send("turn 2")).toBe("turn 2");
    // the context collapses mid-turn (120000 -> 45000), so the next turn is full
    usage(120000, 45000);
    expect(await send("turn 3")).toBe("turn 3");
    usage();
    expect(await send("turn 4")).toBe(FULL + "\n\nturn 4");
    expect(await send("turn 5")).toBe("turn 5");
  });

  it("keeps the split through an ordinary context dip", async () => {
    const { usage, send } = await usageThread("ordinary-dip");
    usage(108641);
    expect(await send("turn 1")).toBe(FULL + "\n\nturn 1");
    // a 17% dip is an ordinary step, not a compaction
    usage(90258);
    expect(await send("turn 2")).toBe("turn 2");
    expect(await send("turn 3")).toBe("turn 3");
  });

  it("detects a collapse against the previous turn's context", async () => {
    const { usage, send } = await usageThread("cross-turn");
    expect(await send("turn 1")).toBe(FULL + "\n\nturn 1");
    usage(150000);
    expect(await send("turn 2")).toBe("turn 2");
    // compaction before the first report: a collapse against last turn's peak
    usage(50000);
    expect(await send("turn 3")).toBe("turn 3");
    usage();
    expect(await send("turn 4")).toBe(FULL + "\n\nturn 4");
  });

  it("detects a cumulative collapse against an earlier turn's high-water mark", async () => {
    const { usage, send } = await usageThread("cumulative-collapse");
    usage(100000, 75000);
    expect(await send("turn 1")).toBe(FULL + "\n\nturn 1");
    // Neither adjacent dip exceeds 40%, but the total fall from the peak does.
    usage(50000);
    expect(await send("turn 2")).toBe("turn 2");
    usage();
    expect(await send("turn 3")).toBe(FULL + "\n\nturn 3");
    expect(await send("turn 4")).toBe("turn 4");
  });

  it.each([40000, 75000])("invalidates a rejected prompt's receipt only after compaction (usage: %s)", async (used) => {
    const failureFile = join(scratch, "compaction-failure.json");
    process.env.FAKE_ACP_RPC_FAILURE_FILE = failureFile;
    process.env.FAKE_ACP_RPC_FAILURE_AFTER_USAGE = "1";
    const { usage, send } = await usageThread("rejected-compaction");
    usage(100000);
    expect(await send("turn 1")).toBe(FULL + "\n\nturn 1");
    usage(used);
    writeFileSync(failureFile, JSON.stringify({ code: -32603, message: "Internal error after compaction" }));
    expect(await send("turn 2", false)).toBe("turn 2");
    unlinkSync(failureFile);
    usage();
    expect(await send("turn 3")).toBe(used < 60000 ? FULL + "\n\nturn 3" : "turn 3");
    expect(await send("turn 4")).toBe("turn 4");
  });

  it("ignores a zero usage report", async () => {
    const { usage, send } = await usageThread("zero-usage");
    usage(150000);
    expect(await send("turn 1")).toBe(FULL + "\n\nturn 1");
    // zero is not a measurement: no trigger, and the peak stays
    usage(0);
    expect(await send("turn 2")).toBe("turn 2");
    expect(await send("turn 3")).toBe("turn 3");
  });

  it("never triggers from a receipt written before lastUsed existed", async () => {
    const { threadId, usage, send } = await usageThread("legacy-receipt");
    // a legacy receipt has no lastUsed, so a low first reading is no collapse
    writePromptSplitReceipt(
      "kimiAgent",
      JSON.stringify([threadId, "fake-acp-session"]),
      promptSplitFingerprints("Standing rules.", "Memory: likes quiet hours."),
    );
    usage(50000);
    expect(await send("turn 1")).toBe("turn 1");
    expect(await send("turn 2")).toBe("turn 2");
  });

  it("fails clearly when an image-capable adapter meets an older ACP runtime", async () => {
    const imagePath = join(scratch, "tiny.png");
    writeFileSync(imagePath, "not-read-before-capability-check");
    await create(NativeImageDriver);

    const { turnId } = await instance.adapter.sendTurn({
      threadId: "t-acp-image-unsupported",
      text: "Inspect this",
      images: [{ path: imagePath, mime: "image/png", bytes: 32 }],
    });
    const done = await recorder.until((event) => event.type === "turn.completed" && event.turnId === turnId);

    expect(done).toMatchObject({ ok: false, stopReason: "rpc_error" });
    expect(recorder.events.find((event) => event.type === "runtime.error")?.message).toMatch(
      /does not advertise ACP image input/,
    );
  });

  it("keeps text-path fallback for adapters that do not advertise image input", async () => {
    const dump = join(scratch, "text-fallback.json");
    process.env.FAKE_ACP_DUMP = dump;
    process.env.FAKE_ACP_DUMP_PROMPT = "1";
    await create(SelectModelDriver);

    const text = '<attached-image path="/private/image.png" name="image.png" />';
    const { turnId } = await instance.adapter.sendTurn({
      threadId: "t-acp-image-fallback",
      text,
      // A missing path proves the driver did not attempt native ingestion.
      images: [{ path: join(scratch, "missing.png"), mime: "image/png", bytes: 12 }],
    });
    await recorder.until((event) => event.type === "turn.completed" && event.turnId === turnId);

    expect(JSON.parse(readFileSync(`${dump}.prompt.json`, "utf8"))).toEqual([{ type: "text", text }]);
  });

  it("emits each assistant text block before the tool that follows it", async () => {
    await create(KimiAgentDriver, "interleave");
    await instance.adapter.sendTurn({ threadId: "t-interleave", text: "go", model: "kimi-test" });
    await recorder.until((e) => e.type === "turn.completed");

    const types = recorder.events.map((e) => e.type);
    expect(types).toEqual([
      "turn.started",
      "session.started",
      "content.delta",
      "item.completed", // before one
      "item.started", // tc-1
      "item.completed", // tc-1
      "content.delta",
      "item.completed", // before two
      "item.started", // tc-2
      "item.completed", // tc-2
      "content.delta",
      "thread.token-usage.updated",
      "item.completed", // after — no following tool, so settle flushes
      "turn.completed",
    ]);
    const texts = recorder.events
      .filter((e) => e.type === "item.completed" && (e as { itemType?: string }).itemType === "assistant_text")
      .map((e) => (e as { text: string }).text);
    expect(texts).toEqual(["before one", "before two", "after"]);
  });

  it("normalizes a structured ACP image block without treating it as text", async () => {
    await create(KimiAgentDriver, "image");
    await instance.adapter.sendTurn({ threadId: "t-image", text: "draw it" });
    await recorder.until((event) => event.type === "turn.completed");

    const image = recorder.events.find(
      (event) => event.type === "item.completed" && event.itemType === "assistant_image",
    );
    expect(image).toMatchObject({
      type: "item.completed",
      itemType: "assistant_image",
      alt: "Generated image",
    });
    expect(image && "data" in image ? image.data : "").toMatch(/^iVBOR/);
    expect(
      recorder.events.some(
        (event) => event.type === "item.completed" && event.itemType === "assistant_text",
      ),
    ).toBe(false);
  });

  it("reads token usage from the root of the prompt result", async () => {
    process.env.FAKE_ACP_USAGE_ROOT = "1";
    await create();
    await instance.adapter.sendTurn({ threadId: "t-usage-root", text: "go" });
    await recorder.until((e) => e.type === "turn.completed");

    const usage = recorder.events.find((e) => e.type === "thread.token-usage.updated");
    expect(usage).toMatchObject({ input: 10, output: 5 });
  });

  /** The fake CLI hardcodes its usage fixture, so the cache-usage tests tap
   *  the spawned child's stdout in flight and splice the wanted usage into
   *  the one reply that carries stopReason — the session/prompt result — at
   *  the root or under _meta, the two shapes real agents send. */
  const stubPromptResultUsage = (usage: Record<string, unknown>, at: "root" | "meta") => {
    const spawn = procs.spawnCli;
    vi.spyOn(procs, "spawnCli").mockImplementation((...args) => {
      const child = spawn(...args);
      const stdout = child.stdout;
      const spliceUsage = (message: any) => {
        if (message?.result && typeof message.result === "object" && "stopReason" in message.result) {
          message.result = at === "root"
            ? { ...message.result, usage: { ...usage }, _meta: {} }
            : { ...message.result, _meta: { ...message.result._meta, ...usage } };
        }
        return message;
      };
      const rewriteChunk = (chunk: string) =>
        chunk.split("\n").map((line) => {
          if (!line.includes("stopReason")) return line;
          try {
            return JSON.stringify(spliceUsage(JSON.parse(line)));
          } catch {
            return line; // not a complete wire message: pass it through untouched
          }
        }).join("\n");
      const realOn = stdout.on.bind(stdout);
      vi.spyOn(stdout, "on").mockImplementation(((event: string, listener: (chunk: string) => void) =>
        realOn(event, event === "data" ? (chunk: string) => listener(rewriteChunk(String(chunk))) : listener)) as never);
      return child;
    });
  };

  it("folds opencode cache reads into input and names them", async () => {
    stubPromptResultUsage(
      { inputTokens: 5030, outputTokens: 3, cachedReadTokens: 152960, totalTokens: 157993 },
      "root",
    );
    await create();
    await instance.adapter.sendTurn({ threadId: "t-usage-cache-exclusive", text: "go" });
    await recorder.until((e) => e.type === "turn.completed");

    const usage = recorder.events.find((e) => e.type === "thread.token-usage.updated");
    expect(usage).toMatchObject({ input: 157990, output: 3, cachedInput: 152960 });
  });

  it("leaves input alone when totalTokens shows cache reads already inside it", async () => {
    stubPromptResultUsage(
      { inputTokens: 1000, outputTokens: 10, cachedReadTokens: 800, totalTokens: 1010 },
      "root",
    );
    await create();
    await instance.adapter.sendTurn({ threadId: "t-usage-cache-inclusive", text: "go" });
    await recorder.until((e) => e.type === "turn.completed");

    const usage = recorder.events.find((e) => e.type === "thread.token-usage.updated");
    expect(usage).toMatchObject({ input: 1000, output: 10, cachedInput: 800 });
  });

  it("emits no cachedInput when the result reports no cache reads", async () => {
    stubPromptResultUsage({ inputTokens: 900, outputTokens: 20, totalTokens: 920 }, "root");
    await create();
    await instance.adapter.sendTurn({ threadId: "t-usage-cache-absent", text: "go" });
    await recorder.until((e) => e.type === "turn.completed");

    const usage = recorder.events.find((e) => e.type === "thread.token-usage.updated");
    expect(usage).toMatchObject({ input: 900, output: 20 });
    expect(usage && "cachedInput" in usage).toBe(false);
  });

  it("reads cache reads from _meta usage too", async () => {
    stubPromptResultUsage(
      { inputTokens: 5030, outputTokens: 3, cachedReadTokens: 152960, totalTokens: 157993 },
      "meta",
    );
    await create();
    await instance.adapter.sendTurn({ threadId: "t-usage-cache-meta", text: "go" });
    await recorder.until((e) => e.type === "turn.completed");

    const usage = recorder.events.find((e) => e.type === "thread.token-usage.updated");
    expect(usage).toMatchObject({ input: 157990, output: 3, cachedInput: 152960 });
  });

  it("passes ACP stdio flags and strips foreign provider keys from the child env", async () => {
    await create();
    const dump = join(scratch, "dump.json");
    process.env.FAKE_ACP_DUMP = dump;
    process.env.XAI_API_KEY = "xai-should-not-leak";
    process.env.OPENCODE_API_KEY = "opencode-should-not-leak";
    process.env.CLINE_API_KEY = "cline-should-not-leak";
    process.env.CURSOR_API_KEY = "cursor-should-not-leak";
    process.env.CURSOR_AUTH_TOKEN = "cursor-token-should-not-leak";
    // workspace credentials with no CLI consumer at all — held by the
    // harness (env-injected at boot by the desktop shell), used in-process
    process.env.BOX_TOKEN = "box-should-not-leak";
    process.env.OMB_TTS_KEY = "tts-should-not-leak";
    Object.assign(process.env, CONTROL_PLANE_FIXTURE);

    await instance.adapter.sendTurn({ threadId: "t-hygiene", text: "go" });
    await recorder.until((e) => e.type === "turn.completed");

    const seen = JSON.parse(readFileSync(dump, "utf8"));
    expect(seen.argv).toEqual(["acp"]);
    expect(seen.env.XAI_API_KEY).toBeUndefined();
    expect(seen.env.OPENCODE_API_KEY).toBeUndefined();
    expect(seen.env.CLINE_API_KEY).toBeUndefined();
    expect(seen.env.CURSOR_API_KEY).toBeUndefined();
    expect(seen.env.CURSOR_AUTH_TOKEN).toBeUndefined();
    expect(seen.env.BOX_TOKEN).toBeUndefined();
    expect(seen.env.OMB_TTS_KEY).toBeUndefined();
    for (const name of Object.keys(CONTROL_PLANE_FIXTURE)) expect(seen.env[name]).toBeUndefined();
  });

  // ACP session/new accepts stdio MCP entries, so connected apps use the
  // same harness-owned bridge as Claude and Codex.
  it("mounts connected apps as a stdio MCP server", async () => {
    await create();
    const dump = join(scratch, "composio.json");
    process.env.FAKE_ACP_DUMP = dump;
    expect(instance.adapter.capabilities.composioMcp).toBe(true);
    await instance.adapter.sendTurn({
      threadId: "t-composio",
      text: "go",
      integrations: {
        composio: {
          command: process.execPath,
          args: ["/tmp/connector-proxy.js"],
          env: { OMB_CONNECTOR_UPSTREAM_URL: "http://127.0.0.1:8799/api/internal/connectors/mcp" },
        },
      },
    });
    await recorder.until((event) => event.type === "turn.completed");
    expect(JSON.parse(readFileSync(`${dump}.mcp.json`, "utf8"))).toContainEqual({
      name: "composio",
      command: process.execPath,
      args: ["/tmp/connector-proxy.js"],
      env: [{ name: "OMB_CONNECTOR_UPSTREAM_URL", value: "http://127.0.0.1:8799/api/internal/connectors/mcp" }],
    });
  });

  it.each([false, true])("mounts Data through the existing stdio transport unless excluded (excluded=%s)", async (excluded) => {
    await create();
    const dump = join(scratch, "data.json");
    process.env.FAKE_ACP_DUMP = dump;
    process.env.FAKE_ACP_DUMP_PROMPT = "1";
    const data = { command: process.execPath, args: [join(scratch, "data-proxy.mjs")], env: { OMB_DATA_TOKEN: "synthetic-turn-capability" } };
    expect(instance.adapter.capabilities.dataMcp).toBe(true);
    await instance.adapter.sendTurn({
      threadId: "t-data", text: "Update the existing result.", system: BUILT_IN_DATA_SYSTEM_PROMPT,
      integrations: { data }, ...(excluded ? { toolScope: { allow: ["native:*"] } } : {}),
    });
    expect(await recorder.until((event) => event.type === "turn.completed")).toMatchObject({ ok: true });
    expect(JSON.parse(readFileSync(`${dump}.mcp.json`, "utf8"))).toEqual(excluded ? [] : [{
      name: "data", ...data, env: [{ name: "OMB_DATA_TOKEN", value: "synthetic-turn-capability" }],
    }]);
    const prompt = JSON.parse(readFileSync(`${dump}.prompt.json`, "utf8")) as Array<{ type: string; text?: string }>;
    expect(prompt.find(block => block.type === "text")?.text).toContain(BUILT_IN_DATA_SYSTEM_PROMPT.trim());
  });

  it("mounts local CUA only on an approval-capable ACP instance", async () => {
    await create();
    const dump = join(scratch, "local-dump.json");
    process.env.FAKE_ACP_DUMP = dump;
    await instance.adapter.sendTurn({
      threadId: "t-local",
      text: "inspect",
      integrations: {
        localComputer: {
          command: "/opt/cua driver/cua-driver",
          args: ["mcp", "--embedded", "--socket", "/run/user/1000/driver.sock"],
          env: { CUA_DRIVER_EMBEDDED: "1" },
          platform: "linux",
          generation: "generation-1",
          scope: "local-computer",
        },
      },
    });
    await recorder.until((event) => event.type === "turn.completed");
    const seen = JSON.parse(readFileSync(dump, "utf8"));
    expect(seen.mcpServers).toContainEqual({
      name: "computer",
      command: "/opt/cua driver/cua-driver",
      args: ["mcp", "--embedded", "--socket", "/run/user/1000/driver.sock"],
      env: [{ name: "CUA_DRIVER_EMBEDDED", value: "1" }],
    });
    expect(instance.adapter.capabilities.localComputerMcp).toBe(true);
  });

  it("mounts user-configured custom MCP servers after the built-ins", async () => {
    await create();
    const dump = join(scratch, "custom-dump.json");
    process.env.FAKE_ACP_DUMP = dump;
    await instance.adapter.sendTurn({
      threadId: "t-custom-mcp",
      text: "go",
      integrations: {
        custom: {
          notes: { command: "npx", args: ["-y", "@x/notes-mcp"], env: { NOTES_TOKEN: "tok-1" } },
        },
      },
    });
    await recorder.until((event) => event.type === "turn.completed");
    const seen = JSON.parse(readFileSync(dump, "utf8"));
    expect(seen.mcpServers).toContainEqual({
      name: "notes",
      command: "npx",
      args: ["-y", "@x/notes-mcp"],
      env: [{ name: "NOTES_TOKEN", value: "tok-1" }],
    });
    expect(instance.adapter.capabilities.customMcp).toBe(true);
  });

  const remoteServers = {
    docs: { type: "http" as const, url: "https://docs.example/mcp", headers: { Authorization: "Bearer tok-docs" } },
    legacy: { type: "sse" as const, url: "https://old.example/sse", headers: {} },
    notes: { command: "npx", args: [], env: {} },
  };

  /** The settings a remote-proxy mount carries in its env (mcp-gate-config.ts). */
  const proxied = (server: { command?: string; args?: string[]; env?: Array<{ name: string; value: string }> }) => {
    const value = server.env?.find((entry) => entry.name === "OMB_REMOTE_MCP_SERVER")?.value;
    return value === undefined ? undefined : JSON.parse(value);
  };

  it.each([
    ["advertises no remote transport", undefined],
    ["advertises http and sse", "http,sse"],
  ])("mounts every url server as OpenMausBot's connector for an agent that %s", async (_name, transports) => {
    if (transports) process.env.FAKE_ACP_MCP_TRANSPORTS = transports;
    try {
      await create();
      const dump = join(scratch, "remote-mcp.json");
      process.env.FAKE_ACP_DUMP = dump;
      await instance.adapter.sendTurn({ threadId: "t-remote-mcp", text: "go", integrations: { custom: remoteServers } });
      await recorder.until((event) => event.type === "turn.completed");
      const seen = JSON.parse(readFileSync(dump, "utf8"));
      expect(seen.mcpServers.map((server: { name: string }) => server.name)).toEqual(["docs", "legacy", "notes"]);
      // never ACP's http/sse shape, so the agent's own client never connects
      expect(seen.mcpServers.some((server: object) => "url" in server || "type" in server)).toBe(false);
      const docs = seen.mcpServers.find((server: { name: string }) => server.name === "docs");
      expect(docs.args.join(" ")).toContain("mcp-remote-proxy");
      expect(docs.args.join(" ")).not.toContain("tok-docs");
      expect(proxied(docs)).toEqual(remoteServers.docs);
      // the agent searches tools itself: the catalog passes through
      expect(docs.env.some((entry: { name: string }) => entry.name === "OMB_REMOTE_MCP_DIRECTORY")).toBe(false);
      expect(proxied(seen.mcpServers.find((server: { name: string }) => server.name === "legacy"))).toEqual(remoteServers.legacy);
    } finally {
      delete process.env.FAKE_ACP_MCP_TRANSPORTS;
    }
  });

  // A server that deserializes initialize strictly (a Voluum server:
  // "Unrecognized field 'schemaValidation'") refuses an agent handshake that
  // sends capabilities.extensions. The fake agent connects each mounted
  // server with that handshake, natively if it were handed a URL.
  it("lists a strict url server's tools through OpenMausBot's connector, even for an agent that speaks http", async () => {
    const strict = await startFakeHttpMcp({ strictInitialize: "http-400", strictSchema: "2025-11-25",
      tools: [{ name: "report", inputSchema: { type: "object" } }, { name: "campaigns", inputSchema: { type: "object" } }],
      requireHeader: { name: "Authorization", value: "Bearer tok-voluum" } });
    process.env.FAKE_ACP_MCP_TRANSPORTS = "http,sse";
    try {
      await create(KimiAgentDriver, "mcp-tools");
      await instance.adapter.sendTurn({ threadId: "t-strict-acp", text: "go", integrations: { custom: {
        voluum: { type: "http", url: strict.url, headers: { Authorization: "Bearer tok-voluum" } },
      } } });
      expect(await recorder.until((event) => event.type === "turn.completed")).toMatchObject({ ok: true });
      const text = recorder.events.find((event) => event.type === "item.completed" && (event as { itemType?: string }).itemType === "assistant_text");
      expect((text as { text?: string } | undefined)?.text).toBe("voluum: report,campaigns");
      expect(strict.initializes).toEqual([{ protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "OpenMausBot tool proxy", version: "1" } }]);
    } finally {
      delete process.env.FAKE_ACP_MCP_TRANSPORTS;
      await strict.close();
    }
  });

  it("keeps a tool selection's gate in front of the connector", async () => {
    const strict = await startFakeHttpMcp({ strictInitialize: "http-400", tools: [{ name: "report", inputSchema: { type: "object" } }, { name: "campaigns", inputSchema: { type: "object" } }] });
    try {
      await create(KimiAgentDriver, "mcp-tools");
      await instance.adapter.sendTurn({ threadId: "t-strict-acp-scoped", text: "go", toolScope: { allow: ["native:*", "mcp:voluum:report"] },
        integrations: { custom: { voluum: { type: "http", url: strict.url, headers: {} } } } });
      expect(await recorder.until((event) => event.type === "turn.completed")).toMatchObject({ ok: true });
      const text = recorder.events.find((event) => event.type === "item.completed" && (event as { itemType?: string }).itemType === "assistant_text");
      expect((text as { text?: string } | undefined)?.text).toBe("voluum: report");
      expect(strict.initializes).toHaveLength(1);
    } finally {
      await strict.close();
    }
  });

  it("surfaces a permission ask as request.opened and completes once allowed", async () => {
    await create(KimiAgentDriver, "permission");
    await instance.adapter.sendTurn({
      threadId: "t-perm",
      text: "go",
      cwd: scratch,
      integrations: {
        localComputer: {
          command: "/cua-driver",
          args: ["mcp"],
          env: {},
          platform: "linux",
          scope: "local-computer",
        },
      },
    });
    const opened = await recorder.until((e) => e.type === "request.opened");
    expect(opened).toMatchObject({
      requestType: "permission",
      tool: "shell",
      approvalScope: "local-computer",
      command: { command: "echo hi", cwd: realpathSync(scratch) },
    });

    expect(await instance.adapter.respondToRequest("t-perm", (opened as any).requestId, { behavior: "allow" })).toBe("allowed-once");
    const resolved = await recorder.until((e) => e.type === "request.resolved");
    expect(resolved).toMatchObject({
      behavior: "allow",
      source: "user",
      approvalScope: "local-computer",
    });
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: true });
  });

  it.each([
    { name: "full raw command", toolCall: { kind: "execute", rawInput: { command: `printf '  ${"complete input ".repeat(30)}'\n  pwd  ` } }, descriptor: true },
    { name: "title without raw input", toolCall: { kind: "execute", title: "echo display only" }, descriptor: false },
    { name: "argv raw input", toolCall: { kind: "execute", rawInput: { command: ["echo", "do not join argv"] } }, descriptor: false },
    { name: "MCP tool", toolCall: { kind: "other", title: "mcp__example__run", rawInput: { command: "echo not a shell approval" } }, descriptor: false },
    { name: "MCP execute tool", toolCall: { kind: "execute", title: "mcp__example__run", rawInput: { command: "echo not a native shell approval" } }, descriptor: false },
    { name: "question", toolCall: { toolCallId: "interaction_command", kind: "execute", rawInput: { command: "echo not a shell approval" } }, descriptor: false },
    { name: "relative cwd", toolCall: { kind: "execute", rawInput: { command: "pwd", cwd: "unknown-relative-directory" } }, descriptor: false },
  ])("keeps command grants scoped to complete shell input: $name", async ({ toolCall, descriptor }) => {
    process.env.FAKE_ACP_PERMISSION_TOOL_CALL = JSON.stringify(toolCall);
    await create(KimiAgentDriver, "permission");
    await instance.adapter.sendTurn({ threadId: "t-acp-command-descriptor", text: "go", cwd: scratch });
    const opened = await recorder.until((event) => event.type === "request.opened");
    expect(opened).toHaveProperty("command", descriptor ? { command: toolCall.rawInput?.command, cwd: realpathSync(scratch) } : undefined);
    await instance.adapter.respondToRequest("t-acp-command-descriptor", opened.requestId!, { behavior: "deny" });
    await recorder.until((event) => event.type === "turn.completed");
  });

  it("uses an explicit ACP shell directory and rejects conflicting directory metadata", async () => {
    const command = "pwd";
    const directory = join(scratch, "execution-directory");
    process.env.FAKE_ACP_PERMISSION_TOOL_CALL = JSON.stringify({ kind: "execute", rawInput: { command, workdir: directory } });
    await create(KimiAgentDriver, "permission");
    await instance.adapter.sendTurn({ threadId: "t-acp-explicit-cwd", text: "go", cwd: scratch });
    const opened = await recorder.until((event) => event.type === "request.opened");
    expect(opened).toHaveProperty("command", { command, cwd: directory });
    await instance.adapter.respondToRequest("t-acp-explicit-cwd", opened.requestId!, { behavior: "deny" });
    await recorder.until((event) => event.type === "turn.completed");

    // A separate process observes a new fake provider payload.
    await instance.dispose();
    recorder.stop();
    process.env.FAKE_ACP_PERMISSION_TOOL_CALL = JSON.stringify({ kind: "execute", rawInput: { command, cwd: scratch, workdir: directory } });
    await create(KimiAgentDriver, "permission");
    await instance.adapter.sendTurn({ threadId: "t-acp-conflicting-cwd", text: "go", cwd: scratch });
    const conflict = await recorder.until((event) => event.type === "request.opened");
    expect(conflict).toHaveProperty("command", undefined);
    await instance.adapter.respondToRequest("t-acp-conflicting-cwd", conflict.requestId!, { behavior: "deny" });
    await recorder.until((event) => event.type === "turn.completed");
  });

  it("emits a structured question beside the flat choices", async () => {
    await create(KimiAgentDriver, "question");
    await instance.adapter.sendTurn({ threadId: "t-question-shape", text: "go" });
    const opened = await recorder.until((e) => e.type === "request.opened");
    expect(opened).toMatchObject({
      requestType: "question",
      summary: "Which color?",
      choices: ["Blue", "Green"],
      questions: [{ question: "Which color?", options: [{ label: "Blue" }, { label: "Green" }] }],
    });
  });

  it("answers a structured card reply by recovering the picked option", async () => {
    const answer = join(scratch, "question-answer.txt");
    process.env.FAKE_ACP_PERMISSION_ANSWER = answer;
    await create(KimiAgentDriver, "question");
    await instance.adapter.sendTurn({ threadId: "t-question-answer", text: "go" });
    const opened = await recorder.until((e) => e.type === "request.opened");
    // Exactly what the tabbed QuestionCard submits: one Q:/A: block for the
    // single question, not a bare option label.
    const outcome = await instance.adapter.respondToRequest("t-question-answer", (opened as { requestId: string }).requestId, {
      behavior: "answer",
      message: "The user answered your questions.\n\nQ: Which color?\nA: Green",
    });
    expect(outcome).toBe("answered");
    const resolved = await recorder.until((e) => e.type === "request.resolved");
    expect(resolved).toMatchObject({ behavior: "answer", source: "user" });
    await recorder.until((e) => e.type === "turn.completed");
    expect(readFileSync(answer, "utf8")).toBe("green-id");
  });

  it("cancels a question whose answer matches no offered option", async () => {
    const answer = join(scratch, "question-cancel.txt");
    process.env.FAKE_ACP_PERMISSION_ANSWER = answer;
    await create(KimiAgentDriver, "question");
    await instance.adapter.sendTurn({ threadId: "t-question-cancel", text: "go" });
    const opened = await recorder.until((e) => e.type === "request.opened");
    await instance.adapter.respondToRequest("t-question-cancel", (opened as { requestId: string }).requestId, {
      behavior: "answer",
      message: "Purple",
    });
    const resolved = await recorder.until((e) => e.type === "request.resolved");
    expect(resolved).toMatchObject({ behavior: "deny", source: "system" });
    const error = recorder.events.find((e) => e.type === "runtime.error") as { message: string } | undefined;
    expect(error?.message).toContain("isn't one of the choices");
    expect(error?.message).not.toContain("Agent");
    await recorder.until((e) => e.type === "turn.completed");
    expect(readFileSync(answer, "utf8")).toBe("cancelled");
  });

  it("marks an ACP question as options-only, so the card offers no free text", async () => {
    await create(KimiAgentDriver, "question");
    await instance.adapter.sendTurn({ threadId: "t-question-custom", text: "go" });
    const opened = await recorder.until((e) => e.type === "request.opened");
    expect(opened).toMatchObject({ questions: [{ question: "Which color?", custom: false }] });
  });

  // Option shape with own ids/names: after a "cancelled" outcome the agent
  // waits for session/cancel instead of going on.
  it.each([
    { name: "a typed answer", response: { behavior: "answer", message: "The user answered your questions.\n\nQ: Which article?\nA: my own article" }, error: true },
    { name: "a closed question", response: { behavior: "deny" }, error: false },
  ])("ends the turn instead of hanging after $name matches no option", async ({ response, error }) => {
    process.env.FAKE_ACP_HOLD_ON_CANCELLED = "1";
    process.env.FAKE_ACP_QUESTION_OPTIONS = JSON.stringify([
      { optionId: "opt-0", kind: "allow_once", name: "Machines of Loving Grace (Recommended)" },
      { optionId: "opt-1", kind: "allow_once", name: "A new piece on reasoning models" },
    ]);
    const answer = join(scratch, "question-hold.txt");
    process.env.FAKE_ACP_PERMISSION_ANSWER = answer;
    await create(KimiAgentDriver, "question");
    await instance.adapter.sendTurn({ threadId: "t-question-hold", text: "go" });
    const opened = await recorder.until((e) => e.type === "request.opened");
    await instance.adapter.respondToRequest("t-question-hold", (opened as { requestId: string }).requestId, response as never);
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ stopReason: "cancelled" });
    expect(readFileSync(answer, "utf8")).toBe("cancelled");
    expect(recorder.events.some((e) => e.type === "runtime.error")).toBe(error);
  });

  it("still sends the picked option id on an option shape with ids and names", async () => {
    process.env.FAKE_ACP_HOLD_ON_CANCELLED = "1";
    process.env.FAKE_ACP_QUESTION_OPTIONS = JSON.stringify([
      { optionId: "opt-0", kind: "allow_once", name: "Machines of Loving Grace (Recommended)" },
      { optionId: "opt-1", kind: "allow_once", name: "A new piece on reasoning models" },
    ]);
    const answer = join(scratch, "question-pick.txt");
    process.env.FAKE_ACP_PERMISSION_ANSWER = answer;
    await create(KimiAgentDriver, "question");
    await instance.adapter.sendTurn({ threadId: "t-question-pick", text: "go" });
    const opened = await recorder.until((e) => e.type === "request.opened");
    expect(await instance.adapter.respondToRequest("t-question-pick", (opened as { requestId: string }).requestId, {
      behavior: "answer",
      message: "The user answered your questions.\n\nQ: Which color?\nA: A new piece on reasoning models",
    })).toBe("answered");
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: true });
    expect(readFileSync(answer, "utf8")).toBe("opt-1");
  });

  it.each([false, true])("maps capped labels back to their option id, refusing collisions (%s)", async collision => {
    const label = "Green ".repeat(30);
    const answer = join(scratch, "long-question-answer.txt");
    process.env.FAKE_ACP_PERMISSION_ANSWER = answer;
    process.env.FAKE_ACP_QUESTION_OPTIONS = JSON.stringify([
      { optionId: "green-id", kind: "allow_once", name: label },
      { optionId: "other-id", kind: "allow_once", name: collision ? label + "other" : "Blue" },
    ]);
    await create(KimiAgentDriver, "question");
    await instance.adapter.sendTurn({ threadId: "t-long-question", text: "go" });
    const opened = await recorder.until(e => e.type === "request.opened");
    expect(opened).toMatchObject({ choices: expect.arrayContaining([label.trim().slice(0, 120).trim()]) });
    await instance.adapter.respondToRequest("t-long-question", (opened as { requestId: string }).requestId, {
      behavior: "answer",
      message: `The user answered your questions.\n\nQ: Which color?\nA: ${label.trim().slice(0, 120)}`,
    });
    await recorder.until(e => e.type === "turn.completed");
    expect(readFileSync(answer, "utf8")).toBe(collision ? "cancelled" : "green-id");
  });


  it.each([false, true])("returns rejected when an allow has no native permission option (always: %s)", async (always) => {
    process.env.FAKE_ACP_PERMISSION_OPTIONS = JSON.stringify([{ optionId: "reject", kind: "reject_once" }]);
    const answer = join(scratch, "reject-only-answer.txt");
    process.env.FAKE_ACP_PERMISSION_ANSWER = answer;
    await create(KimiAgentDriver, "permission");
    const threadId = "t-reject-only";
    await instance.adapter.sendTurn({ threadId, text: "go", approvalMode: "ask" });
    const opened = await recorder.until((event) => event.type === "request.opened");
    expect(await instance.adapter.respondToRequest(threadId, "unknown-request", { behavior: "allow" })).toBe("unavailable");
    expect(await instance.adapter.respondToRequest(threadId, opened.requestId!, { behavior: "allow", always })).toBe("rejected");
    const resolved = await recorder.until((event) => event.type === "request.resolved" && event.requestId === opened.requestId);
    expect(resolved).toMatchObject({ behavior: "deny", source: "system" });
    await recorder.until((event) => event.type === "turn.completed");
    expect(readFileSync(answer, "utf8")).toBe("cancelled");
    expect(await instance.adapter.respondToRequest(threadId, opened.requestId!, { behavior: "allow" })).toBe("unavailable");
  });

  it("hands 'Always allow this session' to the agent's own allow_always option", async () => {
    process.env.FAKE_ACP_ALLOW_ALWAYS = "1";
    const answer = join(scratch, "permission-answer.txt");
    process.env.FAKE_ACP_PERMISSION_ANSWER = answer;
    await create(KimiAgentDriver, "permission");
    await instance.adapter.sendTurn({ threadId: "t-always-native", text: "go", approvalMode: "ask" });
    const opened = await recorder.until((e) => e.type === "request.opened");
    // the card may offer it: the driver can honor a session-wide allow
    expect(opened).toHaveProperty("allowSession", true);
    await instance.adapter.respondToRequest("t-always-native", (opened as { requestId: string }).requestId, { behavior: "allow", always: true });
    await recorder.until((e) => e.type === "turn.completed");
    expect(readFileSync(answer, "utf8")).toBe("allow-always");
  });

  it("remembers the exact operation for the session when the agent offers no allow_always", async () => {
    // When the agent omits allow_always, the driver answers allow_once
    // and repeats the person's answer for that exact operation on the
    // resumed session — and only that operation, and only that session.
    const answer = join(scratch, "permission-answer-once.txt");
    process.env.FAKE_ACP_PERMISSION_ANSWER = answer;
    await create(KimiAgentDriver, "permission");
    await instance.adapter.sendTurn({ threadId: "t-always-memory", text: "go", approvalMode: "ask" });
    const opened = await recorder.until((e) => e.type === "request.opened");
    await instance.adapter.respondToRequest("t-always-memory", (opened as { requestId: string }).requestId, { behavior: "allow", always: true });
    await recorder.until((e) => e.type === "turn.completed");
    expect(readFileSync(answer, "utf8")).toBe("allow-once");

    // same operation on the resumed session: answered without a card
    const second = await instance.adapter.sendTurn({ threadId: "t-always-memory", text: "again", approvalMode: "ask", resumeCursor: "fake-acp-session" });
    await recorder.until((e) => e.type === "turn.completed" && e.turnId === second.turnId);
    expect(recorder.events.filter((e) => e.type === "request.opened")).toHaveLength(1);
    expect(readFileSync(answer, "utf8")).toBe("allow-once");

    // a fresh native session forgets it
    // the pool keeps the native session alive between turns; stopAll closes it so the third turn really is a fresh native session
    await instance.adapter.stopAll();
    const third = await instance.adapter.sendTurn({ threadId: "t-always-memory", text: "fresh", approvalMode: "ask" });
    const reopened = await recorder.until((e) => e.type === "request.opened" && e.turnId === third.turnId);
    await instance.adapter.respondToRequest("t-always-memory", (reopened as { requestId: string }).requestId, { behavior: "deny" });
    await recorder.until((e) => e.type === "turn.completed" && e.turnId === third.turnId);
    expect(recorder.events.filter((e) => e.type === "request.opened")).toHaveLength(2);
  });

  it("rejects a second turn while one is in flight", async () => {
    await create(KimiAgentDriver, "hang");
    await instance.adapter.sendTurn({ threadId: "t-busy", text: "one" });
    await recorder.until((e) => e.type === "session.started");
    await expect(instance.adapter.sendTurn({ threadId: "t-busy", text: "two" })).rejects.toThrow(/already running/);
    await instance.adapter.interruptTurn("t-busy");
    await recorder.until((e) => e.type === "turn.completed");
  });

  it("interrupt settles a hung turn as cancelled", async () => {
    await create(KimiAgentDriver, "hang");
    await instance.adapter.sendTurn({ threadId: "t-int", text: "go" });
    await recorder.until((e) => e.type === "session.started");
    await instance.adapter.interruptTurn("t-int");
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ type: "turn.completed" });
  });

  it("an exit before result becomes runtime.error + failed turn", async () => {
    await create(KimiAgentDriver, "exit-early");
    await instance.adapter.sendTurn({ threadId: "t-crash", text: "go" });
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: false });
    expect(recorder.events.some((e) => e.type === "runtime.error")).toBe(true);
  });

  it("does not expire an agent while a person is answering an approval", async () => {
    process.env.OPENMAUS_ACP_PROMPT_IDLE_TIMEOUT_MS = "150";
    await create(KimiAgentDriver, "permission");
    await instance.adapter.sendTurn({ threadId: "t-idle-approval", text: "go", approvalMode: "ask" });
    const opened = await recorder.until(e => e.type === "request.opened");
    await new Promise(resolve => setTimeout(resolve, 450));
    expect(recorder.events.some(e => e.type === "turn.completed")).toBe(false);
    await instance.adapter.respondToRequest("t-idle-approval", (opened as { requestId: string }).requestId, { behavior: "allow" });
    expect(await recorder.until(e => e.type === "turn.completed")).toMatchObject({ ok: true });
    await new Promise(resolve => setTimeout(resolve, 300));
    expect(recorder.events.some(e => e.type === "runtime.error")).toBe(false);
  });

  // MOCA-260: a quiet `sleep` or build sends nothing while it runs, and the
  // guard used to stop the turn as if the agent had hung.
  it.each([KimiAgentDriver, QwenAgentDriver])("does not expire $driverKind while a tool it started is still running", async (driver) => {
    process.env.OPENMAUS_ACP_PROMPT_IDLE_TIMEOUT_MS = "150";
    process.env.FAKE_ACP_TOOL_MS = "600";
    await create(driver, "slow-tool");
    await instance.adapter.sendTurn({ threadId: "t-slow-tool", text: "go" });
    expect(await recorder.until(e => e.type === "turn.completed")).toMatchObject({ ok: true });
    expect(recorder.events.some(e => e.type === "runtime.error")).toBe(false);
    expect(recorder.events.find(e => e.type === "item.completed" && e.itemType === "tool")).toMatchObject({ ok: true });
  });

  it.each([KimiAgentDriver, QwenAgentDriver])("still fails $driverKind when it goes silent once its tool has finished", async (driver) => {
    process.env.OPENMAUS_ACP_PROMPT_IDLE_TIMEOUT_MS = "150";
    await create(driver, "stall-after-tool");
    await instance.adapter.sendTurn({ threadId: "t-stall-tool", text: "go" });
    expect(await recorder.until(e => e.type === "turn.completed")).toMatchObject({ ok: false, stopReason: "rpc_error" });
    expect(recorder.events.find(e => e.type === "runtime.error")?.message).toMatch(/no tool running/i);
    expect(instance.adapter.hasSession("t-stall-tool")).toBe(false);
  });

  // A quiet agent is not a black box: the person is told what it is doing.
  it("tells the person what a quiet agent is doing, then finishes the turn", async () => {
    // Test the driver's notices, not how fast this OS starts ps/PowerShell.
    // The process probe has its own checks in quiet-status.test.ts.
    vi.spyOn(quietStatus, "sampleProcessTree").mockResolvedValue({ cpuMs: 0, connections: 1 });
    process.env.OMB_ACP_QUIET_NOTICE_MS = "150";
    process.env.OMB_ACP_QUIET_TICK_MS = "50";
    process.env.FAKE_ACP_QUIET_MS = "700";
    await create(KimiAgentDriver, "quiet-then-answer");
    await instance.adapter.sendTurn({ threadId: "t-quiet", text: "go" });
    expect(await recorder.until(e => e.type === "turn.completed")).toMatchObject({ ok: true });
    const notices = recorder.events.filter(e => e.type === "runtime.notice") as Array<{ message: string }>;
    expect(notices.length).toBeGreaterThanOrEqual(1);
    expect(notices[0].message).toMatch(/^Kimi (is waiting on its model|is busy|has sent nothing)/);
    expect(recorder.events.some(e => e.type === "runtime.error")).toBe(false);
  });

  it("reads Qwen's debug log: a logged rate-limit retry is reported and keeps the turn alive", async () => {
    vi.spyOn(quietStatus, "sampleProcessTree").mockResolvedValue({ cpuMs: 0, connections: 1 });
    process.env.QWEN_HOME = scratch;
    process.env.OPENMAUS_ACP_PROMPT_IDLE_TIMEOUT_MS = "400";
    process.env.OMB_ACP_QUIET_NOTICE_MS = "150";
    process.env.OMB_ACP_QUIET_TICK_MS = "50";
    process.env.FAKE_ACP_QUIET_MS = "1200";
    process.env.FAKE_ACP_QWEN_LOG = JSON.stringify([
      "[ERROR] [OPENAI_ERROR] OpenAI API Error: 429 rate limited {",
      "[WARN] [RETRY] Attempt 1 failed with status 429. Retrying after explicit delay of 20000ms... {",
    ]);
    await create(QwenAgentDriver, "quiet-then-answer");
    await instance.adapter.sendTurn({ threadId: "t-qwen-retry", text: "go" });
    // silent on the wire three times longer than the idle limit, yet alive
    expect(await recorder.until(e => e.type === "turn.completed")).toMatchObject({ ok: true });
    const notices = (recorder.events.filter(e => e.type === "runtime.notice") as Array<{ message: string }>).map(n => n.message);
    expect(notices.some(m => /Qwen hit a rate limit \(HTTP 429\) and is retrying \(attempt 2\)\. Next try in 20 s\./.test(m))).toBe(true);
    expect(recorder.events.some(e => e.type === "runtime.error")).toBe(false);
  });

  // Qwen Code goes quiet for minutes while it compresses history, backs off
  // a rate limit (up to 5 min a wait) or waits on one model request (its SDK
  // allows 600 s). At 180 s the guard cut those turns off mid-work.
  it("defaults the prompt idle guard above an agent's longest normal silence", () => {
    expect(DEFAULT_ACP_PROMPT_IDLE_MS).toBeGreaterThan(600_000);
  });

  it("an agent that goes silent mid-answer is failed and closed by the prompt idle guard", async () => {
    process.env.OPENMAUS_ACP_PROMPT_IDLE_TIMEOUT_MS = "150";
    await create(KimiAgentDriver, "stall-after-text");
    await instance.adapter.sendTurn({ threadId: "t-stall", text: "go" });

    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ type: "turn.completed", ok: false, stopReason: "rpc_error" });
    const err = recorder.events.find((e) => e.type === "runtime.error");
    expect(err?.message).toMatch(/no tool running/i);
    expect(err?.message).toContain("OPENMAUS_ACP_PROMPT_IDLE_TIMEOUT_MS");
    // the streamed chunk reached the UI before the child went silent
    expect(recorder.events.some((e) => e.type === "content.delta")).toBe(true);
    expect(instance.adapter.hasSession("t-stall")).toBe(false);
  });

  it("an end_turn with no reply, image, or tool result becomes runtime.error + failed turn", async () => {
    await create(KimiAgentDriver, "empty-reply");
    await instance.adapter.sendTurn({ threadId: "t-empty", text: "go" });
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: false, stopReason: "empty_turn" });
    expect(recorder.events.find((e) => e.type === "runtime.error")?.message)
      .toMatch(/no reply, image, or tool result/);
  });

  it("reasoning with no answer is a lost turn, not a success", async () => {
    // the shape of a provider that never leaves its thinking stream: thought
    // chunks stream, the engine still answers end_turn, and the turn must be
    // reported as failed rather than completed-with-nothing
    await create(KimiAgentDriver, "reasoning-only");
    await instance.adapter.sendTurn({ threadId: "t-reasoning", text: "go" });
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: false, stopReason: "empty_turn" });
    expect(recorder.events.some((e) => e.type === "content.delta" && (e as any).streamKind === "reasoning_text")).toBe(true);
    expect(recorder.events.some((e) => e.type === "item.completed")).toBe(false);
    expect(recorder.events.find((e) => e.type === "runtime.error")?.message)
      .toMatch(/no reply, image, or tool result/);
  });

  it("preserves ACP error codes for provider setup classification", async () => {
    await create(ClassifiedErrorDriver, "auth-required");
    await instance.adapter.sendTurn({ threadId: "t-auth-required", text: "go" });
    const done = await recorder.until((e) => e.type === "turn.completed");

    expect(done).toMatchObject({ ok: false, stopReason: "auth_required" });
    expect(recorder.events.find((e) => e.type === "runtime.error")).toMatchObject({ setup: true });
  });

  it("selectModel confirms the requested model before prompting", async () => {
    process.env.FAKE_ACP_MODELS = "m-one,m-two";
    await create(SelectModelDriver);
    await instance.adapter.sendTurn({ threadId: "t-model", text: "go", model: "m-two" });

    const started = await recorder.until((e) => e.type === "session.started");
    expect(started).toMatchObject({ model: "m-two" });
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: true });
  });

  it("a model the session does not advertise fails the turn instead of running another", async () => {
    process.env.FAKE_ACP_MODELS = "m-one,m-two";
    await create(SelectModelDriver);
    await instance.adapter.sendTurn({ threadId: "t-bad-model", text: "go", model: "m-nope" });

    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: false });
    const err = recorder.events.find((e) => e.type === "runtime.error")!;
    expect(err.message).toMatch(/model not found/);
    // nothing was generated: the prompt is never sent
    expect(recorder.events.some((e) => e.type === "content.delta")).toBe(false);
  });

  // The unadvertised-model test above rides the fake's -32602, so it settles in
  // `request()` and never reaches the guard. This one is the silent case the
  // guard was written for: the agent acknowledges the switch and keeps its old
  // model, which no error surfaces.
  it("a model switch acknowledged but not applied fails the turn", async () => {
    process.env.FAKE_ACP_MODELS = "m-one,m-two";
    process.env.FAKE_ACP_MODEL_STICKS = "1";
    await create(SelectModelDriver);
    await instance.adapter.sendTurn({ threadId: "t-stuck-model", text: "go", model: "m-two" });

    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: false });
    const err = recorder.events.find((e) => e.type === "runtime.error")!;
    expect(err.message).toMatch(/did not switch to m-two \(still m-one\)/);
    // the whole point: no paid turn is spent on the wrong model
    expect(recorder.events.some((e) => e.type === "content.delta")).toBe(false);
  });

  it("selects the model on a resumed session too, not just a new one", async () => {
    process.env.FAKE_ACP_MODELS = "m-one,m-two";
    await create(SelectModelDriver);
    await instance.adapter.sendTurn({
      threadId: "t-resume-model",
      text: "go",
      model: "m-two",
      // deliberately NOT "fake-acp-session", the id session/new returns: with
      // that cursor a session/load that threw and fell back to session/new
      // would emit the same sessionId and this test could not fail
      resumeCursor: "resumed-thread-1",
    });

    // session/load feeds the same sessionResult as session/new, so the model
    // hook must fire on a resumed thread as well
    const started = await recorder.until((e) => e.type === "session.started");
    expect(started).toMatchObject({ sessionId: "resumed-thread-1", model: "m-two" });
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: true });
  });

  it("reuses a resume cursor when session/load returns a session", async () => {
    await create(KimiAgentDriver);
    await instance.adapter.sendTurn({
      threadId: "t-resume-truthy",
      text: "go",
      resumeCursor: "resumed-cursor-1",
    });

    const started = await recorder.until((e) => e.type === "session.started");
    expect(started).toMatchObject({ sessionId: "resumed-cursor-1" });
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: true });
  });

  it("falls through to session/new when session/load returns null", async () => {
    process.env.FAKE_ACP_LOAD_NULL = "1";
    await create(KimiAgentDriver);
    await instance.adapter.sendTurn({
      threadId: "t-resume-null",
      text: "go",
      resumeCursor: "gone-cursor",
    });

    const started = await recorder.until((e) => e.type === "session.started");
    expect(started).toMatchObject({ sessionId: "fake-acp-session" });
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: true });
  });

  it.each(["null", "opencode-not-found"])("rebuilds a missing native session from the canonical recovery text once (%s)", async (failure) => {
    if (failure === "null") process.env.FAKE_ACP_LOAD_NULL = "1";
    else process.env.FAKE_ACP_LOAD_ERROR = JSON.stringify({ code: -32602, message: "Session not found", data: { sessionId: "missing-cursor" } });
    const dump = join(scratch, "recovery.json");
    process.env.FAKE_ACP_DUMP = dump;
    process.env.FAKE_ACP_DUMP_PROMPT = "1";
    await create(KimiAgentDriver);
    const recoveryText = "User: Remember ALPHA.\nAssistant: Remembered.\nUser: What did I say?";
    await instance.adapter.sendTurn({
      threadId: "t-resume-history", text: "What did I say?", resumeCursor: "missing-cursor",
      recoveryText, system: "Keep current bot instructions.",
    });
    expect(await recorder.until((e) => e.type === "turn.completed")).toMatchObject({ ok: true });
    expect(recorder.events.find((e) => e.type === "session.started")).toMatchObject({ rebuilt: true });
    const prompt = JSON.parse(readFileSync(`${dump}.prompt.json`, "utf8"));
    expect(prompt).toEqual([{ type: "text", text: `Keep current bot instructions.\n\n${recoveryText}` }]);
  });

  it.each([
    [-32000, "authentication required", "auth_required"],
    [-32602, "Invalid params", "rpc_error"],
  ])("does not replace native history after a resume refusal (%s)", async (code, message, stopReason) => {
    process.env.FAKE_ACP_LOAD_ERROR = JSON.stringify({ code, message });
    const rpcFile = join(scratch, "refused-load.json");
    process.env.FAKE_ACP_RPC_DUMP = rpcFile;
    const countFile = join(scratch, "launches");
    process.env.FAKE_ACP_LAUNCH_COUNT_FILE = countFile;
    await create(ClassifiedErrorDriver);
    const turn = {
      threadId: "t-load-refused", text: "Continue", resumeCursor: "saved-session", recoveryText: "Prior messages\nContinue",
    };
    await instance.adapter.sendTurn(turn);
    expect(await recorder.until((e) => e.type === "turn.completed")).toMatchObject({ ok: false, stopReason });
    const methods = JSON.parse(readFileSync(rpcFile, "utf8"));
    expect(methods).toContain("session/load");
    expect(methods).not.toContain("session/new");
    expect(methods).not.toContain("session/prompt");
    const retry = await instance.adapter.sendTurn(turn);
    expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === retry.turnId)).toMatchObject({ ok: false, stopReason });
    expect(Number(readFileSync(countFile, "utf8"))).toBe(1);
  });

  it("applyTurnEnv sees the picker model after resolveTurnModel", async () => {
    const dump = join(scratch, "turn-env.json");
    process.env.FAKE_ACP_DUMP = dump;
    const TurnEnvDriver = createAcpDriver({
      ...SELECT_MODEL_SUPPORT,
      driverKind: "turnEnvTest",
      selectModel: undefined,
      resolveTurnModel: (model) => (model ? `resolved/${model}` : model),
      applyTurnEnv: (env, { model, requestedModel }) => {
        env.TEST_TURN_MODEL = `${model ?? ""}|${requestedModel ?? ""}`;
      },
    });
    instance = await TurnEnvDriver.create({
      instanceId: "turn-env-test",
      displayName: undefined,
      environment: {},
      enabled: true,
      config: { cli: FAKE_CLI, fullAuto: false },
    });
    recorder = recordEvents(instance.adapter);

    await instance.adapter.sendTurn({
      threadId: "t-turn-env",
      text: "go",
      model: "ollama::ornith:35b-bf16",
    });
    await recorder.until((e) => e.type === "turn.completed");

    expect(JSON.parse(readFileSync(dump, "utf8")).env.TEST_TURN_MODEL).toBe(
      "resolved/ollama::ornith:35b-bf16|ollama::ornith:35b-bf16",
    );
  });

  it("transformEnv sees the instance config", async () => {
    const dump = join(scratch, "policy.json");
    process.env.FAKE_ACP_DUMP = dump;
    instance = await EnvPolicyDriver.create({
      instanceId: "policy-test",
      displayName: undefined,
      environment: {},
      enabled: true,
      config: { cli: FAKE_CLI, fullAuto: true },
    });
    recorder = recordEvents(instance.adapter);

    await instance.adapter.sendTurn({ threadId: "t-policy", text: "go" });
    await recorder.until((e) => e.type === "turn.completed");

    expect(JSON.parse(readFileSync(dump, "utf8")).env.TEST_POLICY).toBe("auto");
  });

  it("per-bot Ask overrides a legacy full-auto instance for the whole turn", async () => {
    const dump = join(scratch, "ask-overrides-full-auto.json");
    process.env.FAKE_ACP_DUMP = dump;
    instance = await EnvPolicyDriver.create({
      instanceId: "policy-override-test",
      displayName: undefined,
      environment: {},
      enabled: true,
      config: { cli: FAKE_CLI, fullAuto: true },
    });
    recorder = recordEvents(instance.adapter);

    await instance.adapter.sendTurn({
      threadId: "t-policy-override",
      text: "go",
      approvalMode: "ask",
    });
    await recorder.until((e) => e.type === "turn.completed");

    expect(JSON.parse(readFileSync(dump, "utf8")).env.TEST_POLICY).toBe("ask");
  });

  describe("opt-in startup recovery", () => {
    const failStartup = async (message = "503 Service Unavailable") => {
      const failureFile = join(scratch, "startup-failure.json");
      process.env.FAKE_ACP_RPC_FAILURE_FILE = failureFile;
      process.env.FAKE_ACP_RPC_FAILURE_METHOD = "session/new";
      writeFileSync(failureFile, JSON.stringify({ code: -32603, message }));
      await create();
    };

    it("rejects a transient setup failure only after its exact child stops, without terminal events", async () => {
      await failStartup();
      const kill = vi.spyOn(procs, "killCliTree");
      const failed = await instance.adapter.sendTurn({ threadId: "startup", text: "go", startupRecovery: true }).catch((error) => error);
      expect(failed).toBeInstanceOf(TurnNotStartedError);
      expect(failed.turnId).toBe(recorder.events.find((event) => event.type === "turn.started")?.turnId);
      expect(recorder.events.some((event) => event.type === "runtime.error" || event.type === "turn.completed")).toBe(false);
      expect(kill).toHaveBeenCalled();
      expect(await kill.mock.results[0].value).toBe(true);
      const child = kill.mock.calls[0][0];
      expect(child.exitCode !== null || child.signalCode !== null).toBe(true);
      expect(instance.adapter.hasSession("startup")).toBe(false);
    });

    it.each([
      "503 invalid api key", "503 quota exceeded", "503 unknown model", "503 blocked by our safety systems", "unrecognized failure",
      "503 permission denied", "approval required after timeout", "503 policy restriction", "timeout blocked by policy",
    ])("keeps normal completion for %s", async (message) => {
      await failStartup(message);
      const ack = await instance.adapter.sendTurn({ threadId: "startup-terminal", text: "go", startupRecovery: true });
      expect(await recorder.until((event) => event.type === "turn.completed" && event.turnId === ack.turnId)).toMatchObject({ ok: false });
      expect(recorder.events.some((event) => event.type === "runtime.error")).toBe(true);
    });

    it("acknowledges before a prompt timeout and never offers startup recovery afterward", async () => {
      process.env.OPENMAUS_ACP_PROMPT_IDLE_TIMEOUT_MS = "100";
      await create(KimiAgentDriver, "stall-after-text");
      const ack = await instance.adapter.sendTurn({ threadId: "startup-prompt", text: "go", startupRecovery: true });
      expect(recorder.events.some((event) => event.type === "turn.completed")).toBe(false);
      expect(await recorder.until((event) => event.type === "turn.completed" && event.turnId === ack.turnId))
        .toMatchObject({ ok: false, stopReason: "rpc_error" });
    });

    it("resolves the pending ACK when Stop interrupts initialization", async () => {
      await create(KimiAgentDriver, "hang-initialize");
      const pending = instance.adapter.sendTurn({ threadId: "startup-stop", text: "go", startupRecovery: true });
      let acknowledged = false;
      void pending.then(() => { acknowledged = true; });
      await recorder.until((event) => event.type === "turn.started");
      expect(acknowledged).toBe(false);
      await instance.adapter.interruptTurn("startup-stop");
      const ack = await pending;
      await recorder.until((event) => event.type === "turn.completed" && event.turnId === ack.turnId);
      expect(instance.adapter.hasSession("startup-stop")).toBe(false);
    });

    it("honors Stop while confirming cleanup instead of rejecting for recovery", async () => {
      await failStartup();
      const spawn = procs.spawnCli;
      let startupChild: ReturnType<typeof procs.spawnCli>;
      vi.spyOn(procs, "spawnCli").mockImplementation((...args) => startupChild = spawn(...args));
      let release!: () => void;
      let stopping!: () => void;
      const gate = new Promise<void>((resolve) => { release = resolve; });
      const startedStopping = new Promise<void>((resolve) => { stopping = resolve; });
      const kill = procs.killCliTree;
      vi.spyOn(procs, "killCliTree").mockImplementation(async (child, timeout) => {
        // Delayed cleanup from another test must not release this turn's gate.
        if (child === startupChild) { stopping(); await gate; }
        return kill(child, timeout);
      });
      try {
        const pending = instance.adapter.sendTurn({ threadId: "startup-cleanup-stop", text: "go", startupRecovery: true });
        await startedStopping;
        await instance.adapter.interruptTurn("startup-cleanup-stop");
        release();
        const ack = await pending;
        expect(await recorder.until((event) => event.type === "turn.completed" && event.turnId === ack.turnId))
          .toMatchObject({ ok: true, stopReason: "cancelled" });
        expect(recorder.events.some((event) => event.type === "runtime.error")).toBe(false);
      } finally { release(); }
    });

    it("fails normally when process cleanup cannot be confirmed", async () => {
      await failStartup();
      const kill = procs.killCliTree;
      const stop = vi.spyOn(procs, "killCliTree").mockResolvedValue(false);
      try {
        const ack = await instance.adapter.sendTurn({ threadId: "startup-kill-failed", text: "go", startupRecovery: true });
        expect(await recorder.until((event) => event.type === "turn.completed" && event.turnId === ack.turnId)).toMatchObject({ ok: false });
        expect(recorder.events.some((event) => event.type === "runtime.error")).toBe(true);
      } finally {
        await Promise.all(stop.mock.calls.map(([child]) => kill(child, 0)));
      }
    });

    it.each(["request", "output"])("does not recover after pre-prompt %s activity", async (kind) => {
      const spawn = procs.spawnCli;
      vi.spyOn(procs, "spawnCli").mockImplementation((...args) => {
        const child = spawn(...args);
        const write = child.stdin.write.bind(child.stdin);
        vi.spyOn(child.stdin, "write").mockImplementation((...writeArgs) => {
          const message = JSON.parse(String(writeArgs[0]));
          if (message.method === "session/new") {
            const activity = kind === "request"
              ? { jsonrpc: "2.0", id: "before-prompt", method: "fs/read_text_file", params: { path: join(scratch, "absent") } }
              : { jsonrpc: "2.0", method: "session/update", params: { update: { sessionUpdate: "tool_call", toolCallId: "before-prompt" } } };
            child.stdout.emit("data", Buffer.from(`${JSON.stringify(activity)}\n`));
          }
          return write(...writeArgs);
        });
        return child;
      });
      await failStartup();
      const ack = await instance.adapter.sendTurn({ threadId: "startup-activity", text: "go", startupRecovery: true });
      expect(await recorder.until((event) => event.type === "turn.completed" && event.turnId === ack.turnId)).toMatchObject({ ok: false });
    });
  });

  describe("ACP session pool (persistent child)", () => {
    let countFile: string;
    let rpcFile: string;
    const launches = () => Number(readFileSync(countFile, "utf8"));
    const rpc = () => JSON.parse(readFileSync(rpcFile, "utf8")) as string[];

    it("recovers on the next explicit turn after session establishment fails", async () => {
      countFile = join(scratch, "launches");
      const appendFile = join(scratch, "rpc-all.jsonl");
      const failureFile = join(scratch, "failure.json");
      process.env.FAKE_ACP_LAUNCH_COUNT_FILE = countFile;
      process.env.FAKE_ACP_RPC_APPEND_FILE = appendFile;
      process.env.FAKE_ACP_RPC_FAILURE_FILE = failureFile;
      process.env.FAKE_ACP_RPC_FAILURE_METHOD = "session/new";
      writeFileSync(failureFile, JSON.stringify({ code: -32603, message: "Internal error", data: {
        service: "directory", details: "OpenCode service failure", errorName: "Error",
      } }));
      await create();
      const threadId = "t-new-rpc-recovery";
      const first = await instance.adapter.sendTurn({ threadId, text: "Hello" });
      expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === first.turnId)).toMatchObject({ ok: false, stopReason: "rpc_error" });
      const calls = () => readFileSync(appendFile, "utf8").trim().split("\n").map((line) => JSON.parse(line));
      expect(launches()).toBe(1);
      expect(calls().filter((entry) => entry.method === "session/prompt")).toHaveLength(0);
      expect(recorder.events.some((e) => e.type === "session.started")).toBe(false);
      unlinkSync(failureFile);
      const next = await instance.adapter.sendTurn({ threadId, text: "Hello again" });
      expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === next.turnId)).toMatchObject({ ok: true });
      expect(launches()).toBe(2);
      expect(calls().filter((entry) => entry.method === "session/prompt")).toHaveLength(1);
      expect(recorder.events.filter((e) => e.type === "turn.completed" && e.turnId === first.turnId)).toHaveLength(1);
    });

    it.each([false, true])("evicts internal-error processes without replaying a failed prompt (output: %s)", async (afterOutput) => {
      countFile = join(scratch, "launches");
      const appendFile = join(scratch, "rpc-all.jsonl");
      const failureFile = join(scratch, "failure.json");
      process.env.FAKE_ACP_LAUNCH_COUNT_FILE = countFile;
      process.env.FAKE_ACP_RPC_APPEND_FILE = appendFile;
      process.env.FAKE_ACP_RPC_FAILURE_FILE = failureFile;
      if (afterOutput) process.env.FAKE_ACP_RPC_FAILURE_AFTER_OUTPUT = "1";
      await create();
      const threadId = "t-rpc-recovery";
      const first = await instance.adapter.sendTurn({ threadId, text: "Remember ALPHA" });
      expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === first.turnId)).toMatchObject({ ok: true });
      writeFileSync(failureFile, JSON.stringify({ code: -32603, message: "Internal error", data: {
        details: "OpenCode service failure; api_key=fixture-secret",
        service: "session", errorName: "APIError",
        responseBody: "PRIVATE RESPONSE BODY MUST NOT APPEAR",
      } }));
      const failed = await instance.adapter.sendTurn({ threadId, text: "Do work", resumeCursor: "fake-acp-session" });
      expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === failed.turnId)).toMatchObject({ ok: false, stopReason: "rpc_error" });
      expect(launches()).toBe(1);
      expect(instance.adapter.hasSession(threadId)).toBe(false);
      const error = recorder.events.find((e) => e.type === "runtime.error" && e.turnId === failed.turnId);
      const message = error?.type === "runtime.error" ? error.message : "";
      expect(message).toContain("Internal error: OpenCode service failure");
      expect(message).toContain("session/prompt, service: session, APIError");
      expect(message).not.toContain("fixture-secret");
      expect(message).not.toContain("PRIVATE RESPONSE");
      expect(recorder.events.filter((e) => e.type === "item.completed" && e.itemType === "tool" && e.turnId === failed.turnId)).toHaveLength(afterOutput ? 1 : 0);
      unlinkSync(failureFile);
      const next = await instance.adapter.sendTurn({ threadId, text: "Continue after the interruption", resumeCursor: "fake-acp-session" });
      expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === next.turnId)).toMatchObject({ ok: true });
      expect(launches()).toBe(2);
      const calls = readFileSync(appendFile, "utf8").trim().split("\n").map((line) => JSON.parse(line));
      expect(calls.filter((entry) => entry.method === "session/prompt")).toHaveLength(3);
      expect(calls.filter((entry) => entry.method === "session/load")).toHaveLength(1);
      expect(recorder.events.filter((e) => e.type === "turn.completed" && e.turnId === failed.turnId)).toHaveLength(1);
    });

    it("honors explicit session reset even when a healthy child and cursor are retained", async () => {
      countFile = join(scratch, "launches");
      rpcFile = join(scratch, "rpc.json");
      process.env.FAKE_ACP_LAUNCH_COUNT_FILE = countFile;
      process.env.FAKE_ACP_RPC_DUMP = rpcFile;
      await create();
      const first = await instance.adapter.sendTurn({ threadId: "t-reset", text: "one" });
      await recorder.until((e) => e.type === "turn.completed" && e.turnId === first.turnId);
      const second = await instance.adapter.sendTurn({ threadId: "t-reset", text: "rebuilt conversation", sessionReset: true, resumeCursor: "fake-acp-session" });
      expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === second.turnId)).toMatchObject({ ok: true });
      expect(launches()).toBe(2);
      expect(rpc()).toContain("session/new");
      expect(rpc()).not.toContain("session/load");
    });

    it("reuses one agent process across turns and skips the handshake", async () => {
      countFile = join(scratch, "launches");
      rpcFile = join(scratch, "rpc.json");
      process.env.FAKE_ACP_LAUNCH_COUNT_FILE = countFile;
      process.env.FAKE_ACP_RPC_DUMP = rpcFile;
      await create();
      const first = await instance.adapter.sendTurn({ threadId: "t-pool-reuse", text: "one" });
      const firstDone = await recorder.until((e) => e.type === "turn.completed" && e.turnId === first.turnId);
      expect(firstDone).toMatchObject({ ok: true });

      const second = await instance.adapter.sendTurn({
        threadId: "t-pool-reuse",
        text: "two",
        resumeCursor: "fake-acp-session",
      });
      const secondDone = await recorder.until((e) => e.type === "turn.completed" && e.turnId === second.turnId);
      expect(secondDone).toMatchObject({ ok: true });

      // one live child per thread: the second prompt rides the same process
      expect(launches()).toBe(1);
      expect(rpc().filter((m) => m === "initialize")).toHaveLength(1);
      expect(rpc().filter((m) => m === "session/new")).toHaveLength(1);
      expect(rpc().filter((m) => m === "session/load")).toHaveLength(0);
      expect(rpc().filter((m) => m === "session/prompt")).toHaveLength(2);
      expect(recorder.events.filter((e) => e.type === "session.started")).toMatchObject([
        { sessionId: "fake-acp-session" },
        { sessionId: "fake-acp-session" },
      ]);
    });

    it("closes the idle process and resumes on the next turn", async () => {
      // Ten seconds is the lowest window the floor allows now; exercise the
      // close at the floor itself and give the poll room past it.
      process.env.OMB_ACP_SESSION_IDLE_MIN_MS = "10000";
      process.env.OMB_ACP_SESSION_IDLE_MS = "10000";
      countFile = join(scratch, "launches");
      rpcFile = join(scratch, "rpc.json");
      process.env.FAKE_ACP_LAUNCH_COUNT_FILE = countFile;
      process.env.FAKE_ACP_RPC_DUMP = rpcFile;
      await create();
      const first = await instance.adapter.sendTurn({ threadId: "t-pool-idle", text: "one" });
      await recorder.until((e) => e.type === "turn.completed" && e.turnId === first.turnId);
      // the close reason is only logged, never emitted — poll the native log
      // for it rather than sleeping a fixed window past the idle deadline
      await new Promise<void>((resolve, reject) => {
        const deadline = Date.now() + 20_000;
        const log = join(NATIVE_DIR, "t-pool-idle.ndjson");
        const check = () => {
          if (Date.now() > deadline) return reject(new Error("idle close was never logged"));
          try {
            if (readFileSync(log, "utf8").includes('"close":"idle"')) return resolve();
          } catch (error) {
            // appendNative() suppresses append errors, so the file may not exist yet
            if ((error as NodeJS.ErrnoException).code !== "ENOENT") return reject(error);
          }
          setTimeout(check, 25);
        };
        check();
      });
      expect(launches()).toBe(1);

      const second = await instance.adapter.sendTurn({
        threadId: "t-pool-idle",
        text: "two",
        resumeCursor: "fake-acp-session",
      });
      const done = await recorder.until((e) => e.type === "turn.completed" && e.turnId === second.turnId);
      expect(done).toMatchObject({ ok: true });
      expect(launches()).toBe(2);
      // the dump is per-process and overwritten on spawn, so this is the resumed child
      expect(rpc()).toContain("session/load");
      expect(rpc()).toContain("initialize");
    }, 30_000);

    it("respawns when the spawn contract changes", async () => {
      countFile = join(scratch, "launches");
      process.env.FAKE_ACP_LAUNCH_COUNT_FILE = countFile;
      const dirA = mkdtempSync(join(scratch, "a-"));
      const dirB = mkdtempSync(join(scratch, "b-"));
      await create();
      const first = await instance.adapter.sendTurn({ threadId: "t-pool-contract", text: "one", cwd: dirA });
      const firstDone = await recorder.until((e) => e.type === "turn.completed" && e.turnId === first.turnId);
      expect(firstDone).toMatchObject({ ok: true });

      const second = await instance.adapter.sendTurn({
        threadId: "t-pool-contract",
        text: "two",
        cwd: dirB,
        resumeCursor: "fake-acp-session",
      });
      const secondDone = await recorder.until((e) => e.type === "turn.completed" && e.turnId === second.turnId);
      expect(secondDone).toMatchObject({ ok: true });
      expect(launches()).toBe(2);
    });

    it.each([
      { driver: KimiAgentDriver, expectedLaunches: 1 },
      { driver: QwenAgentDriver, expectedLaunches: 2 },
    ])("rotating credentials refreshes $driver.driverKind with $expectedLaunches process(es)", async ({ driver, expectedLaunches }) => {
      countFile = join(scratch, "launches");
      const appendFile = join(scratch, "rpc-all.jsonl");
      process.env.FAKE_ACP_LAUNCH_COUNT_FILE = countFile;
      process.env.FAKE_ACP_RPC_APPEND_FILE = appendFile;
      instance = await driver.create({
        instanceId: "acp-test", displayName: "ACP Test", enabled: true,
        environment: { HOME: scratch, USERPROFILE: scratch },
        config: { cli: FAKE_CLI, fullAuto: false },
      });
      recorder = recordEvents(instance.adapter);
      // Qwen caches MCP clients on live load; other agents apply the new
      // credentials without a process restart. Both preserve the cursor.
      const integration = (token: string) => ({
        command: process.execPath,
        args: [FAKE_CLI],
        env: { OMB_COMMS_TOKEN: token },
      });
      const first = await instance.adapter.sendTurn({
        threadId: "t-pool-token",
        text: "one",
        integrations: { agents: integration("token-one") },
      });
      const firstDone = await recorder.until((e) => e.type === "turn.completed" && e.turnId === first.turnId);
      expect(firstDone).toMatchObject({ ok: true });

      const second = await instance.adapter.sendTurn({
        threadId: "t-pool-token",
        text: "two",
        resumeCursor: "fake-acp-session",
        integrations: { agents: integration("token-two") },
      });
      const secondDone = await recorder.until((e) => e.type === "turn.completed" && e.turnId === second.turnId);
      expect(secondDone).toMatchObject({ ok: true });

      const third = await instance.adapter.sendTurn({
        threadId: "t-pool-token", text: "unchanged MCP inputs",
        resumeCursor: "fake-acp-session",
        integrations: { agents: integration("token-two") },
      });
      expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === third.turnId)).toMatchObject({ ok: true });
      expect(launches()).toBe(expectedLaunches);
      const calls = readFileSync(appendFile, "utf8").trim().split("\n").map((line) => JSON.parse(line).method);
      expect(calls.filter((m) => m === "initialize")).toHaveLength(expectedLaunches);
      expect(calls.filter((m) => m === "session/new")).toHaveLength(1);
      expect(calls.filter((m) => m === "session/load")).toHaveLength(1);
      expect(calls.filter((m) => m === "session/prompt")).toHaveLength(3);
      expect(recorder.events.filter((e) => e.type === "session.started").map((e) => e.sessionId)).toEqual([
        "fake-acp-session", "fake-acp-session", "fake-acp-session",
      ]);
    });

    it("does not load or prompt Qwen after Stop during old-process cleanup", async () => {
      const appendFile = join(scratch, "rpc-all.jsonl");
      process.env.FAKE_ACP_RPC_APPEND_FILE = appendFile;
      instance = await QwenAgentDriver.create({
        instanceId: "qwen-stop", displayName: "Qwen Stop", enabled: true,
        environment: { HOME: scratch, USERPROFILE: scratch },
        config: { cli: FAKE_CLI, fullAuto: false },
      });
      recorder = recordEvents(instance.adapter);
      const integration = (token: string) => ({ command: process.execPath, args: [FAKE_CLI], env: { OMB_COMMS_TOKEN: token } });
      const first = await instance.adapter.sendTurn({ threadId: "qwen-stop", text: "one", integrations: { agents: integration("one") } });
      await recorder.until((e) => e.type === "turn.completed" && e.turnId === first.turnId);

      let stopped!: () => void;
      let release!: () => void;
      const stopping = new Promise<void>((resolve) => { stopped = resolve; });
      const gate = new Promise<void>((resolve) => { release = resolve; });
      const kill = procs.killCliTree;
      vi.spyOn(procs, "killCliTree").mockImplementationOnce(async (child) => {
        stopped();
        await gate;
        return kill(child);
      });
      try {
        const second = await instance.adapter.sendTurn({
          threadId: "qwen-stop", text: "two", resumeCursor: "fake-acp-session",
          integrations: { agents: integration("two") },
        });
        await stopping;
        await instance.adapter.interruptTurn("qwen-stop");
        release();
        expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === second.turnId)).toMatchObject({ ok: false });
        const calls = readFileSync(appendFile, "utf8").trim().split("\n").map((line) => JSON.parse(line).method);
        expect(calls.filter((method) => method === "initialize")).toHaveLength(1);
        expect(calls.filter((method) => method === "session/load")).toHaveLength(0);
        expect(calls.filter((method) => method === "session/prompt")).toHaveLength(1);
        expect(instance.adapter.hasSession("qwen-stop")).toBe(false);
      } finally { release(); }
    });

    it.each(["Stop", "prompt stall", "failed cleanup"])("waits for Qwen cleanup before an immediate retry after %s", async (reason) => {
      const children: ReturnType<typeof procs.spawnCli>[] = [];
      const spawn = procs.spawnCli;
      vi.spyOn(procs, "spawnCli").mockImplementation((...args) => {
        const child = spawn(...args);
        vi.spyOn(child.stdin, "write");
        children.push(child);
        return child;
      });
      const calls = () => children.flatMap((child) => vi.mocked(child.stdin.write).mock.calls
        .map(([chunk]) => JSON.parse(String(chunk)) as { method: string; params: { sessionId?: string } }));
      let release!: () => void;
      let stopping!: () => void;
      const gate = new Promise<void>((resolve) => { release = resolve; });
      const startedStopping = new Promise<void>((resolve) => { stopping = resolve; });
      const kill = procs.killCliTree;
      let rejectCleanup = reason === "failed cleanup";
      vi.spyOn(procs, "killCliTree").mockImplementation(async (child, timeout) => {
        // Hold only the original writer: Stop must still close its uninitialized replacement.
        if (child === children[0]) {
          stopping();
          await gate;
          if (rejectCleanup) return false;
        }
        return kill(child, timeout);
      });
      if (reason === "prompt stall") {
        process.env.OPENMAUS_ACP_PROMPT_IDLE_TIMEOUT_MS = "100";
        process.env.FAKE_ACP_MODE = "stall-after-text";
      }
      instance = await QwenAgentDriver.create({
        instanceId: "qwen-retry", displayName: "Qwen Retry", enabled: true,
        environment: { HOME: scratch, USERPROFILE: scratch },
        config: { cli: FAKE_CLI, fullAuto: false },
      });
      recorder = recordEvents(instance.adapter);
      const integration = (token: string) => ({ agents: { command: process.execPath, args: [FAKE_CLI], env: { OMB_COMMS_TOKEN: token } } });
      try {
        const first = await instance.adapter.sendTurn({ threadId: "qwen-retry", text: "one", integrations: integration("one") });
        expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === first.turnId))
          .toMatchObject({ ok: reason !== "prompt stall" });
        if (reason !== "prompt stall") {
          const second = await instance.adapter.sendTurn({
            threadId: "qwen-retry", text: "two", resumeCursor: "fake-acp-session", integrations: integration("two"),
          });
          await startedStopping;
          if (reason === "Stop") await instance.adapter.interruptTurn("qwen-retry");
          else release();
          expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === second.turnId)).toMatchObject({ ok: false });
          if (rejectCleanup) {
            expect(recorder.events.find((e) => e.type === "runtime.error" && e.turnId === second.turnId))
              .toMatchObject({ message: expect.stringMatching(/could not close its previous tool session/) });
            rejectCleanup = false;
          }
        }
        process.env.FAKE_ACP_MODE = "happy";
        const retry = await instance.adapter.sendTurn({
          threadId: "qwen-retry", text: "retry", resumeCursor: "fake-acp-session", integrations: integration("three"),
        });
        // Observe writes directly: the retry must not even initialize while cleanup is unconfirmed.
        expect(calls().filter((call) => call.method === "initialize")).toHaveLength(1);
        expect(calls().filter((call) => call.method === "session/load")).toHaveLength(0);
        release();
        expect(await recorder.until((e) => e.type === "turn.completed" && e.turnId === retry.turnId)).toMatchObject({ ok: true });
        expect(calls().filter((call) => call.method === "session/new")).toHaveLength(1);
        expect(calls().filter((call) => call.method === "session/load")).toEqual([
          expect.objectContaining({ params: expect.objectContaining({ sessionId: "fake-acp-session" }) }),
        ]);
        expect(calls().filter((call) => call.method === "session/prompt")).toHaveLength(2);
      } finally {
        release();
        await instance.dispose();
        await Promise.all(children.map((child) => kill(child, 0)));
      }
    });

    it("an agent that refuses to re-load its live session gets one fresh process, then resumes", async () => {
      countFile = join(scratch, "launches");
      const appendFile = join(scratch, "rpc-all.jsonl");
      const rejectFile = join(scratch, "reject-live-load");
      process.env.FAKE_ACP_LAUNCH_COUNT_FILE = countFile;
      // the per-process dump would hold only the replacement child's calls;
      // the append log keeps both children's, pid-tagged
      process.env.FAKE_ACP_RPC_APPEND_FILE = appendFile;
      process.env.FAKE_ACP_REJECT_LIVE_LOAD_FILE = rejectFile;
      await create();
      const rpcAll = () =>
        readFileSync(appendFile, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line) as { pid: number; method: string });
      const integration = (token: string) => ({
        command: process.execPath,
        args: [FAKE_CLI],
        env: { OMB_COMMS_TOKEN: token },
      });
      const first = await instance.adapter.sendTurn({
        threadId: "t-pool-reject",
        text: "one",
        integrations: { agents: integration("token-one") },
      });
      const firstDone = await recorder.until((e) => e.type === "turn.completed" && e.turnId === first.turnId);
      expect(firstDone).toMatchObject({ ok: true });

      // from here the fake refuses session/load for a session already live
      // in its own process, the way a real agent can
      writeFileSync(rejectFile, "1");
      const second = await instance.adapter.sendTurn({
        threadId: "t-pool-reject",
        text: "two",
        resumeCursor: "fake-acp-session",
        integrations: { agents: integration("token-two") },
      });
      const secondDone = await recorder.until((e) => e.type === "turn.completed" && e.turnId === second.turnId);
      expect(secondDone).toMatchObject({ ok: true });

      // the pooled child is closed and the recorded session resumes on one
      // fresh process — the conversation is never traded for session/new
      expect(launches()).toBe(2);
      const calls = rpcAll();
      expect(calls.filter((c) => c.method === "initialize")).toHaveLength(2);
      expect(calls.filter((c) => c.method === "session/new")).toHaveLength(1);
      expect(calls.filter((c) => c.method === "session/load")).toHaveLength(2);
      expect(calls.filter((c) => c.method === "session/prompt")).toHaveLength(2);
      // the refused load and the successful one really hit two processes
      expect(new Set(calls.map((c) => c.pid)).size).toBe(2);
    });

    it("stopAll closes the pooled process; the next turn resumes", async () => {
      countFile = join(scratch, "launches");
      rpcFile = join(scratch, "rpc.json");
      process.env.FAKE_ACP_LAUNCH_COUNT_FILE = countFile;
      process.env.FAKE_ACP_RPC_DUMP = rpcFile;
      await create();
      const first = await instance.adapter.sendTurn({ threadId: "t-pool-stop", text: "one" });
      await recorder.until((e) => e.type === "turn.completed" && e.turnId === first.turnId);
      await instance.adapter.stopAll();
      const second = await instance.adapter.sendTurn({
        threadId: "t-pool-stop",
        text: "two",
        resumeCursor: "fake-acp-session",
      });
      const done = await recorder.until((e) => e.type === "turn.completed" && e.turnId === second.turnId);
      expect(done).toMatchObject({ ok: true });
      expect(launches()).toBe(2);
      expect(rpc()).toContain("session/load");
    });

    it("an interrupt that the agent honors keeps the process pooled", async () => {
      countFile = join(scratch, "launches");
      process.env.FAKE_ACP_LAUNCH_COUNT_FILE = countFile;
      await create(KimiAgentDriver, "hang");
      await instance.adapter.sendTurn({ threadId: "t-pool-interrupt", text: "go" });
      await recorder.until((e) => e.type === "session.started");
      await instance.adapter.interruptTurn("t-pool-interrupt");
      const done = await recorder.until((e) => e.type === "turn.completed");
      expect(done).toMatchObject({ stopReason: "cancelled" });
      // hang never resolves prompts, so a second turn would hang too — the
      // process staying pooled is the launch count, not another prompt
      expect(launches()).toBe(1);
    });
  });
});

describe("ACP snapshot", () => {
  it("a missing binary is unavailable", async () => {
    const instance = await KimiAgentDriver.create({
      instanceId: "grok-missing",
      displayName: undefined,
      environment: {},
      enabled: true,
      config: { cli: "definitely-not-a-real-grok-binary", fullAuto: false },
    });
    const snap = await instance.snapshot();
    expect(snap.state).toBe("unavailable");
    await instance.dispose();
  });

  it("kimi checks KIMI_CODE_HOME before the child HOME", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "omb-kimi-auth-"));
    const kimiHome = join(scratch, "custom-kimi-home");
    const childHome = join(scratch, "child-home");
    mkdirSync(join(childHome, ".kimi-code", "credentials"), { recursive: true });
    writeFileSync(join(childHome, ".kimi-code", "credentials", "kimi-code.json"), "{}");

    const instance = await KimiAgentDriver.create({
      instanceId: "kimi-custom-home",
      displayName: undefined,
      environment: { KIMI_CODE_HOME: kimiHome, HOME: childHome },
      enabled: true,
      config: { cli: FAKE_CLI, fullAuto: false },
    });
    try {
      expect((await instance.snapshot()).authenticated).toBe(false);
      mkdirSync(join(kimiHome, "credentials"), { recursive: true });
      writeFileSync(join(kimiHome, "credentials", "kimi-code.json"), "{}");
      expect((await instance.snapshot()).authenticated).toBe(true);
    } finally {
      await instance.dispose();
      await removeTempDir(scratch);
    }
  });

  it("kimi resolves default credentials from the child HOME", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "omb-kimi-home-"));
    const credentialDir = join(scratch, ".kimi-code", "credentials");
    mkdirSync(credentialDir, { recursive: true });
    writeFileSync(join(credentialDir, "kimi-code.json"), "{}");

    const instance = await KimiAgentDriver.create({
      instanceId: "kimi-child-home",
      displayName: undefined,
      environment: { HOME: scratch },
      enabled: true,
      config: { cli: FAKE_CLI, fullAuto: false },
    });
    try {
      expect((await instance.snapshot()).authenticated).toBe(true);
    } finally {
      await instance.dispose();
      await removeTempDir(scratch);
    }
  });

  it("awaits an async isAuthenticated", async () => {
    const instance = await AsyncAuthDriver.create({
      instanceId: "async-auth",
      displayName: undefined,
      environment: {},
      enabled: true,
      config: { cli: FAKE_CLI, fullAuto: false },
    });
    try {
      // without the await this is a Promise: truthy, but not `true`
      expect((await instance.snapshot()).authenticated).toBe(true);
    } finally {
      await instance.dispose();
    }
  });
});
