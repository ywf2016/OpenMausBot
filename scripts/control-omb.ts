#!/usr/bin/env -S node --experimental-strip-types
// The openmausbot headless verbs (server/headless/main.ts) for verifying an
// isolated fixture: every verb, with its flags and its JSON result, plus the
// fixture launcher and the renderer verbs (`ui`). One difference on purpose:
// a verb that changes something never runs on a server it merely found, so a
// recipe cannot reach the person's own app by accident.
import { spawn, type ChildProcess } from "node:child_process";
import { closeSync, mkdirSync, mkdtempSync, openSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs, type ParseArgsOptionsConfig } from "node:util";

import { EXIT, HeadlessError } from "../server/headless/args.ts";
import { runHeadlessCommand } from "../server/headless/main.ts";
import type { handleToolCall, request } from "./mcp-server.ts";
import { launchUi, runControlOmbUi } from "./testing/control-omb-ui.ts";
import { removeTempDir, waitForExit } from "../server/testing/cleanup.ts";
import { freePortBlock } from "../server/testing/ports.ts";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
/** Deterministic name for the seeded starter bot of a verification fixture,
 * deliberately outside the server/names.ts pool so no suite can plan a bot
 * that collides with the starter (#1257). */
export const FIXTURE_STARTER_BOT_NAME = "Fixture Starter";
const FAKE_CLI = join(ROOT, "server", "testing", "fake-claude-cli.ts");

/** A control-omb refusal: the headless verbs' own error, with control-omb's (message, hint) shape. */
export class ControlOmbError extends HeadlessError {
  constructor(message: string, hint?: string) {
    super(message, EXIT.failed, hint);
  }
}

export interface ControlOmbDependencies {
  callTool?: typeof handleToolCall;
  request?: typeof request;
  env?: NodeJS.ProcessEnv;
}

export const HELP_UI = `renderer (needs a ui launch handle; every verb takes --ui HANDLE, never discovery):
  node --experimental-strip-types scripts/control-omb.ts ui launch [--entry threads] [--tool-calls JSON] [--mode happy]
  ui snapshot --ui HANDLE [--interactive]
  ui click --ui HANDLE (--ref @eN | --name NAME)
  ui type --ui HANDLE (--ref @eN | --name NAME) --text TEXT
  ui press --ui HANDLE --keys KEYS
  ui screenshot --ui HANDLE --out PATH.png
  ui console --ui HANDLE
  ui eval --ui HANDLE --js CODE
  ui flag --ui HANDLE --set features.NAME=VALUE [--dry-run]
  ui wait-settle --ui HANDLE [--timeout 30]
  ui help`;

export const HELP = `control-omb — the openmausbot headless verbs (docs/headless-cli.md), for verifying a fixture

read-only (every verb also takes --url URL):
  doctor
  bots
  channels
  models
  messages --bot ID [--task ID] [--limit 30]
  messages --channel ID [--task ID] [--limit 30]
  wait --bot ID [--task ID] [--timeout 30] [--for job|turn]
  wait --channel ID [--task ID] [--timeout 30]
  export --thread ID --out DIR
  events --job THREAD_ID --until-settled

mutating (an explicit --url or OPENMAUSBOT_URL/OMB_PORT is required):
  new-bot --name NAME [--section TEAM]
  new-channel --name NAME --members ID,ID
  send --bot ID --text TEXT [--task ID] [--dry-run]
  send-channel --channel ID --text TEXT [--task ID] [--dry-run]
  run --bot ID --text TEXT [--wait job|turn|none] [--timeout 5m] [--export DIR]
  interrupt --bot ID [--task ID] [--tree] [--dry-run]
  interrupt --channel ID [--task ID] [--dry-run]
  edit --bot ID --message ID --text TEXT [--task ID] [--dry-run]
  set-model --bot ID --instance ID --model ID [--task ID] [--effort LEVEL] [--dry-run]
  team apply FILE [--dry-run | --check]

${HELP_UI}

isolated fixture:
  node --experimental-strip-types scripts/control-omb.ts launch

Output is JSON. wait defaults to 30 seconds here. launch and ui launch own a temporary fake-engine server until interrupted.`;

const commonOptions = {
  url: { type: "string" },
} satisfies ParseArgsOptionsConfig;

/** Strict flag parsing for one `ui` verb. */
export function parse(
  command: string,
  args: string[],
  options: ParseArgsOptionsConfig = {},
): Record<string, unknown> & { url?: string } {
  try {
    return parseArgs({
      args,
      options: { ...commonOptions, ...options },
      strict: true,
      allowPositionals: false,
    }).values as Record<string, unknown> & { url?: string };
  } catch (error) {
    throw new ControlOmbError(
      error instanceof Error ? error.message : String(error),
      `run control-omb help for the ${command} syntax`,
    );
  }
}

/** Run one verb and return its result: the same verbs as `openmausbot`, never on a found server when it changes something. */
export async function runControlOmb(
  argv: string[],
  dependencies: ControlOmbDependencies = {},
): Promise<unknown> {
  const [command = "help", ...args] = argv;
  if (command === "help" || command === "--help" || command === "-h") return HELP;
  if (command === "launch") throw new ControlOmbError("launch is available only from the executable CLI");
  // The first positional after `ui` is the verb; its tail is parsed strictly per verb.
  if (command === "ui") return runControlOmbUi(args);
  return runHeadlessCommand(argv, {
    env: dependencies.env ?? process.env,
    ...(dependencies.callTool ? { callTool: dependencies.callTool } : {}),
    ...(dependencies.request ? { request: dependencies.request } : {}),
    requireExplicitForMutations: true,
    defaultWaitMs: 30_000,
  });
}

export interface VerificationServer {
  info: { url: string; pid: number; dataDir: string; logPath: string };
  fixtureDumpPath: string;
  child: ChildProcess;
  close(): Promise<void>;
}

/** The environment of a verification server child: a temporary home in
 * `dataDir`, the fake engine's knobs from `parentEnv`, node on PATH, and
 * nothing else from the parent shell or this machine's installed CLIs. A
 * test that restarts its own fixture server on the same data uses this too. */
export function verificationServerEnvironment(parentEnv: NodeJS.ProcessEnv, dataDir: string, port: number): NodeJS.ProcessEnv {
  const childEnv: NodeJS.ProcessEnv = {};
  const platformKeys = new Set(["SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL", "TZ"]);
  for (const [key, value] of Object.entries(parentEnv)) {
    const normalized = key.toUpperCase();
    if (value && platformKeys.has(normalized)) childEnv[normalized] = value;
  }
  const fixtureTemp = join(dataDir, "tmp");
  Object.assign(childEnv, {
    HOME: dataDir,
    USERPROFILE: dataDir,
    APPDATA: join(dataDir, "AppData", "Roaming"),
    LOCALAPPDATA: join(dataDir, "AppData", "Local"),
    XDG_CONFIG_HOME: join(dataDir, ".config"),
    XDG_CACHE_HOME: join(dataDir, ".cache"),
    XDG_DATA_HOME: join(dataDir, ".local", "share"),
    TEMP: fixtureTemp,
    TMP: fixtureTemp,
    TMPDIR: fixtureTemp,
    HERMES_HOME: join(dataDir, ".hermes"),
    OMB_DATA_DIR: dataDir,
    OMB_PORT: String(port),
    OMB_WEBHOOK_PORT: String(port + 1),
    // The fixture's default CLI behaviour; a caller that sets
    // FAKE_CLAUDE_MODE explicitly overrides it below to drive the CLI's
    // failure paths (exit-early, dead-session, hang...) through the real
    // server. Nothing else from the parent shell reaches the fixture.
    FAKE_CLAUDE_MODE: parentEnv.FAKE_CLAUDE_MODE || "happy",
    FAKE_CLAUDE_DUMP: join(dataDir, "fake-claude-dump.json"),
    // Keep the environment hermetic while allowing POSIX to resolve the
    // fake CLI's `#!/usr/bin/env node` shebang. Windows resolves that same
    // fixture through spawnCli without a shell.
    PATH: dirname(process.execPath),
    // ...and keep engine discovery to that PATH and this home. Without it the
    // server also scans /opt/homebrew/bin, /usr/local/bin and the login
    // shell's PATH, so a developer's own `codex` (or any engine CLI) becomes
    // an "available" engine that CI never has (#2035).
    OMB_TEST_SEALED_PATH: "1",
  });
  // The fake engine's own knobs (mode, replies, tool calls) are the one thing
  // a caller may script into the child: FAKE_CLAUDE_* crosses, and the Codex
  // fake's background-call knobs (FAKE_CODEX_TEXT_*); nothing else.
  for (const [key, value] of Object.entries(parentEnv)) {
    // FAKE_CLAUDE_DUMP stays the launcher's: assertions read fixtureDumpPath.
    if ((key.startsWith("FAKE_CLAUDE_") || key.startsWith("FAKE_CODEX_TEXT_")) && key !== "FAKE_CLAUDE_DUMP" && value) childEnv[key] = value;
  }
  // A test's key for relaying an organization library into the fixture
  // (POST /api/testing/org-library); the route does not exist without it.
  if (parentEnv.OMB_TEST_ORG_LIBRARY_KEY) childEnv.OMB_TEST_ORG_LIBRARY_KEY = parentEnv.OMB_TEST_ORG_LIBRARY_KEY;
  // One log line per browser call and preview frame (server/browser-runtime.ts).
  if (parentEnv.OMB_BROWSER_TIMING === "1") childEnv.OMB_BROWSER_TIMING = "1";
  // Live calls against server/testing/fake-openai-live.ts only: a loopback
  // URL, and a key that only ever reaches that fake.
  const liveUrl = parentEnv.OMB_OPENAI_LIVE_URL?.trim() ?? "";
  if (/^http:\/\/127\.0\.0\.1:\d{1,5}$/.test(liveUrl)) {
    childEnv.OMB_OPENAI_LIVE_URL = liveUrl;
    if (parentEnv.OMB_OPENAI_LIVE_KEY) childEnv.OMB_OPENAI_LIVE_KEY = parentEnv.OMB_OPENAI_LIVE_KEY;
  }
  // Live calls need Pro (electron/pro-plan.mjs): a recipe that starts calls
  // says so with OMB_PRO_PLAN=1, the way any server the desktop app did not
  // start does. Without it the fixture refuses a call with 402 needsPro.
  if (parentEnv.OMB_PRO_PLAN?.trim() === "1") childEnv.OMB_PRO_PLAN = "1";
  // Voice-note e2e fault injection: arms the one-shot audio-append failure
  // prelude inside the fixture server (see fail-audio-append-once.mjs).
  if (parentEnv.OMB_TEST_FAIL_AUDIO_APPEND_ONCE) {
    childEnv.OMB_TEST_FAIL_AUDIO_APPEND_ONCE = parentEnv.OMB_TEST_FAIL_AUDIO_APPEND_ONCE;
  }
  // Unanswered-approval e2e: the Claude broker's deadline fires on a marker
  // file in the fixture's home instead of after 15 minutes
  // (approval-clock-hooks.mjs).
  if (parentEnv.OMB_TEST_APPROVAL_CLOCK === "1") childEnv.OMB_TEST_APPROVAL_CLOCK = "1";
  // Stalled-turn e2e: the stall watchdog's window elapses on a marker file
  // instead of after 20 minutes of silence (stall-clock-hooks.mjs).
  if (parentEnv.OMB_TEST_STALL_CLOCK === "1") childEnv.OMB_TEST_STALL_CLOCK = "1";
  // Runtime coordination e2e: the server runs teammate coordination on the
  // Desktop mode: the server runs as the desktop app runs it, and the owner
  // capability the app would hand it is this one (see desktop-parent.mjs).
  if (parentEnv.OMB_TEST_DESKTOP_OWNER_TOKEN) {
    childEnv.OMB_TEST_DESKTOP_OWNER_TOKEN = parentEnv.OMB_TEST_DESKTOP_OWNER_TOKEN;
    childEnv.OMB_DESKTOP_PARENT = "1";
  }
  return childEnv;
}

/** The node arguments of a verification server child: the fixture hooks its
 * environment asks for, then the server. A test that restarts its own
 * fixture server on the same data starts it with these too. */
export function verificationServerArgs(childEnv: NodeJS.ProcessEnv): string[] {
  const hook = (file: string) => ["--import", pathToFileURL(join(ROOT, "server", "testing", file)).href];
  return [
    "--experimental-strip-types",
    ...(childEnv.OMB_TEST_FAIL_AUDIO_APPEND_ONCE === "1" ? hook("fail-audio-append-once.mjs") : []),
    ...(childEnv.OMB_TEST_DESKTOP_OWNER_TOKEN ? hook("desktop-parent.mjs") : []),
    ...(childEnv.OMB_TEST_APPROVAL_CLOCK === "1" ? hook("approval-clock-hooks.mjs") : []),
    ...(childEnv.OMB_TEST_STALL_CLOCK === "1" ? hook("stall-clock-hooks.mjs") : []),
    join(ROOT, "server", "index.ts"),
  ];
}

/** Start one foreground-owned, fake-engine server with no access to user data. */
export async function launchVerificationServer(
  parentEnv: NodeJS.ProcessEnv = process.env,
  signal?: AbortSignal,
  localVm?: { binDir: string; host: string; sshKey: string; staticDir: string },
  browser?: { binaryPath: string; executablePath: string },
  /** A stand-in enterprise layer (the folder shape core loads) and the key
   * it should accept, so a recipe can prove entitled behaviour offline. */
  enterprise?: { dir: string; licenseKey: string },
  room?: { scripted: boolean },
  /** Optional repository-owned fake providers for multi-engine setup checks. */
  extraProviders: Array<"codex"> = [],
  /** Programmatic tests only: an owned loopback Boat provider, never a live account. */
  boatFixtureApi?: string,
  /** Programmatic web UI qualification against the fixture's static server. */
  staticDir?: string,
): Promise<VerificationServer> {
  if (boatFixtureApi) {
    if (!/^http:\/\/127\.0\.0\.1:[1-9]\d{0,4}$/.test(boatFixtureApi)) {
      throw new ControlOmbError("Boat verification requires an explicit loopback HTTP provider");
    }
    try { new URL(boatFixtureApi); }
    catch { throw new ControlOmbError("Boat verification requires a valid loopback port"); }
  }
  if (localVm) {
    const endpoint = new URL(localVm.host);
    if (endpoint.protocol !== "ssh:" || endpoint.hostname !== "127.0.0.1" || endpoint.password) {
      throw new ControlOmbError("Local VM verification requires an explicit loopback Podman machine");
    }
  }
  // OMB_TEST_PORTS=19975-19999 keeps a recipe's fixture in a range of its own:
  // from the environment it is given, else this process's (a whole suite).
  const range = /^(\d+)-(\d+)$/.exec((parentEnv.OMB_TEST_PORTS ?? process.env.OMB_TEST_PORTS)?.trim() ?? "");
  const port = range ? await freePortBlock([0, 1], Number(range[1]), Number(range[2]) - Number(range[1])) : await freePortBlock([0, 1]);
  if (signal?.aborted) throw new ControlOmbError("verification launch cancelled");
  const url = `http://127.0.0.1:${port}`;
  // Native browser daemons use UNIX sockets; a macOS temp home can exceed
  // their path limit. This is still an owned, randomly named fixture only.
  const dataDir = mkdtempSync(join(browser && process.platform !== "win32" ? "/tmp" : tmpdir(), "openmausbot-verify-data-"));
  const fixtureTemp = join(dataDir, "tmp");
  const fixtureDumpPath = join(dataDir, "fake-claude-dump.json");
  mkdirSync(fixtureTemp, { recursive: true });
  const evidenceDir = join(tmpdir(), "openmausbot-verification-evidence");
  mkdirSync(evidenceDir, { recursive: true });
  const logPath = join(evidenceDir, `server-${Date.now()}-${process.pid}.log`);
  writeFileSync(join(dataDir, "config.json"), JSON.stringify({
    ...(boatFixtureApi ? { box: { token: "box_verification_fixture" } } : {}),
    instances: {
      ...(extraProviders.includes("codex") ? { codex: {
        driver: "codex", displayName: "Verification Codex", config: { cli: fileURLToPath(new URL("../server/testing/fake-codex-app-server.ts", import.meta.url)) },
      } } : {}),
      claude: {
        driver: "claudeAgent",
        displayName: "Verification fixture",
        config: { cli: FAKE_CLI },
        ...(room?.scripted ? { environment: { FAKE_CLAUDE_ROOM_PLAN: join(dataDir, "room-plan.json") } } : {}),
      },
    },
  }, null, 2));

  const log = openSync(logPath, "a", 0o600);
  const childEnv = verificationServerEnvironment(parentEnv, dataDir, port);
  // Opt-in live Local VM fixture: keep the temporary home and fake engine,
  // granting only the explicitly selected machine connection and static UI.
  if (localVm) Object.assign(childEnv, {
    OMB_EXTRA_PATH: [localVm.binDir, ...(process.platform === "win32" ? [join(childEnv.SYSTEMROOT || "C:\\Windows", "System32")] : [])].join(delimiter),
    CONTAINER_HOST: localVm.host,
    CONTAINER_SSHKEY: localVm.sshKey,
    OMB_STATIC_DIR: localVm.staticDir,
  });
  if (enterprise) Object.assign(childEnv, { OMB_ENTERPRISE_DIR: enterprise.dir, OMB_LICENSE_KEY: enterprise.licenseKey });
  if (browser) Object.assign(childEnv, {
    OMB_AGENT_BROWSER_PATH: browser.binaryPath,
    AGENT_BROWSER_EXECUTABLE_PATH: browser.executablePath,
  });
  if (boatFixtureApi) childEnv.OMB_BOAT_API = boatFixtureApi;
  if (staticDir) childEnv.OMB_STATIC_DIR = staticDir;
  const child = spawn(process.execPath, verificationServerArgs(childEnv), {
    cwd: ROOT,
    env: childEnv,
    stdio: ["ignore", log, log],
  });
  closeSync(log);

  const deadline = Date.now() + 20_000;
  try {
    for (;;) {
      if (signal?.aborted) throw new ControlOmbError("verification launch cancelled");
      if (child.exitCode !== null || child.signalCode !== null) {
        throw new Error(`verification server exited before it was ready; see ${logPath}`);
      }
      try {
        const timeout = AbortSignal.timeout(1_000);
        const response = await fetch(`${url}/api/health`, {
          signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
        });
        const body = response.ok ? await response.json() as { app?: string } : null;
        if (body?.app === "openmausbot") break;
      } catch {
        // The server is still starting.
      }
      if (Date.now() >= deadline) throw new Error(`verification server did not become ready; see ${logPath}`);
      await new Promise((resolve) => setTimeout(resolve, 150));
    }
  } catch (error) {
    await waitForExit(child, { signal: "SIGTERM" });
    await removeTempDir(dataDir);
    throw error;
  }

  // The first-run seed gives the starter a random friendly name from
  // server/names.ts, and that pool shares names with bots e2e suites plan
  // ("Quill" among them). A name-based assertion then reports the starter as
  // a phantom leaked bot and flakes (#1257). Pin the name so fixture state is
  // deterministic for every suite built on this launcher; identity remains
  // the honest comparison in tests either way.
  try {
    // Seeding is part of launch, not a one-second health probe. First-run
    // filesystem work can exceed a second on Windows; keep the launch deadline.
    const timeout = AbortSignal.timeout(Math.max(1, deadline - Date.now()));
    const list = await fetch(`${url}/api/bots`, {
      headers: { origin: url },
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
    const body = list.ok ? await list.json() as { bots?: Array<{ id: string }> } : null;
    const seeded = body?.bots;
    if (!Array.isArray(seeded) || seeded.length !== 1) {
      throw new Error(`verification fixture did not seed exactly one starter bot; see ${logPath}`);
    }
    const renameTimeout = AbortSignal.timeout(Math.max(1, deadline - Date.now()));
    const rename = await fetch(`${url}/api/bots/${seeded[0].id}/profile`, {
      method: "PATCH",
      headers: { "content-type": "application/json", origin: url },
      body: JSON.stringify({ name: FIXTURE_STARTER_BOT_NAME }),
      signal: signal ? AbortSignal.any([signal, renameTimeout]) : renameTimeout,
    });
    if (!rename.ok) throw new Error(`verification starter rename failed (${rename.status}); see ${logPath}`);
  } catch (error) {
    await waitForExit(child, { signal: "SIGTERM" });
    await removeTempDir(dataDir);
    throw error;
  }

  let closed = false;
  return {
    info: { url, pid: child.pid!, dataDir, logPath },
    fixtureDumpPath,
    child,
    async close() {
      if (closed) return;
      closed = true;
      await waitForExit(child, { signal: "SIGTERM" });
      await removeTempDir(dataDir);
    },
  };
}

export function controlResultSucceeded(command: string, result: unknown): boolean {
  if (command === "doctor") return (result as { ok?: unknown })?.ok === true;
  if (command === "wait") return (result as { status?: unknown })?.status === "settled";
  if (command === "ui" || command === "run" || command === "team") return (result as { ok?: unknown })?.ok !== false;
  return true;
}

/** pnpm swallows Ctrl-C; a launcher that owns processes must get the signal itself. */
function requireForegroundTerminal(command: string): void {
  if (process.env.npm_lifecycle_event === "control:omb") {
    throw new ControlOmbError(
      `${command} must own the terminal directly so Ctrl-C can clean up its children`,
      `run \`node --experimental-strip-types scripts/control-omb.ts ${command}\``,
    );
  }
}

async function main() {
  const command = process.argv[2] ?? "help";
  if (command === "ui" && process.argv[3] === "launch") {
    requireForegroundTerminal("ui launch");
    await launchUi(process.argv.slice(4));
    return;
  }
  if (command === "launch") {
    requireForegroundTerminal("launch");
    const startup = new AbortController();
    const cancelStartup = () => startup.abort();
    process.once("SIGINT", cancelStartup);
    process.once("SIGTERM", cancelStartup);
    let session: VerificationServer;
    try {
      session = await launchVerificationServer(process.env, startup.signal);
    } finally {
      process.removeListener("SIGINT", cancelStartup);
      process.removeListener("SIGTERM", cancelStartup);
    }
    process.stdout.write(`${JSON.stringify({ ok: true, ...session.info }, null, 2)}\n`);
    await new Promise<void>((resolve) => {
      let stopping = false;
      const stop = () => {
        if (stopping) return;
        stopping = true;
        void session.close().finally(resolve);
      };
      process.once("SIGINT", stop);
      process.once("SIGTERM", stop);
      session.child.once("close", () => {
        if (!stopping) {
          stopping = true;
          process.exitCode = 1;
          process.stderr.write(`${JSON.stringify({
            ok: false,
            error: `verification server exited unexpectedly; see ${session.info.logPath}`,
          }, null, 2)}\n`);
          void removeTempDir(session.info.dataDir).finally(resolve);
        }
      });
    });
    return;
  }
  const result = await runControlOmb(process.argv.slice(2));
  process.stdout.write(typeof result === "string" ? `${result}\n` : `${JSON.stringify(result, null, 2)}\n`);
  if (!controlResultSucceeded(command, result)) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    const failure = error instanceof HeadlessError
      ? { ok: false, error: error.message, ...(error.hint ? { hint: error.hint } : {}) }
      : { ok: false, error: error instanceof Error ? error.message : String(error) };
    process.stderr.write(`${JSON.stringify(failure, null, 2)}\n`);
    process.exitCode = 1;
  });
}
