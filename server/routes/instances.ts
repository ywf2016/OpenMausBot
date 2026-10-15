// The provider instances (the model picker's engines and accounts), moved
// out of server/index.ts's inline routes into the route table
// (server/routes/README.md): list and describe them, add a ChatGPT or Claude
// account, sign in and out of an instance, change an instance's icon, CLI
// path, account name or tools setting, remove an added Claude account, find
// and probe CLI binaries, and update a Claude Code install. Provider-settings
// writes share index.ts's single-flight lock (`providerConfigBusy`) with the
// config and account routes; it reaches here as a get/set view.
import { randomUUID } from "node:crypto";
import { providerIconError } from "../../shared/provider-icon.ts";
import {
  assertSeparateClaudeAccount,
  configuredAccountDirectory,
  createClaudeAccountSchema,
  instanceSettingsSchema,
  newClaudeAccount,
} from "../claude-accounts.ts";
import { updateClaudeCli } from "../claude-update.ts";
import { persistableInstanceConfigs, saveConfig, withInstanceCli, type AppConfig } from "../config.ts";
import type { InstanceConfigMap, ModelSelection } from "../contracts.ts";
import { BUILT_IN_DRIVERS } from "../drivers/builtIn.ts";
import { findCliCandidates, resetPathCache } from "../env-path.ts";
import type { ProviderRegistry } from "../harness/registry.ts";
import { HOSTED_PROVIDER_SETTINGS_ERROR } from "../hosted-models.ts";
import type { ProviderAuthSessions } from "../provider-auth-sessions.ts";
import { providerIconPatchSchema, withInstanceIcon } from "../provider-icon.ts";
import { isLoopbackHost, isProxied } from "../request-auth.ts";
import type { SessionRegistry } from "../sessions.ts";
import type { Store } from "../store.ts";
import type { BusyFlag, ConfigStatus } from "./config.ts";
import { PASS, type RouteHandler } from "./table.ts";

export interface InstanceRouteDeps {
  store: Pick<Store, "bots" | "tasks">;
  cfg: AppConfig;
  registry: Pick<ProviderRegistry, "get" | "refreshModels" | "installRuntime" | "cliTarget">;
  sessions: Pick<SessionRegistry, "isLive">;
  providerAuthSessions: Pick<ProviderAuthSessions, "status" | "start" | "revokeOwner" | "signOut" | "complete" | "cancel">;
  /** Whether a hosted model policy applies here (index.ts `hostedModels`); the routes only test it. */
  hostedModels: boolean;
  /** The provider-settings lock, shared with the config and account routes. */
  providerConfigBusy: BusyFlag;
  /** The instances being replaced right now; a turn on one of them waits. */
  providerInstancesChanging: Pick<Set<string>, "add" | "delete">;
  /** Every provider instance, as GET /api/instances describes them. */
  describeInstances(): Promise<unknown>;
  /** Saves the map with this instance changed and applies it. */
  persistProviderInstance(instanceId: string, instances: InstanceConfigMap): Promise<void>;
  configStatus(): ConfigStatus;
  broadcast(payload: Record<string, unknown>): void;
  /** Spawns `<cli> --version` the way a turn would. */
  testCliBinary(cli: string, driver: (typeof BUILT_IN_DRIVERS)[number] | undefined): Promise<unknown>;
  /** The environment a CLI probe or update runs with. */
  cliProbeEnvironment(): NodeJS.ProcessEnv;
  threadBusy(botId: string, threadId: string): boolean;
  botForThread(botId: string, threadId: string): Pick<{ modelSelection: ModelSelection }, "modelSelection"> | null;
  activeGroupTurnForBot(botId: string): { threadId: string } | null;
}

export function createInstanceRoutes(deps: InstanceRouteDeps): RouteHandler {
  const {
    store, cfg, registry, sessions, providerAuthSessions, hostedModels, providerConfigBusy, providerInstancesChanging,
    describeInstances, persistProviderInstance, configStatus, broadcast, testCliBinary, cliProbeEnvironment, threadBusy,
    botForThread, activeGroupTurnForBot,
  } = deps;
  // One updater per executable: multiple Claude instances can point at the same
  // install, and running two self-updates against it would race its files.
  const claudeUpdatesInFlight = new Set<string>();
  return async ({ req, res, url, path, method, auth, json, readBody }) => {
    // ── provider instances (model picker) ──
    if (method === "GET" && path === "/api/instances") {
      // Rescan PATH first: this endpoint is how the app answers "what can I
      // run?", and the interesting case is a CLI installed since launch.
      // Windows never pushes PATH changes into a live process, so without
      // this the answer is frozen at boot and "check again" is a no-op.
      resetPathCache();
      return json(res, 200, { instances: await describeInstances() });
    }
    const companyMutation = /^\/api\/instances\/(company\.[\w.-]+)(?:\/|$)/.exec(path);
    if (hostedModels && path.startsWith("/api/instances/") && method !== "GET") return json(res, 403, { error: HOSTED_PROVIDER_SETTINGS_ERROR });
    if (companyMutation && method !== "GET") return json(res, 403, { error: "Company accounts are read-only here. Manage this connection in desktop Settings." });

    const instanceIconPatch = /^\/api\/instances\/([\w.-]+)\/icon$/.exec(path);
    if (method === "PATCH" && instanceIconPatch) {
      if (!String(req.headers["content-type"] ?? "").toLowerCase().startsWith("application/json")) {
        return json(res, 415, { error: "content-type must be application/json" });
      }
      const parsed = providerIconPatchSchema.safeParse(await readBody(req, 192 * 1024));
      if (!parsed.success) {
        const detail = parsed.error.issues[0]?.message;
        return json(res, 400, { error: detail && detail !== "Invalid input" ? detail : "Choose a built-in icon or upload a PNG, JPEG, or WebP image up to 128 KB." });
      }
      if (parsed.data.icon) {
        const invalid = providerIconError(parsed.data.icon);
        if (invalid) return json(res, 400, { error: invalid });
      }
      if (providerConfigBusy.get()) return json(res, 409, { error: "provider settings are already being updated" });
      providerConfigBusy.set(true);
      try {
        const changed = withInstanceIcon(cfg, instanceIconPatch[1], parsed.data.icon);
        if (!changed.ok) return json(res, 404, { error: `unknown instance "${instanceIconPatch[1]}"` });
        saveConfig({ instances: changed.instances }, { replaceInstances: true });
        cfg.instances = changed.instances;
        broadcast({ kind: "config", ...configStatus() });
        return json(res, 200, { instances: await describeInstances() });
      } finally { providerConfigBusy.set(false); }
    }

    if (method === "POST" && path === "/api/instances/chatgpt-accounts") {
      if (!String(req.headers["content-type"] ?? "").toLowerCase().startsWith("application/json")) return json(res, 415, { error: "content-type must be application/json" });
      const parsed = createClaudeAccountSchema.pick({ displayName: true }).safeParse(await readBody(req, 8192));
      if (!parsed.success) return json(res, 400, { error: "Enter an account name (up to 80 characters)." });
      if (providerConfigBusy.get()) return json(res, 409, { error: "provider settings are already being updated" });
      providerConfigBusy.set(true);
      try {
        const instances = persistableInstanceConfigs(cfg);
        const instanceId = `chatgpt-${randomUUID()}`;
        // Reuse the selected executable, never another account's environment.
        const cli = (instances.chatgpt?.config as { cli?: unknown } | undefined)?.cli
          ?? (instances.codex?.config as { cli?: unknown } | undefined)?.cli;
        instances[instanceId] = { driver: "codex", displayName: parsed.data.displayName,
          config: { authMode: "chatgpt-plan", ...(typeof cli === "string" && cli ? { cli } : {}) } };
        await persistProviderInstance(instanceId, instances);
        return json(res, 201, { instanceId, instances: await describeInstances() });
      } finally { providerConfigBusy.set(false); }
    }

    if (method === "POST" && path === "/api/instances/claude-accounts") {
      if (!String(req.headers["content-type"] ?? "").toLowerCase().startsWith("application/json")) {
        return json(res, 415, { error: "content-type must be application/json" });
      }
      const parsed = createClaudeAccountSchema.safeParse(await readBody(req, 8192));
      if (!parsed.success) return json(res, 400, { error: "Enter an account name (up to 80 characters) and an optional configuration directory." });
      if (providerConfigBusy.get()) return json(res, 409, { error: "provider settings are already being updated" });
      providerConfigBusy.set(true);
      try {
        const { instanceId, instances } = newClaudeAccount(cfg, parsed.data);
        await persistProviderInstance(instanceId, instances);
        return json(res, 201, { instanceId, instances: await describeInstances() });
      } finally { providerConfigBusy.set(false); }
    }

    const authStatus = /^\/api\/instances\/([\w.-]+)\/auth\/status$/.exec(path);
    if (method === "GET" && authStatus) {
      res.setHeader("cache-control", "no-store");
      const state = await providerAuthSessions.status(authStatus[1], auth.kind === "session" ? auth.session.id : "loopback", url.searchParams.get("flowId") ?? "");
      return json(res, 200, { auth: state });
    }
    const instanceAction = /^\/api\/instances\/([\w.-]+)\/(refresh-models|install|auth\/start|auth\/complete|auth\/cancel|auth\/sign-out)$/.exec(path);
    if (method === "POST" && instanceAction) {
      if (!String(req.headers["content-type"] ?? "").toLowerCase().startsWith("application/json")) {
        return json(res, 415, { error: "content-type must be application/json" });
      }
      const instanceId = instanceAction[1];
      const action = instanceAction[2];
      const owner = auth.kind === "session" ? auth.session.id : "loopback";
      if (action.startsWith("auth/")) res.setHeader("cache-control", "no-store");
      try {
        if (action === "refresh-models") {
          if (!(await registry.refreshModels(instanceId))) return json(res, 404, { error: "unknown instance" });
          return json(res, 200, { instances: await describeInstances() });
        }
        if (action === "install") {
          if (!(await registry.installRuntime(instanceId))) return json(res, 404, { error: "Installing this engine from Settings is not available on this server. Use the install command on the machine running OpenMausBot." });
          return json(res, 200, { instances: await describeInstances() });
        }
        if (action === "auth/start") {
          const instance = registry.get(instanceId);
          if (!instance) return json(res, 404, { error: "unknown instance" });
          // The official local OAuth callback terminates on this machine,
          // not a phone or the browser visiting a remotely hosted workspace.
          if (instance.authenticationMethod === "browser-pkce" && (isProxied(req) || !isLoopbackHost(req.socket.remoteAddress))) {
            return json(res, 403, { error: "Continue with ChatGPT on the computer running OpenMausBot. Hosted Pro sign-in requires OpenAI's hosted-app approval." });
          }
          const started = await providerAuthSessions.start(instance, owner);
          // Revocation can arrive while the CLI is obtaining a device code.
          if (auth.kind === "session" && !sessions.isLive(auth.session.id)) {
            providerAuthSessions.revokeOwner(owner);
            return json(res, 401, { error: "Your session ended. Start a new sign-in." });
          }
          return json(res, 200, { auth: started });
        }
        if (action === "auth/sign-out") {
          const instance = registry.get(instanceId);
          if (!instance) return json(res, 404, { error: "unknown instance" });
          await providerAuthSessions.signOut(instance, owner);
          return json(res, 200, { instances: await describeInstances() });
        }
        if (action === "auth/complete") {
          const body = await readBody(req);
          const flowId = typeof body?.flowId === "string" ? body.flowId : "";
          // `code` for a pasted sign-in code (Claude), `callbackUrl` for a browser callback
          const callbackUrl = typeof body?.callbackUrl === "string" ? body.callbackUrl : typeof body?.code === "string" ? body.code : "";
          if (!flowId || !callbackUrl) return json(res, 400, { error: "flowId and a code or callbackUrl are required" });
          await providerAuthSessions.complete(instanceId, owner, flowId, callbackUrl);
          return json(res, 200, { ok: true });
        }
        const body = await readBody(req, 4096);
        await providerAuthSessions.cancel(instanceId, owner, typeof body?.flowId === "string" ? body.flowId : "");
        return json(res, 200, { ok: true });
      } catch (error) {
        const requestedStatus = error && typeof error === "object" ? (error as { status?: unknown }).status : undefined;
        const status = typeof requestedStatus === "number" && [400, 401, 404, 409, 413, 415].includes(requestedStatus) ? requestedStatus : 500;
        return json(res, status, { error: error instanceof Error ? error.message : String(error) });
      }
    }

    // ── CLI binary discovery for the Engines "detected" dropdown ──
    // ?name=claude → absolute paths of every `claude` on the augmented PATH,
    // in PATH order (first = what a bare name runs). Polled when the user
    // opens the Custom picker so a just-installed CLI appears without a restart.
    if (method === "GET" && path === "/api/cli-candidates") {
      const name = url.searchParams.get("name") ?? "";
      resetPathCache();
      return json(res, 200, { candidates: findCliCandidates(name) });
    }

    // ── pre-save CLI probe: does this path actually run? ──
    // POST {cli, driver} → spawn `<cli> --version` with the same PATH the
    // turn itself would use. A miss here (typo, missing exec bit, a binary
    // the GUI app can't see) means every turn would fail, so the UI asks
    // before saving rather than registering a dead engine.
    if (method === "POST" && path === "/api/cli-test") {
      // same gate as the local-VM lifecycle routes: this executes a local
      // binary, so a hostile page must not be able to submit it as a simple
      // text/plain cross-origin request
      if (!String(req.headers["content-type"] ?? "").toLowerCase().startsWith("application/json")) {
        return json(res, 415, { error: "content-type must be application/json" });
      }
      const body = await readBody(req);
      const cli = typeof body?.cli === "string" ? body.cli.trim() : "";
      if (!cli || /[\n\r]/.test(cli)) return json(res, 400, { error: "cli must be a non-empty path" });
      const driver = typeof body?.driver === "string" ? BUILT_IN_DRIVERS.find((d) => d.driverKind === body.driver) : undefined;
      // Probe the exact configured wrapper plus --version. testCliBinary uses
      // a credential-redacted environment, so fixed wrapper arguments cannot
      // turn this endpoint into an inherited-secret reader.
      const probe = await testCliBinary(cli, driver);
      return json(res, 200, probe);
    }

    // ── instance-scoped Claude Code update ──
    // No command or path comes from the request: the registry supplies the
    // executable already configured for this Claude instance. The JSON gate
    // keeps a hostile page from triggering a local process with a simple
    // cross-origin form request.
    const busyProviderSelections = () => store.bots.flatMap((bot) => {
      const busyTasks = store.tasks(bot.id).filter((task) => threadBusy(bot.id, task.threadId));
      const selections = busyTasks.map((task) => botForThread(bot.id, task.threadId)!.modelSelection);
      // Rooms still run from the profile default; a direct thread does not.
      if (activeGroupTurnForBot(bot.id) || (bot.busy && busyTasks.length === 0)) selections.push(bot.modelSelection);
      return selections;
    });
    const claudeUpdate = /^\/api\/instances\/([\w.-]+)\/claude-update$/.exec(path);
    if (method === "POST" && claudeUpdate) {
      if (!String(req.headers["content-type"] ?? "").toLowerCase().startsWith("application/json")) {
        return json(res, 415, { error: "content-type must be application/json" });
      }
      await readBody(req);
      const target = registry.cliTarget(claudeUpdate[1]);
      if (!target) return json(res, 404, { error: "no such provider instance" });
      if (target.driverKind !== "claudeAgent") {
        return json(res, 400, { error: "only Claude Code instances can be updated here" });
      }
      if (!target.cli) return json(res, 409, { error: "this Claude instance has no configured executable" });
      if (claudeUpdatesInFlight.has(target.cli)) {
        return json(res, 409, { error: "this Claude installation is already updating" });
      }
      const active = busyProviderSelections().some((selection) => registry.cliTarget(selection.instanceId)?.cli === target.cli);
      if (active) {
        return json(res, 409, { error: "wait for running Claude tasks to finish before updating" });
      }

      claudeUpdatesInFlight.add(target.cli);
      try {
        const result = await updateClaudeCli(target.cli, cliProbeEnvironment());
        resetPathCache();
        return json(res, 200, { ok: true, version: result.version });
      } catch (error) {
        return json(res, 500, { error: error instanceof Error ? error.message : String(error) });
      } finally {
        claudeUpdatesInFlight.delete(target.cli);
      }
    }

    // ── per-instance settings (CLI/account or API tool support) ──
    // PATCH /api/instances/:id {cli: "/path/to/cli" | ""} — "" reverts to the
    // driver default. Only this idle instance is replaced; siblings keep running.
    const instancePatch = /^\/api\/instances\/([\w.-]+)$/.exec(path);
    if (method === "PATCH" && instancePatch) {
      // same non-simple-request gate as the local-VM lifecycle routes
      if (!String(req.headers["content-type"] ?? "").toLowerCase().startsWith("application/json")) {
        return json(res, 415, { error: "content-type must be application/json" });
      }
      const parsed = instanceSettingsSchema.safeParse(await readBody(req, 16384));
      if (!parsed.success) return json(res, 400, { error: "Supply a valid CLI path, account name, configuration directory or boolean tools setting." });
      const body = parsed.data;
      const instanceId = instancePatch[1];
      if (providerConfigBusy.get()) return json(res, 409, { error: "provider settings are already being updated" });
      if (busyProviderSelections().some((selection) => selection.instanceId === instanceId)) {
        return json(res, 409, { error: "Wait for bots using this account to finish before changing its settings." });
      }
      providerConfigBusy.set(true);
      providerInstancesChanging.add(instanceId);
      try {
        const result = body.cli === undefined ? { ok: true, config: cfg } : withInstanceCli(cfg, instanceId, body.cli);
        const instances = persistableInstanceConfigs(result.config);
        if (!result.ok || !Object.hasOwn(instances, instanceId)) return json(res, 404, { error: `unknown instance "${instanceId}"` });
        const entry = instances[instanceId];
        if ((body.displayName !== undefined || body.configDir !== undefined) && entry.driver !== "claudeAgent") {
          return json(res, 400, { error: "Account settings are currently available for Claude only." });
        }
        if (body.tools !== undefined) {
          if (entry.driver !== "openai-compat") {
            return json(res, 400, { error: "The tools setting is available for OpenAI-compatible instances only." });
          }
          entry.config = { ...entry.config as Record<string, unknown>, tools: body.tools };
        }
        if (body.displayName !== undefined) entry.displayName = body.displayName;
        if (body.configDir !== undefined) {
          let previousDir: string | undefined;
          try { previousDir = configuredAccountDirectory(entry); } catch { /* Allow repairing an unused malformed account. */ }
          entry.config = { ...entry.config as Record<string, unknown>, configDir: body.configDir };
          if (previousDir !== configuredAccountDirectory(entry)) {
            const used = store.bots.some((bot) => bot.modelSelection.instanceId === instanceId || bot.fallback?.some(candidate => candidate.instanceId === instanceId) || store.tasks(bot.id).some((task) =>
              task.modelSelection?.instanceId === instanceId || task.resumeCursors[instanceId] || task.lastInstanceId === instanceId));
            if (used) return json(res, 409, { error: "This account is used by bots or conversation history. Add another account and select it for the bot instead." });
            assertSeparateClaudeAccount(instances, instanceId, entry);
          }
        }
        await persistProviderInstance(instanceId, instances);
        return json(res, 200, { instances: await describeInstances() });
      } finally {
        providerInstancesChanging.delete(instanceId);
        providerConfigBusy.set(false);
      }
    }

    if (method === "DELETE" && instancePatch) {
      const instanceId = instancePatch[1];
      if (providerConfigBusy.get()) return json(res, 409, { error: "provider settings are already being updated" });
      const instances = persistableInstanceConfigs(cfg);
      if (!Object.hasOwn(instances, instanceId)) return json(res, 404, { error: "unknown instance" });
      if (instances[instanceId].driver !== "claudeAgent" || instanceId === "claude") {
        return json(res, 400, { error: "Only added Claude accounts can be removed here." });
      }
      if (cfg.defaultModelSelection?.instanceId === instanceId || store.bots.some((bot) =>
        bot.modelSelection.instanceId === instanceId || bot.fallback?.some(candidate => candidate.instanceId === instanceId) || store.tasks(bot.id).some((task) => task.modelSelection?.instanceId === instanceId)) ||
        busyProviderSelections().some((selection) => selection.instanceId === instanceId)) {
        return json(res, 409, { error: "Choose another account for the bots and default model using this account before removing it." });
      }
      providerConfigBusy.set(true);
      providerInstancesChanging.add(instanceId);
      try {
        delete instances[instanceId];
        await persistProviderInstance(instanceId, instances);
        return json(res, 200, { instances: await describeInstances() });
      } finally {
        providerInstancesChanging.delete(instanceId);
        providerConfigBusy.set(false);
      }
    }
    return PASS;
  };
}
