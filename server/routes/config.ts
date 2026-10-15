// GET and PUT/PATCH /api/config: the app settings (API keys never echoed
// back, booleans only), moved out of server/index.ts's inline routes into the
// route table (server/routes/README.md). The write validates provider keys,
// browser profiles, the Boat and VPS accounts and the Local VM mode against
// the running server before it saves, then rebuilds only the engines whose
// launch config changed. The provider-settings lock (`providerConfigBusy`)
// and the Local VM flags stay in index.ts, shared with the instance and
// computer routes; they reach here as get/set views.
import * as boat from "../boat.ts";
import {
  boatCreateRecoverySnapshot,
  retireDeletedBoatCreate,
} from "../boat-create-idempotency.ts";
import { boatDeletionSnapshot } from "../boat-delete-journal.ts";
import {
  browserEngineStatus,
  browserSessionId,
  closeBrowserSession,
} from "../browser-engine.ts";
import type {
  BrowserCleanupCoordinator,
  BrowserCleanupRequest,
} from "../browser-lifecycle-cleanup.ts";
import {
  finalizeBrowserCleanupMutation,
  requireBrowserCleanupAcknowledged,
} from "../browser-lifecycle-cleanup.ts";
import type { BrowserLive } from "../browser-live.ts";
import { browserRuntimeEnv, type BrowserRuntime } from "../browser-runtime.ts";
import {
  boatAccountResourceChangeError,
  vpsAliasResourceChangeError,
} from "../cloud-backend.ts";
import { CLOUD_PERSONAL_REFUSAL } from "../cloud-owner.ts";
import * as composio from "../composio.ts";
import {
  browserProfilePartitionTarget,
  browserProfileReplacementConflict,
  builtInBrowserEnabled,
  changedInstanceIds,
  engineLaunches,
  loadConfig,
  localVmMode,
  mergeOpenCodeProviderKeys,
  parseConfigPatch,
  saveConfig,
  syncCredentialEnv,
  vpsSshAlias,
  type AppConfig,
} from "../config.ts";
import type { InstanceConfigMap, ModelSelection } from "../contracts.ts";
import {
  deciderIncludedHere,
  deciderSavePatch,
  type Decider,
} from "../decider/index.ts";
import type { ProviderRegistry } from "../harness/registry.ts";
import { HOSTED_PROVIDER_SETTINGS_ERROR } from "../hosted-models.ts";
import { forgetKey } from "../key-rejections.ts";
import type { LocalVmIdleTimer } from "../local-vm-idle.ts";
import { PROVIDER_KEY_KINDS } from "../provider-key-check.ts";
import type { SessionRegistry } from "../sessions.ts";
import type { SharedComputerControl } from "../shared-computer-control.ts";
import type { SharedComputers } from "../shared-computers.ts";
import { parseSkillMd } from "../skills.ts";
import type { Store } from "../store.ts";
import * as tts from "../tts/index.ts";
import * as vps from "../vps-computer.ts";
import { PASS, type RouteHandler } from "./table.ts";

/** A `let` flag index.ts owns, seen live: the settings routes read it and
 * take or release it, index.ts keeps the variable. */
export interface BusyFlag {
  get(): boolean;
  set(busy: boolean): void;
}

/** The remote computer providers a config write can switch accounts for
 * (index.ts's `RemoteComputerProvider`); the computer routes refuse
 * inventory changes on a provider while its account is in transition. */
export type RemoteComputerProvider = "box" | "vps";

/** What a model-selection check answers (index.ts `checkedModelSelection`). */
export type ModelSelectionCheck =
  | { ok: true; selection: ModelSelection }
  | { ok: false; status: number; error: string };

/** The config frame as GET /api/config and the `config` broadcast carry it
 * (index.ts `configStatus()`); the routes only pass it on or spread it. */
export type ConfigStatus = Record<string, unknown>;

export interface ConfigRouteDeps {
  store: Pick<Store, "bots" | "patchBot" | "clearVoiceSelections">;
  cfg: AppConfig;
  registry: Pick<ProviderRegistry, "get">;
  sessions: Pick<SessionRegistry, "revalidateEmailSessions">;
  browserCleanup: Pick<
    BrowserCleanupCoordinator,
    "hasPendingProfile" | "prepare" | "abort" | "commit" | "ensure"
  >;
  browserLive: Pick<BrowserLive, "closeAll" | "closeForSession">;
  browserRuntime: Pick<BrowserRuntime, "heldBy">;
  /** The browser sessions turns have launched, by session id (index.ts Map); a
   * profile whose browser changed drops its entry after the close. */
  usedBrowserSessions: Pick<
    Map<string, { command: string; env: Record<string, string> }>,
    "get" | "delete"
  >;
  decider: Pick<Decider, "testKey">;
  sharedComputers: Pick<SharedComputers, "close">;
  sharedComputerControl: Pick<SharedComputerControl, "close">;
  /** Whether a hosted model policy applies here (index.ts `hostedModels`); the routes only test it. */
  hostedModels: boolean;
  /** Whether this server is a Cloud home (index.ts `CLOUD_HOME`); the write only tests it. */
  cloudHome: boolean;
  lendingEnabled(): boolean;
  /** The provider instance configs engines are launched from. */
  providerConfigs(): InstanceConfigMap;
  localVmIdleMs(): number;
  /** Every Local VM idle timer, so a new idle window applies at once. */
  localVmIdles: { values(): Iterable<Pick<LocalVmIdleTimer, "setIdleMs">> };
  localVmActiveThreads: { readonly size: number };
  localVmLifecycleBusy: { readonly size: number };
  localVmImageBusy: Pick<BusyFlag, "get">;
  localVmModeChangeBusy: Pick<BusyFlag, "set">;
  /** The provider-settings lock, shared with the instance routes and the account routes. */
  providerConfigBusy: BusyFlag;
  /** The remote providers whose account is being switched right now. */
  computerProviderConfigTransitions: Pick<
    Set<RemoteComputerProvider>,
    "add" | "delete"
  >;
  managedBoatOwners(): boat.ManagedBoatOwner[];
  providerOperationConflict(provider: RemoteComputerProvider): string | null;
  perBotLocalVmCountForModeChange(): Promise<number | null>;
  checkedModelSelection(
    raw: unknown,
    current?: { selection: ModelSelection; busy: boolean },
    requireAvailableModel?: boolean
  ): ModelSelectionCheck;
  policyModelRefusal(instance: {
    instanceId: string;
    driverKind: string;
  }): string | undefined;
  browserSessionSpec(
    binaryPath: string,
    session: string,
    profile: string | undefined
  ): { command: string; env: Record<string, string> };
  configStatus(): ConfigStatus;
  configForAccess(status: ConfigStatus, admin: boolean): unknown;
  resyncOpenCodeProviderKeys(): void;
  revokeEngineCapabilities(engines: ReadonlySet<string>): void;
  reloadProviders(before: Record<string, unknown>): Promise<void>;
  broadcast(payload: Record<string, unknown>): void;
  drainQueuedSends(): void;
}

export function createConfigRoutes(deps: ConfigRouteDeps): RouteHandler {
  const {
    store,
    cfg,
    registry,
    sessions,
    browserCleanup,
    browserLive,
    browserRuntime,
    usedBrowserSessions,
    decider,
    sharedComputers,
    sharedComputerControl,
    hostedModels,
    cloudHome: CLOUD_HOME,
    lendingEnabled,
    providerConfigs,
    localVmIdleMs,
    localVmIdles,
    localVmActiveThreads,
    localVmLifecycleBusy,
    localVmImageBusy,
    localVmModeChangeBusy,
    providerConfigBusy,
    computerProviderConfigTransitions,
    managedBoatOwners,
    providerOperationConflict,
    perBotLocalVmCountForModeChange,
    checkedModelSelection,
    policyModelRefusal,
    browserSessionSpec,
    configStatus,
    configForAccess,
    resyncOpenCodeProviderKeys,
    revokeEngineCapabilities,
    reloadProviders,
    broadcast,
    drainQueuedSends,
  } = deps;
  return async ({ req, res, url, path, method, auth, json, readBody }) => {
    // ── app config (API keys — never echoed back, booleans only) ──
    if (method === "GET" && path === "/api/config") {
      return json(
        res,
        200,
        configForAccess(configStatus(), auth.scopes.includes("admin"))
      );
    }
    if ((method === "PUT" || method === "PATCH") && path === "/api/config") {
      const body = await readBody(req);
      if (
        hostedModels &&
        [
          "instances",
          "anthropic",
          "openai",
          "openrouter",
          "openaiCompat",
          "xai",
          "opencodeGo",
        ].some((key) => Object.hasOwn(body, key))
      )
        return json(res, 403, { error: HOSTED_PROVIDER_SETTINGS_ERROR });
      const patch = parseConfigPatch(body);
      // A Cloud home is personal: nobody is invited to sign in to it.
      if (
        CLOUD_HOME &&
        (patch.signIn?.admins?.length || patch.signIn?.members?.length)
      ) {
        return json(res, 403, {
          error: CLOUD_PERSONAL_REFUSAL,
          code: "cloud_personal",
        });
      }
      if (patch.newBotDefaults) {
        if (patch.newBotDefaults.profile.modelSelection) {
          const checked = checkedModelSelection(
            patch.newBotDefaults.profile.modelSelection
          );
          if (!checked.ok)
            return json(res, checked.status, { error: checked.error });
          patch.newBotDefaults.profile.modelSelection = checked.selection;
        }
        for (const skill of patch.newBotDefaults.skills) {
          const parsed = parseSkillMd(skill.text);
          if ("error" in parsed || parsed.name !== skill.name)
            return json(res, 400, {
              error: "Skill name does not match its contents",
            });
        }
      }
      if (hostedModels && patch.defaultModelSelection) {
        const checked = checkedModelSelection(patch.defaultModelSelection);
        if (!checked.ok)
          return json(res, checked.status, { error: checked.error });
      }
      if (patch.automaticRecovery?.enabled) {
        const checked = checkedModelSelection(
          patch.automaticRecovery.backup,
          undefined,
          true
        );
        if (!checked.ok)
          return json(res, checked.status, { error: checked.error });
        const target = registry.get(checked.selection.instanceId)!;
        if (!target.enabled)
          return json(res, 400, {
            error: "Enable the backup engine before selecting it for recovery.",
          });
        const refusal = policyModelRefusal(target);
        if (refusal)
          return json(res, 403, { error: refusal, code: "managed_policy" });
        patch.automaticRecovery.backup = checked.selection;
      }
      if (!Object.keys(patch).length)
        return json(res, 400, { error: "nothing to save" });
      if (patch.decider) {
        // Saving a key is the "turn it on"; clearing it turns it off, or
        // falls back to Cloud Pro's included decisions where there are some.
        const planned = deciderSavePatch(
          patch.decider,
          cfg.decider,
          deciderIncludedHere()
        );
        if (!planned.ok) return json(res, 400, { error: planned.error });
        patch.decider = planned.patch;
      }
      const changingVoiceProvider =
        patch.tts?.provider !== undefined &&
        patch.tts.provider !== tts.voiceProvider(cfg);
      if (changingVoiceProvider && patch.tts?.voice === undefined) {
        // Voice ids are provider-owned opaque values. Never carry a default
        // from one catalog into another. A caller may explicitly supply a
        // voice for the newly selected provider in this same atomic patch.
        patch.tts = { ...patch.tts, voice: "" };
      }
      if (providerConfigBusy.get())
        return json(res, 409, {
          error: "provider settings are already being updated",
        });
      if (
        patch.browserProfiles !== undefined &&
        body.expectedBrowserProfiles !== undefined
      ) {
        const current = (cfg.browserProfiles ?? []).map(
          ({ id, name, browser }) => ({
            id,
            name,
            ...(browser ? { browser } : {}),
          })
        );
        if (
          JSON.stringify(body.expectedBrowserProfiles) !==
          JSON.stringify(current)
        ) {
          return json(res, 409, {
            error:
              "Browser profiles changed in another window. Review the refreshed list and try again.",
          });
        }
      }
      const disablingBuiltInBrowser =
        patch.features?.browser === false && builtInBrowserEnabled(cfg);
      const removedBrowserProfileIds =
        patch.browserProfiles === undefined
          ? []
          : (cfg.browserProfiles ?? [])
              .map((profile) => profile.id)
              .filter(
                (id) =>
                  !patch.browserProfiles!.some((profile) => profile.id === id)
              );
      // Profiles whose browser changes between built-in and My Chrome: their
      // open browser closes after the save, so the next turn uses the new one.
      const browserChoiceChanged = (patch.browserProfiles ?? [])
        .filter((profile) => {
          const current = cfg.browserProfiles?.find(
            (candidate) => candidate.id === profile.id
          );
          return (
            current !== undefined &&
            (current.browser ?? null) !== (profile.browser ?? null)
          );
        })
        .map((profile) => profile.id);
      const profileControlConflict = () =>
        removedBrowserProfileIds.some((id) => {
          const target = browserProfilePartitionTarget(cfg, id);
          return (
            target &&
            browserRuntime.heldBy(browserSessionId("", target.partitionId))
          );
        });
      if (profileControlConflict())
        return json(res, 409, {
          error: "Release browser control before deleting its profile.",
        });
      if (
        browserChoiceChanged.some((id) => {
          const target = browserProfilePartitionTarget(cfg, id);
          return (
            target &&
            browserRuntime.heldBy(browserSessionId("", target.partitionId))
          );
        })
      )
        return json(res, 409, {
          error: "Release browser control before changing its browser.",
        });
      if (patch.browserProfiles !== undefined) {
        const currentProfiles = new Map(
          (cfg.browserProfiles ?? []).map((profile) => [profile.id, profile])
        );
        const nextProfiles = patch.browserProfiles.map((profile) => {
          const partitionId = currentProfiles.get(profile.id)?.partitionId;
          return partitionId ? { ...profile, partitionId } : profile;
        });
        const routingConflict = browserProfileReplacementConflict(
          cfg.browserProfiles ?? [],
          nextProfiles
        );
        if (routingConflict) return json(res, 409, { error: routingConflict });
        const currentIds = new Set(
          (cfg.browserProfiles ?? []).map((profile) => profile.id)
        );
        const pendingReuse = patch.browserProfiles.find(
          (profile) =>
            !currentIds.has(profile.id) &&
            browserCleanup.hasPendingProfile(profile.id)
        );
        if (pendingReuse) {
          return json(res, 409, {
            error: `the previous “${pendingReuse.name}” browser session is still being erased — wait before reusing it`,
          });
        }
      }
      if (patch.browserProfiles !== undefined) {
        const retained = new Set(
          patch.browserProfiles.map((profile) => profile.id)
        );
        const activeReference = store.bots.find(
          (bot) =>
            bot.busy &&
            bot.browserProfile &&
            bot.browserProfile !== "guest" &&
            !retained.has(bot.browserProfile)
        );
        if (activeReference) {
          return json(res, 409, {
            error: `stop ${activeReference.name}'s turn before removing its browser profile`,
          });
        }
      }
      if (patch.box?.token !== undefined)
        patch.box.token = patch.box.token.trim();
      const currentBoatToken = cfg.box?.token?.trim() ?? "";
      const nextBoatToken =
        patch.box?.token === undefined ? currentBoatToken : patch.box.token;
      const changingBoatToken =
        patch.box?.token !== undefined && nextBoatToken !== currentBoatToken;
      const currentVpsAlias = vpsSshAlias(cfg);
      const nextVpsAlias =
        patch.vps === undefined
          ? currentVpsAlias
          : vpsSshAlias({ ...cfg, vps: patch.vps });
      const changingVpsAlias =
        patch.vps !== undefined && nextVpsAlias !== currentVpsAlias;
      const transitioningProviders: RemoteComputerProvider[] = [
        ...(changingBoatToken ? ["box" as const] : []),
        ...(changingVpsAlias ? ["vps" as const] : []),
      ];
      providerConfigBusy.set(true);
      const changingLocalVmMode =
        patch.localVm?.mode !== undefined &&
        patch.localVm.mode !== localVmMode(cfg);
      if (changingLocalVmMode) localVmModeChangeBusy.set(true);
      try {
        for (const provider of transitioningProviders) {
          const conflict = providerOperationConflict(provider);
          if (conflict) return json(res, 409, { error: conflict });
        }
        for (const provider of transitioningProviders)
          computerProviderConfigTransitions.add(provider);

        if (changingVpsAlias && currentVpsAlias) {
          const inventory = await vps.listManagedVpsComputers(
            { vps: { sshAlias: currentVpsAlias } },
            managedBoatOwners()
          );
          if (!inventory.available) {
            return json(res, 503, {
              error: `${
                inventory.problem ?? "VPS computer inventory is unavailable"
              }. Keep the current SSH config alias and retry`,
            });
          }
          const resourceError = vpsAliasResourceChangeError(
            inventory.instances.length
          );
          if (resourceError) return json(res, 409, { error: resourceError });
        }

        let boatRecovery = changingBoatToken
          ? boatCreateRecoverySnapshot()
          : [];
        let boatDeletions = changingBoatToken ? boatDeletionSnapshot() : [];
        if (changingBoatToken && boatDeletions.length > 0 && !nextBoatToken) {
          return json(res, 409, {
            error:
              "finish or retry pending cloud computer deletion before removing the Boat account",
          });
        }
        let replacementProvedByDeletion = false;
        if (changingBoatToken && boatDeletions.length > 0 && nextBoatToken) {
          try {
            await boat.verifyBoatDeletionCredential({
              box: { token: nextBoatToken },
            });
            replacementProvedByDeletion = true;
            // Verification can observe a completed operation and retire both
            // its deletion fence and matching create receipt. Never continue
            // with the pre-verification snapshots: they would demand access
            // to a Boat whose exact operation just proved it was deleted.
            boatRecovery = boatCreateRecoverySnapshot();
            boatDeletions = boatDeletionSnapshot();
          } catch (error) {
            return json(res, (error as { status?: number })?.status ?? 503, {
              error: error instanceof Error ? error.message : String(error),
            });
          }
        }
        let currentBoatInventory: boat.ManagedBoatInventory | null = null;
        let replacingRejectedBoatToken = false;
        let currentBoatResources: Array<{
          boxId: string;
          name: string;
        }> | null = null;
        const journalBoatResources: Array<{ boxId: string; name: string }> = [];
        const deletingBoatIds = new Set(
          boatDeletions.map((entry) => entry.boxId)
        );
        if (changingBoatToken && currentBoatToken) {
          currentBoatInventory = await boat.listManagedBoats(
            { box: { token: currentBoatToken } },
            managedBoatOwners()
          );
          if (!currentBoatInventory.available) {
            replacingRejectedBoatToken =
              currentBoatInventory.credentialRejected === true &&
              Boolean(nextBoatToken);
            if (!replacementProvedByDeletion && !replacingRejectedBoatToken) {
              return json(res, 503, {
                error: `${
                  currentBoatInventory.problem ??
                  "cloud computer inventory is unavailable"
                }. Keep the current Boat account and retry`,
              });
            }
            // A rejected old key cannot authorize its own rotation. Below,
            // validate the replacement and prove access to remembered Boats;
            // pending deletions retain their target-bound verification.
            currentBoatInventory = null;
          }
          if (currentBoatInventory) {
            const currentById = new Map(
              currentBoatInventory.instances.map((instance) => [
                instance.boxId,
                { boxId: instance.boxId, name: instance.name },
              ])
            );
            for (const recovery of boatRecovery) {
              if (!recovery.boxId) continue;
              // A failed provisioning attempt may never have reached the
              // deterministic rename. The replacement credential already
              // proved the exact deletion target, so its in-flight resource
              // is governed by that stronger target-bound receipt rather
              // than an OpenMausBot name check.
              if (
                replacementProvedByDeletion &&
                deletingBoatIds.has(recovery.boxId)
              )
                continue;
              const inspected = await boat.inspectBoatIdentity(
                { box: { token: currentBoatToken } },
                recovery.boxId
              );
              if (!inspected.available) {
                return json(res, 503, {
                  error: `${
                    inspected.problem ??
                    "a remembered cloud computer could not be verified"
                  }. Keep the current Boat account and retry`,
                });
              }
              if (!inspected.identity) {
                // Reconcile exact stale receipts while the current credential is
                // still available. Leaving one behind would make a later token
                // addition demand access to a Boat the provider proved is gone.
                retireDeletedBoatCreate(recovery.boxId);
                continue;
              }
              const listed = currentById.get(inspected.identity.boxId);
              if (listed && listed.name !== inspected.identity.name) {
                return json(res, 503, {
                  error:
                    "boat.dev returned conflicting cloud computer identities; keep the current Boat account and retry",
                });
              }
              currentById.set(inspected.identity.boxId, inspected.identity);
              journalBoatResources.push(inspected.identity);
            }
            currentBoatResources = [...currentById.values()];
          }
        }

        if (changingLocalVmMode) {
          if (
            localVmActiveThreads.size > 0 ||
            localVmLifecycleBusy.size > 0 ||
            localVmImageBusy.get()
          ) {
            return json(res, 409, {
              error:
                "stop Local VM turns and setup actions before changing the Local VM isolation mode",
            });
          }
          if (
            localVmMode(cfg) === "per-bot" &&
            patch.localVm?.mode === "shared"
          ) {
            const existing = await perBotLocalVmCountForModeChange();
            if (existing === null) {
              return json(res, 409, {
                error:
                  "start the container runtime and delete every per-bot VM before switching to shared mode",
              });
            }
            if (existing > 0) {
              return json(res, 409, {
                error: `delete the ${existing} per-bot Local VM${
                  existing === 1 ? "" : "s"
                } before switching to shared mode`,
              });
            }
          }
        }
        // A project key is useful only if it can create/reuse the Session that
        // powers both the connections UI and the agent MCP. Validate it before
        // persisting, and save the non-secret ids needed to reuse that Session.
        const requestedComposioKey = patch.composio?.apiKey;
        if (requestedComposioKey !== undefined) {
          if (requestedComposioKey.trim()) {
            try {
              const prepared = await composio.prepareProjectSession(
                requestedComposioKey,
                cfg.composio
              );
              patch.composio = { ...patch.composio, ...prepared };
            } catch (error) {
              return json(res, 400, {
                error: error instanceof Error ? error.message : String(error),
              });
            }
          } else {
            patch.composio = { ...patch.composio, apiKey: "", sessionId: "" };
          }
        }
        // check a boat token against the provider before storing it: a
        // rejected token used to save happily and only surface as a 401 in
        // another panel later, with nothing the user could act on
        const newBoatToken = patch.box?.token;
        if (newBoatToken?.trim()) {
          const check = await boat.verifyToken(newBoatToken);
          if (!check.ok) return json(res, 400, { error: check.message });
        }
        if (replacingRejectedBoatToken) {
          // An expired credential cannot authorize its own replacement. Probe
          // the new account without adopting or retiring any local identities.
          const replacement = await boat.listManagedBoats(
            { box: { token: nextBoatToken } },
            managedBoatOwners(),
            { adoptLegacy: false }
          );
          if (!replacement.available)
            return json(res, 503, {
              error:
                replacement.problem ??
                "Could not verify the replacement Boat key. Your saved key and computers are unchanged.",
            });
        }
        if (
          changingBoatToken &&
          (!currentBoatToken ||
            replacementProvedByDeletion ||
            replacingRejectedBoatToken) &&
          boatRecovery.length > 0
        ) {
          if (!nextBoatToken) {
            return json(res, 409, {
              error:
                "restore the Boat account that owns the remembered cloud computers before clearing it",
            });
          }
          for (const recovery of boatRecovery) {
            if (!recovery.boxId) {
              return json(res, 409, {
                error:
                  "finish reconciling pending cloud computer creation before changing the Boat account",
              });
            }
            if (
              replacementProvedByDeletion &&
              deletingBoatIds.has(recovery.boxId)
            )
              continue;
            const inspected = await boat.inspectBoatIdentity(
              { box: { token: nextBoatToken } },
              recovery.boxId
            );
            if (!inspected.available) {
              return json(res, 503, {
                error: `${
                  inspected.problem ??
                  "a remembered cloud computer could not be verified"
                }. Retry with the Boat account that created it`,
              });
            }
            if (
              !inspected.identity ||
              !(await boat.boatNameMatchesBot(
                recovery.botId,
                inspected.identity.name
              ))
            ) {
              return json(res, 409, {
                error:
                  "that Boat token cannot access the remembered cloud computers from this installation",
              });
            }
          }
        }
        if (changingBoatToken && currentBoatInventory && currentBoatResources) {
          let replacementResources: Array<{
            boxId: string;
            name: string;
          }> | null = null;
          if (nextBoatToken) {
            const replacementInventory = await boat.listManagedBoats(
              { box: { token: nextBoatToken } },
              managedBoatOwners(),
              { adoptLegacy: false }
            );
            if (!replacementInventory.available) {
              return json(res, 503, {
                error: `${
                  replacementInventory.problem ??
                  "cloud computer inventory is unavailable"
                }. Keep the current Boat account and retry`,
              });
            }
            replacementResources = replacementInventory.instances.map(
              (instance) => ({
                boxId: instance.boxId,
                name: instance.name,
              })
            );
            const replacementById = new Map(
              replacementResources.map((instance) => [
                instance.boxId,
                { boxId: instance.boxId, name: instance.name },
              ])
            );
            for (const identity of journalBoatResources) {
              const inspected = await boat.inspectBoatIdentity(
                { box: { token: nextBoatToken } },
                identity.boxId
              );
              if (!inspected.available) {
                return json(res, 503, {
                  error: `${
                    inspected.problem ??
                    "a remembered cloud computer could not be verified"
                  }. Keep the current Boat account and retry`,
                });
              }
              if (
                !inspected.identity ||
                inspected.identity.name !== identity.name
              ) {
                return json(res, 409, {
                  error:
                    "the replacement Boat token does not access the same cloud computers",
                });
              }
              replacementById.set(inspected.identity.boxId, inspected.identity);
            }
            replacementResources = [...replacementById.values()];
          }
          const resourceError = boatAccountResourceChangeError(
            currentBoatResources,
            replacementResources
          );
          if (resourceError) return json(res, 409, { error: resourceError });
        }
        // Each cloud voice provider owns its own credential. Validate the field
        // against that provider rather than whichever provider happens to be
        // selected, so switching engines never sends one service another's key.
        const newTts = patch.tts;
        if (newTts?.key?.trim()) {
          const check = await tts.verifyKey("elevenlabs", newTts.key.trim());
          if (!check.ok) return json(res, 400, { error: check.message });
        }
        if (newTts?.fishKey?.trim()) {
          const check = await tts.verifyKey("fish", newTts.fishKey.trim());
          if (!check.ok) return json(res, 400, { error: check.message });
        }
        // A pasted decision-model key is tried once (one tiny yes/no call).
        // Only a refusal blocks the save: an offline or overloaded service must
        // not stop a person saving a key that may well be fine.
        if (patch.decider?.key) {
          const check = await decider.testKey({ key: patch.decider.key });
          if (!check.ok && check.reason === "rejected") {
            return json(res, 400, {
              error:
                "Jev rejected this key. Check it at typesafe.ai and paste it again.",
              code: "decider_rejected",
            });
          }
        }
        if (patch.browserProfiles !== undefined) {
          // Provider/credential validation above may await the network. A turn
          // can start during that window and claim a profile which looked idle
          // at the route's first check, so validate again at the mutation
          // boundary. Keep this check and the synchronous save/reference cleanup
          // below free of awaits.
          const retained = new Set(
            patch.browserProfiles.map((profile) => profile.id)
          );
          const activeReference = store.bots.find(
            (bot) =>
              bot.busy &&
              bot.browserProfile &&
              bot.browserProfile !== "guest" &&
              !retained.has(bot.browserProfile)
          );
          if (activeReference) {
            return json(res, 409, {
              error: `stop ${activeReference.name}'s turn before removing its browser profile`,
            });
          }
          // Built-in and My Chrome are different browsers: a turn keeps the
          // one it started with, so the switch waits for it.
          const switching = store.bots.find(
            (bot) =>
              bot.busy &&
              bot.browserProfile &&
              browserChoiceChanged.includes(bot.browserProfile)
          );
          if (switching)
            return json(res, 409, {
              error: `stop ${switching.name}'s turn before changing its browser`,
            });
        }
        // Provider validation above awaits remote services. The transition flag
        // blocks new work, while this second observation catches any operation
        // that already held a claim at the initial boundary.
        for (const provider of transitioningProviders) {
          const conflict = providerOperationConflict(provider);
          if (conflict) return json(res, 409, { error: conflict });
        }
        // A change names only the OpenCode provider keys it saves or removes;
        // the rest stay. Merged with no await before the save below, so it
        // starts from what is saved at that moment.
        if (patch.opencodeGo?.providerKeys) {
          const merged = mergeOpenCodeProviderKeys(
            cfg,
            patch.opencodeGo.providerKeys
          );
          if (!merged.ok) return json(res, 400, { error: merged.error });
          patch.opencodeGo = { ...patch.opencodeGo, providerKeys: merged.keys };
        }
        const browserCleanupRequests: BrowserCleanupRequest[] = [];
        if (profileControlConflict())
          return json(res, 409, {
            error: "Release browser control before deleting its profile.",
          });
        try {
          for (const profileId of removedBrowserProfileIds) {
            const target = browserProfilePartitionTarget(cfg, profileId);
            if (!target)
              throw new Error(
                `browser profile cleanup target “${profileId}” is unavailable`
              );
            browserCleanupRequests.push(
              browserCleanup.prepare(
                "profile",
                target.profileId,
                target.partitionId
              )
            );
          }
        } catch (error) {
          for (const request of browserCleanupRequests)
            browserCleanup.abort(request);
          throw error;
        }
        // A saved or cleared key starts over: neither it nor the key it
        // replaced stays marked as rejected.
        const resetKeys = PROVIDER_KEY_KINDS.flatMap((kind) => {
          const next = patch[kind]?.key;
          return next === undefined ? [] : [cfg[kind]?.key, next];
        });
        const engineForSwitch = browserEngineStatus();
        const browsersToClose =
          engineForSwitch.kind === "ready"
            ? browserChoiceChanged.flatMap((id) => {
                const target = browserProfilePartitionTarget(cfg, id);
                if (!target) return [];
                const session = browserSessionId("", target.partitionId);
                return [
                  {
                    session,
                    spec:
                      usedBrowserSessions.get(session) ??
                      browserSessionSpec(
                        engineForSwitch.binaryPath,
                        session,
                        id
                      ),
                  },
                ];
              })
            : [];
        let configWriteCommitted = false;
        // What each engine is launched with before this write (reloadProviders).
        const enginesBefore = engineLaunches(providerConfigs());
        const externalSecretStorage =
          url.searchParams.get("secretStorage") === "external";
        try {
          // Provider-owned voice ids must be invalidated before the provider
          // commit. If bots.json cannot be written, leave the old provider in
          // place rather than committing a new provider with stale bot voices.
          // A later config-write failure may leave voices cleared, which is the
          // safe side of this cross-file mutation: no foreign id can be spoken.
          if (changingVoiceProvider) store.clearVoiceSelections();
          if (externalSecretStorage) {
            // The packaged Electron caller commits supplied credentials to the
            // OS-encrypted store before entering this route. Persist every
            // non-secret sibling in the same request, but replace each supplied
            // credential with an empty tombstone so an older plaintext value can
            // never survive the merge in config.json.
            const persisted = structuredClone(patch);
            if (persisted.xai?.key !== undefined) persisted.xai.key = "";
            if (persisted.composio?.apiKey !== undefined)
              persisted.composio.apiKey = "";
            if (persisted.box?.token !== undefined) persisted.box.token = "";
            if (persisted.opencodeGo?.apiKey !== undefined)
              persisted.opencodeGo.apiKey = "";
            if (persisted.tts?.key !== undefined) persisted.tts.key = "";
            if (persisted.tts?.fishKey !== undefined)
              persisted.tts.fishKey = "";
            if (persisted.decider?.key !== undefined)
              persisted.decider.key = "";
            if (persisted.imageGen?.key !== undefined)
              persisted.imageGen.key = "";
            if (persisted.imageGen?.customApiKey !== undefined)
              persisted.imageGen.customApiKey = "";
            if (persisted.live?.key !== undefined) persisted.live.key = "";
            saveConfig(persisted);
            configWriteCommitted = true;
            syncCredentialEnv(patch);
            Object.assign(cfg, loadConfig());
          } else {
            saveConfig(patch);
            configWriteCommitted = true;
            // loadConfig prefers env over the file for credentials, so the env
            // must follow the save — otherwise the value injected at boot would
            // shadow the new key until the next launch
            syncCredentialEnv(patch);
            Object.assign(cfg, loadConfig());
          }
        } catch (error) {
          if (configWriteCommitted) {
            for (const request of browserCleanupRequests) {
              const committed = browserCleanup.commit(request);
              void browserCleanup.ensure(committed);
            }
          } else {
            for (const request of browserCleanupRequests)
              browserCleanup.abort(request);
          }
          throw error;
        }
        for (const key of resetKeys) if (key) forgetKey(key);
        // Desktops already counting down move to the new window at once, in
        // every isolation mode; busy work still defers its expiry.
        if (patch.localVm?.idleTimeoutMinutes !== undefined) {
          for (const idle of localVmIdles.values())
            idle.setIdleMs(localVmIdleMs());
        }
        let browserReferenceCleanupError: unknown = null;
        if (patch.signIn !== undefined) {
          sessions.revalidateEmailSessions();
          resyncOpenCodeProviderKeys();
        }
        if (!lendingEnabled()) {
          sharedComputers.close();
          sharedComputerControl.close();
        }
        if (disablingBuiltInBrowser) browserLive.closeAll();
        for (const request of browserCleanupRequests) {
          if (request.kind === "profile")
            browserLive.closeForSession(
              browserSessionId("", request.partitionId)
            );
        }
        for (const { session, spec } of browsersToClose) {
          browserLive.closeForSession(session);
          usedBrowserSessions.delete(session);
          // Closing a built-in browser saves its sign-ins for when it comes
          // back; closing an attached one only lets go of the person's Chrome.
          void closeBrowserSession(
            spec.command,
            browserRuntimeEnv({ ...spec.env, AGENT_BROWSER_SESSION: session })
          ).then((closed) => {
            if (!closed)
              console.warn(
                `browser ${session}: could not close it after its browser changed; the next turn may still use the old one`
              );
          });
        }
        if (patch.browserProfiles !== undefined) {
          const retained = new Set(
            patch.browserProfiles.map((profile) => profile.id)
          );
          try {
            for (const bot of store.bots) {
              if (
                bot.browserProfile &&
                bot.browserProfile !== "guest" &&
                !retained.has(bot.browserProfile)
              ) {
                // The profile list and every bot reference change in the same
                // config request. Non-renderer clients therefore cannot leave a
                // bot pointing at a deleted cookie partition.
                store.patchBot(bot.id, { browserProfile: undefined });
              }
            }
          } catch (error) {
            // Config is already durable. Keep the cleanup intent prepared (so
            // it cannot wipe ambiguous state and its id remains locked), but do
            // not let this secondary write failure skip revocation/reload below.
            browserReferenceCleanupError = error;
          }
        }
        // Only an engine whose launch config this write changed is rebuilt: a
        // provider key, URL or model default reaches its own engines and no
        // others, and a setting no driver reads (profile, language, voice, VPS,
        // room timeout, onboarding progress), or the same values saved again,
        // rebuilds nothing and interrupts no turn.
        const changedEngines = new Set(
          changedInstanceIds(enginesBefore, engineLaunches(providerConfigs()))
        );
        // Config is already durable. A changed engine's old children are
        // invalid immediately, including when browser cleanup below has to
        // await Electron before reloadProviders begins.
        revokeEngineCapabilities(changedEngines);
        // The cleanup marker becomes committed only after both pieces of durable
        // application state agree. Commit/ACK failures are deferred until every
        // mandatory consequence of the config write has run: no journal I/O
        // failure may leave a two-hour bearer or stale provider fleet active.
        const finalized = await finalizeBrowserCleanupMutation({
          requests: browserCleanupRequests,
          referenceError: browserReferenceCleanupError,
          commit: (request) => browserCleanup.commit(request),
          ensure: (request) => browserCleanup.ensure(request),
          mandatory: async () => {
            let mandatoryError: unknown = null;
            if (disablingBuiltInBrowser) {
              try {
              } catch (error) {
                mandatoryError = error;
              }
            }
            if (changedEngines.size > 0) {
              try {
                await reloadProviders(enginesBefore);
              } catch (error) {
                if (!mandatoryError) mandatoryError = error;
              }
            }
            const status = configStatus();
            broadcast({ kind: "config", ...status });
            if (patch.threads !== undefined) {
              drainQueuedSends();
            }
            if (mandatoryError) throw mandatoryError;
            return status;
          },
        });
        // Normal desktop deletes wait for Electron's acknowledgement. If
        // Electron is restarting, the committed journal keeps retrying and the
        // id-reuse guard above prevents stale logins from resurfacing. Delaying
        // this assertion until after every mandatory post-commit effect keeps
        // the runtime aligned with the config even on a truthful 503 response.
        requireBrowserCleanupAcknowledged(
          finalized.acknowledgements.every(Boolean),
          removedBrowserProfileIds.length === 1
            ? "The browser profile"
            : "The browser profiles"
        );
        return json(res, 200, finalized.value);
      } finally {
        for (const provider of transitioningProviders)
          computerProviderConfigTransitions.delete(provider);
        if (changingLocalVmMode) localVmModeChangeBusy.set(false);
        providerConfigBusy.set(false);
      }
    }
    return PASS;
  };
}
