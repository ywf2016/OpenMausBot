import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import {
  ApprovalModeSelector,
  approvalModeOptions,
  approvalModeOptionsFor,
  approvalModeSelectionRequiresLocalDesktop,
} from "./ApprovalModeSelector";

describe("approval mode selector", () => {
  it("lists the approval levels with their plain-language copy", () => {
    expect(approvalModeOptions().map(({ mode, label, description }) => ({ mode, label, description }))).toEqual([
      {
        mode: "ask",
        label: "Ask for approval",
        description: "Requests approval for commands and file changes",
      },
      {
        mode: "edits",
        label: "Auto-accept edits",
        description: "Approves file edits automatically; other actions can still require approval",
      },
      {
        mode: "auto",
        label: "Approve for me",
        description: "Uses the provider's automatic review to approve routine actions and ask about others. Web search is allowed without asking.",
      },
      {
        mode: "full",
        label: "Full access",
        description: "Full computer access (elevated risk)",
      },
      {
        mode: "custom",
        label: "Custom (config.toml)",
        description: "Uses permissions defined in config.toml",
      },
    ]);
  });

  it("offers provider-supported Full access but keeps config.toml Codex-only", () => {
    expect(approvalModeOptionsFor("codex").map((option) => option.mode)).toEqual([
      "ask",
      "auto",
      "full",
      "custom",
    ]);
    expect(approvalModeOptionsFor("claudeAgent").map((option) => option.mode)).toEqual([
      "ask",
      "edits",
      "auto",
      "full",
    ]);
  });

  it.each(["cursorAgent", "opencodeGo"])("offers Full access for %s, and no Edits level it cannot map", (kind) => {
    expect(approvalModeOptionsFor(kind).map((option) => option.mode)).toEqual(["ask", "auto", "full"]);
    expect(approvalModeOptionsFor(kind, false).map((option) => option.mode)).toEqual(["ask", "auto", "full"]);
  });

  it("offers Qwen its native auto-edit as Auto-accept edits", () => {
    expect(approvalModeOptionsFor("qwenAgent").map((option) => option.mode)).toEqual(["ask", "edits", "auto", "full"]);
    expect(approvalModeOptionsFor("qwenAgent", false).map((option) => option.mode)).toEqual(["ask", "edits", "auto", "full"]);
  });

  it("explains Auto fallbacks and does not elevate unknown providers", () => {
    expect(approvalModeOptionsFor("qwenAgent").find((option) => option.mode === "auto")?.description).toContain("automatic review to approve routine actions");
    expect(approvalModeOptionsFor("claudeAgent").find((option) => option.mode === "auto")?.description).toContain("automatic review to approve routine actions");
    expect(approvalModeOptionsFor("customAgent").find((option) => option.mode === "auto")?.description).toContain("behaves like Ask");
    expect(approvalModeOptionsFor("customAgent").map((option) => option.mode)).toEqual(["ask", "auto"]);
  });

  it("offers Full without the packaged desktop bridge and hides only Custom", () => {
    expect(approvalModeOptionsFor("codex", false).map((option) => option.mode)).toEqual([
      "ask",
      "auto",
      "full",
    ]);
    expect(approvalModeOptionsFor("claudeAgent", false).map((option) => option.mode)).toEqual(["ask", "edits", "auto", "full"]);
  });

  it("keeps the composer trigger icon-only and the settings trigger labeled", () => {
    const compact = renderToStaticMarkup(createElement(ApprovalModeSelector, {
      approvalMode: "full", providerName: "Claude", driverKind: "claudeAgent", onSelect: () => {},
    }));
    expect(compact).toContain('aria-label="Full access for Claude"');
    expect(compact).toContain('title="Full access"');
    expect(compact).not.toMatch(/<span class="truncate">Full access<\/span>/);
    const settings = renderToStaticMarkup(createElement(ApprovalModeSelector, {
      approvalMode: "full", providerName: "Claude", driverKind: "claudeAgent", onSelect: () => {}, wide: true,
    }));
    expect(settings).toMatch(/<span class="truncate">Full access<\/span>/);
  });

  it("locks an existing Custom bot to the local packaged desktop", () => {
    expect(approvalModeSelectionRequiresLocalDesktop("custom", false)).toBe(true);
    expect(approvalModeSelectionRequiresLocalDesktop("ask", false)).toBe(false);
    expect(approvalModeSelectionRequiresLocalDesktop("custom", true)).toBe(false);
    const html = renderToStaticMarkup(createElement(ApprovalModeSelector, {
      approvalMode: "custom", trustedModesAvailable: false,
      providerName: "Missing provider", driverKind: "", onSelect: () => {},
    }));
    expect(html).toContain('aria-label="Custom (config.toml) for Missing provider"');
  });
});
