# 10. Fire control and safety

Read this fully before enabling any fire control, manual or automatic.
Fire control code exists in two places in this repo, added under
explicit, deliberate confirmation — this is not something either script
does by default or silently.

## Where fire control exists

| Location | Manual fire | Auto fire |
|---|---|---|
| `20RajRif/Old/main.py` | FIRE button (hold) | AUTO FIRE toggle, pre-existing |
| `drone_tracking_module/detect_and_track.py` | `f` key | `r`-armed, added this session |
| `20RajRif/tracker_gui.py` | fire checkbox (manual only) | **none — never add one here** |

## `detect_and_track.py` — exact behavior

**Manual FIRE (`f`)** — toggles directly, any time, either RAW or LOCK
mode. Zero gating. Pure operator judgment, same as a manual trigger has
always worked elsewhere in this repo.

**AUTO FIRE (`r` arms/disarms)** — while armed, fires only when **ALL** of
the following hold, checked every single frame:

1. `mode == "lock"` — never in RAW mode. RAW has no identity check at all,
   the same reason `Old/main.py` disables AUTO FIRE in BLOB mode:
   "locked" there doesn't mean "confirmed the same target."
2. `tracker.state == "lock"` — never on a coasted prediction or a search
   guess.
3. The target is centered — `zone < 30` (same formula and threshold as
   `Old/main.py`'s legacy condition).

These are **the same three gates** `Old/main.py`'s AUTO FIRE already used
before this session — re-implemented for `detect_and_track.py`'s own
state, nothing about the proven design changed.

**Panic stop (`space`)** — unconditionally disarms AUTO FIRE, forces
manual fire off, and stops all movement. One key, no ambiguity, works
regardless of current state.

**Fail-safe on exit** — `mount.fire(False)` is called on every exit path
(clean quit, Ctrl+C, or an unexpected crash falling through to the
`finally` block). `mount.stop()` alone does **not** stop firing — fire is
a separate command from movement, and this is called out explicitly in
the code because it's easy to assume otherwise.

## `Old/main.py`'s pre-existing AUTO FIRE gates (unchanged)

From `CLAUDE.md`: fires only in the `lock` state (never coast/search), and
only in PROFILE mode (never BLOB, since "locked" in blob mode just means
"a dark blob," which can be a bird). The legacy centered/`zone<30`
condition applies on top. **Do not remove either gate.**

## The auto-acquire + auto-fire interaction — read this specifically

This session added optional YOLO auto-acquire to `Old/main.py` and
`tracker_gui.py`. In `Old/main.py`, **an auto-acquired lock is exactly as
eligible for AUTO FIRE as a manually-acquired one** — the fire gate only
checks `state == "lock"`, not how the lock was originated. This was an
explicit, confirmed decision during development, not an oversight — but it
means enabling both AUTO-ACQUIRE and AUTO FIRE together is a materially
higher-stakes combination than either one alone. `tracker_gui.py` has no
fire capability at all, so this combination doesn't exist there.

## What auto-fire still cannot defend against

Both fire-gating systems assume the underlying lock is trustworthy. The
detection model is single-class (`drone` only) — it has no explicit
"definitely not a drone" signal, so a bird/kite/plane that's small,
confident, and holds still relative to the frame for the confirm window
could theoretically pass every gate. The confidence/size/persistence
filters substantially reduce this risk (see `09-old-new-raw-lock.md`), but
don't eliminate it. **The manual clear / panic-stop remains the real
backstop** — keep a hand near it whenever auto-fire is armed.

## Recommended operating procedure

1. Never arm AUTO FIRE as a default/startup state — it starts disarmed in
   both apps, keep it that way.
2. Confirm LOCK mode has actually locked onto the intended target (watch
   the overlay/console) before arming.
3. Keep the panic-stop control (`space` in `detect_and_track.py`) within
   immediate reach the entire time AUTO FIRE is armed.
4. Test the full chain — detect → lock → arm → centered → fires — with
   the weapon mechanism itself disarmed/unloaded first, if your hardware
   allows that kind of dry-run.
