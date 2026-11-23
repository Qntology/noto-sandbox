# SDS ScopeKey v2 Registry

Generated from `crates/noto-sds/src/schema.rs` (Phase 2 – Part 1 · D1 & D4).

## Syntax

- Key grammar: `<track>/<subject>;<facet>=<value>;...`
- Reserved characters in values (`/`, `;`, `=`, `%`) are percent-encoded.
- Facets appear in fixed canonical order listed below; empty values are omitted.

## Facets

| Facet | Syntax | Source | Cap | Description |
| --- | --- | --- | ---: | --- |
| `ph` | `[a-z0-9_.:+-]{1..48}` | `PhaseSpec.id / act phase` | 64 | N:N or act phase identifier |
| `set` | `dev | holdout` | `ScenarioSpec.holdout` | 4 | Development vs holdout evaluation split |
| `stg` | `[a-z0-9_]{1..24}` | `Progression stage vocabulary` | 24 | Progression stage (approach/look/grasp/lift/stow/reacquire/move/step/brake) |
| `form` | `<plan>:<base>:f<n>` | `bench::form_id` | 64 | Robot body form identifier |
| `body` | `[a-z0-9_@:+-]{1..48}` | `BodyPlan::id` | 64 | Body plan identifier (including rig edit hash) |
| `gnd` | `firm|rock|dirt|grass|gravel|sand|wet_sand|water` | `Ground::id` | 16 | Terrain ground surface |
| `slope` | `flat|step|up_lo|up_hi|down_lo|down_hi` | `World::slope_bucket` | 16 | Terrain slope or curb bucket |
| `act` | `[a-z0-9_]{1..24}` | `World action / posture tag` | 32 | Activity label (move/brake/mine/pick/deliver/drop/idle/hold) |
| `src` | `quiet|arm_raise|arm_lift|foot_lift|step_up|slope|carry_front|accel` | `Source::id` | 16 | Dominant balance excitation source |
| `mode` | `[a-z0-9_.:@+-]{1..48}` | `Recognizer / ML / TTM mode` | 32 | Perception, depth, locate, tracker, or predictor mode |
| `wk` | `lead | crew-<form>` | `Worker role` | 16 | Worker role identifier (lead or sub-worker role) |
| `pass` | `game | ml | holdout | eval` | `Runtime::pass` | 8 | Execution pass partition |

## Track Fallback Chains

| Track | Prefix | Allowed Facets | Fallback Chain |
| --- | --- | --- | --- |
| `balance` | `nn:` | `ph, set, stg, form, gnd, slope, wk` | `subject -> +ph -> +ph+form -> +ph+form+gnd+slope` |
| `balance` | `fx:` | `gnd, slope, act, form` | `subject -> +gnd -> +gnd+slope+act` |
| `balance` | `*` | `slope, act, form, gnd, stg, wk` | `subject -> +slope+act -> +form+slope+act` |
| `effort` | `*` | `src, slope, act, form, gnd, stg` | `subject -> +src -> +slope+act+src` |
| `vision` | `*` | `mode, stg` | `subject -> +mode -> +stg+mode` |
| `depth` | `*` | `mode, stg` | `subject -> +mode -> +stg+mode` |
| `locate` | `*` | `mode, stg` | `subject -> +mode -> +stg+mode` |
| `tracker` | `*` | `mode, stg` | `subject -> +mode -> +stg+mode` |
| `act` | `*` | `ph, stg, form, body, gnd, slope, act, wk` | `subject -> +form -> +gnd -> +ph -> +stg+form` |
| `forecast` | `*` | `mode` | `subject -> +mode` |
| `memory` | `*` | `mode, gnd` | `subject -> +mode -> +gnd+mode` |
| `quest` | `*` | `stg, pass` | `subject -> +pass -> +stg+pass` |
| `motion` | `*` | `mode, stg, form, gnd` | `subject -> +mode` |
