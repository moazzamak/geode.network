# E10 Production Recovery Runbook

## Objectives

- Recovery time objective: 1 second on the local rehearsal host.
- Recovery point objective: 0 accepted requests lost.
- Recovery target: the direct parent of the interrupted or failed canary.

## Bad Canary

1. Stop canary traffic and keep production outputs authoritative.
2. Read `promotion_journal.json` and verify `phase` is
   `promotion_in_progress`.
3. Activate `previous_bundle_id` through `LocalModelBundleStore.activate()`.
4. Reload every service replica and verify all report the restored bundle ID.
5. Confirm no `*.partial` artifact remains and retain the failed child bundle.

## Coordinator Loss

1. Start a replacement coordinator against the same bundle registry.
2. Call `ProductionPromotionCoordinator.recover()` before accepting traffic.
3. Reload replicas from `CURRENT` and compare the ID with
   `previous_bundle_id` from the journal.
4. Record recovery duration, pointer identity, and journal state in telemetry.

## Rehearsal Command

```powershell
& '.\.venv\Scripts\python.exe' -m experiments.e2e.run_e10_production_rehearsal
```
