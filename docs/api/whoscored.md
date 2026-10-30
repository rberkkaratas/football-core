# io.whoscored

The private WhoScored (Opta) CSV archive, loaded into the canonical `shots` table (with an empty
`freeze_frames`) and the canonical `matches` table. The archive is read in place and never
committed; the snapshot manifest records counts and a digest but no file list.

```bash
uv run fc snapshot whoscored-v1 --builder whoscored --input "<archive>/data"
uv run fc verify manifests/whoscored-v1.json
```

::: football_core.io.whoscored
