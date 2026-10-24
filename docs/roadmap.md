# Roadmap

Built just ahead of what the model repos need. Order follows the shot-models work.

1. **`pitch`**: provider conversions + distance / angle-to-goal helpers, with tests.
2. **`schema`**: canonical shot table (shared by xG, xGOT, goalkeeper models).
3. **`io.statsbomb`**: open data events + shot freeze frames.
4. **`io.whoscored`**: Postgres → parquet snapshot export with manifest; snapshot reader.
5. **`viz`**: shot map and goal-mouth plots.
6. Later, as other repos start: canonical action table (passing / possession value) and
   tracking loaders.
