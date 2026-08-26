# Bundled TLE seed data

This directory holds a minimal, last-known-good set of real TLEs per
CelesTrak group, checked into the repo so the application still shows
real satellites even on a machine with no network access at all, or
during a CelesTrak outage with an empty local cache (both have happened
repeatedly during this project's own development — CelesTrak has no
uptime SLA and does rate-limit/throttle at times).

`core.tle_manager.TLEManager.fetch_group()` only reads these files as a
last resort: live network fetch is always tried first, then the local
SQLite cache (any previously fetched data, even if stale), and only then
this bundled seed.

**Refreshing:** run `python scripts/update_seed_tle.py` whenever
CelesTrak is reachable, to overwrite these files with a current fetch.
The data here is real but will drift out of date over time — propagation
error from stale TLEs grows slowly, so this is fine as a fallback of
last resort, not as a primary data source.
