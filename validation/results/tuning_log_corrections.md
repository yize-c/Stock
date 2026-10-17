# Corrections to tuning_log.csv

`tuning_log.csv` is append-only; its rows are not edited. Errors found later are recorded here.

## Attempt 1 (started 2026-10-17T04:01:52+00:00)

- Logged `git_commit` = `5953bbb`, `git_dirty` = `False`. Wrong: the command read the commit
  when the run ended (05:47 UTC), after later commits. The code that ran is **`2ef3098`**
  (HEAD from 04:01:33 UTC, per `git reflog`); whether the working tree had uncommitted changes
  at 04:01:52 cannot be confirmed afterwards.
- Commits made during the run (`25d6acc` … `5953bbb`) should not affect it: the config is read
  once at the start, and Python loads each module once, on the first stock, before the first
  of those commits (04:05:30 UTC); this cannot be proven for every code path afterwards. The
  only config change among them that touches the data (`63b818e`, listing date of 4904)
  concerns a stock outside the 24-stock walk-forward sample.
- Fixed in the CLI: the commit is now read before the run starts
  (`tests/test_validation_tools.py::test_cli_dev_run_logs_the_commit_from_the_start`).
