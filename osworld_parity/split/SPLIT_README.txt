OSWorld no-leak parity split (source: test_all.json, 369 tasks OSWorld-Verified)
seed=20260728  eval_fraction=0.3  stratified by app  DISJOINT
EVAL (held-out parity metric): 110 tasks   hash=176c675e620a97dd
TRAIN (rollout generation only): 259 tasks  hash=2699831dfc2143bf

app                   eval  train  total
chrome                  14     32     46
gimp                     8     18     26
libreoffice_calc        14     33     47
libreoffice_impress     14     33     47
libreoffice_writer       7     16     23
multi_apps              30     71    101
os                       7     17     24
thunderbird              4     11     15
vlc                      5     12     17
vs_code                  7     16     23
TOTAL                  110    259    369

--- gdrive exclusion (2026-07-28) ---
3 eval tasks require Google Drive OAuth (client_secrets.json, absent here) -> structurally unscorable; OSWorld test_nogdrive.json excludes them too. Effective held-out metric denominator = 107 scorable tasks (see gdrive_unscorable.txt). aggregate.py counts only tasks with result.json, so baseline AND all treatment evals drop these 3 identically (apples-to-apples).
