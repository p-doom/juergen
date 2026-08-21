#!/usr/bin/env bash
# Print the immutable source a dispatched job must run from, or refuse.
#
# A job reads a detached worktree pinned at the sha labctl recorded at dispatch,
# keyed by that sha, created once and thereafter never written, so an edit to the
# live tree cannot reach a running job. Recipes resolve `provenance.repo_path`
# -- the live checkout -- because labctl's staged snapshot cannot resolve the
# sibling `desktop` path dependency.
#
# It refuses rather than falling back to the live path: a fallback converts an
# honest crash into a plausible wrong result.
#
# Keyed by sha, so concurrent jobs at one commit share one tree and distinct
# commits never collide. The trees are disposable: `git worktree prune` after
# `rm -rf` reclaims both the directory and the registration.
#
# Usage: dispatch_source.sh <live-repo> <40-char-sha> <pin-root>

set -uo pipefail

fail() { echo "FATAL dispatch_source: $*" >&2; exit 2; }

[ $# -eq 3 ] || fail "usage: dispatch_source.sh <live-repo> <sha> <pin-root>"
live="$1"
sha="$2"
root="$3"

git -C "$live" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "not a git checkout: $live"
# A short sha resolves differently as the repository grows; the recorded one is
# always full, so anything else is a caller bug rather than something to widen.
[ "${#sha}" -eq 40 ] || fail "not a full 40-character sha: $sha"
git -C "$live" cat-file -e "$sha^{commit}" 2>/dev/null || fail "not a commit in $live: $sha"

pin="$root/$sha"
mkdir -p "$root" || fail "cannot create pin root: $root"

# The first job at a sha creates the tree and the rest wait, rather than racing
# on `.git/worktrees`.
exec 9>"$root/.$sha.lock" || fail "cannot open lock in $root"
flock 9 || fail "cannot take lock in $root"
if [ ! -e "$pin" ]; then
  git -C "$live" worktree add --detach "$pin" "$sha" >/dev/null 2>&1 \
    || fail "cannot create pinned worktree for $sha under $root"
fi
exec 9<&-

# A pin is only a pin while nobody has written to it. Both halves are checked on
# every job, not only at creation.
[ "$(git -C "$pin" rev-parse HEAD 2>/dev/null)" = "$sha" ] || fail "pinned checkout is not at $sha: $pin"
[ -z "$(git -C "$pin" status --porcelain 2>/dev/null)" ] || fail "pinned checkout has been edited: $pin"

printf '%s\n' "$pin"
