#!/bin/sh

# Make sure we can run it again if a backup exists
export FILTER_BRANCH_SQUELCH_WARNING=1

git filter-branch -f --env-filter '
CORRECT_NAME="Lorenz Heiler"
CORRECT_EMAIL="lheiler@me.com"

# This targets ANY email containing "ic.ac.uk" OR the github "noreply" address
if echo "$GIT_COMMITTER_EMAIL" | grep -qE "ic.ac.uk|noreply"; then
    export GIT_COMMITTER_NAME="$CORRECT_NAME"
    export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
fi
if echo "$GIT_AUTHOR_EMAIL" | grep -qE "ic.ac.uk|noreply"; then
    export GIT_AUTHOR_NAME="$CORRECT_NAME"
    export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
fi
' --tag-name-filter cat -- --branches --tags
