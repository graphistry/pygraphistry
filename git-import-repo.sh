repo="$1"
dir="$(echo "$2" | sed 's/\/$//')"
local_branch="$(git rev-parse --abbrev-ref HEAD)"
mkdir -p "$dir"

tmp="$(mktemp -d)"
remote="$(echo "$tmp" | sed 's/\///g'| sed 's/\./_/g')"

git clone "$repo" "$tmp"
pushd "$tmp"

git filter-branch --index-filter '
    git ls-files -s |
    sed "s,\t,&'"$dir"'/," |
    GIT_INDEX_FILE="$GIT_INDEX_FILE.new" git update-index --index-info &&
    mv "$GIT_INDEX_FILE.new" "$GIT_INDEX_FILE"
' HEAD

popd
git remote add -f "$remote" "file://$tmp/.git"
git fetch "$remote"
git merge --allow-unrelated-histories -m "Merge $repo into $local_branch" --no-edit "$remote/master"
git remote remove "$remote"
rm -rf "$tmp"
