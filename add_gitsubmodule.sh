cd public
for d in */; do
  if [ -d "$d/.git" ]; then
    # Replace the URL below with the actual remote URL for each repo
    REPO_URL=$(cd "$d" && git config --get remote.origin.url)
    cd ..
    git submodule add -f "$REPO_URL" "public/${d%/}"
    cd public
  fi
done