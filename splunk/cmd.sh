#quick helper for testing wrapper

node wrapper.js "`cat config.json | tr -d '\n' | sed 's#\"#\\\"#g'`"
