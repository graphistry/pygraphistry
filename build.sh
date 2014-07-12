#### Generates "naivebundle.js", "naivebundle.min.js"

EXTERNAL="-x jQuery -x node-webcl -x node-webgl --ignore fs"
REQUIRE="-r jQuery"
OUT="dist/naivebundle"
VENDORS="dist/vendors"
ENTRY="js/main.js"
OUTPACKAGE="Grapher"



browserify $EXTERNAL -d --standalone $OUTPACKAGE $ENTRY | tee $OUT.js | uglifyjs > $OUT.min.js
browserify $REQUIRE | tee $VENDORS.js | uglifyjs > $VENDORS.min.js
