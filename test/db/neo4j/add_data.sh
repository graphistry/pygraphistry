#!/bin/bash
set -ex


NEO4J_PROTOCOL=${NEO4J_PROTOCOL:-http}
NEO4J_HOST=${NEO4J_HOST:-localhost}
#7474 -> 10003, 10006
NEO4J_PORT=${NEO4J_PORT:-7474}
NEO4J_USER=${NEO4J_USER:-neo4j}
NEO4J_PASS=${NEO4J_PASS:-test}

NEO4J_BASE="${NEO4J_PROTOCOL}://${NEO4J_HOST}:${NEO4J_PORT}"


curl -f \
    -X POST "${NEO4J_BASE}/db/data/transaction/commit" \
    --user "${NEO4J_USER}:${NEO4J_PASS}" \
    -H "Accept: application/json; charset=UTF-8" \
    -H "Content-Type: application/json" \
    -d '{ "statements" : [ { "statement" : "CREATE (a:A { name: \"AA\", title: \"tAA\", dt: datetime(\"2019-06-01T18:40:32.142+0100\"), x: 10, y: 20 }) RETURN a" } ] }'

curl -f \
    -X POST "${NEO4J_BASE}/db/data/transaction/commit" \
    --user "${NEO4J_USER}:${NEO4J_PASS}" \
    -H "Accept: application/json; charset=UTF-8" \
    -H "Content-Type: application/json" \
    -d '{ "statements" : [ { "statement" : "CREATE (a:B { name: \"BB\", title: \"tBB\", dt: datetime(\"2019-07-01T18:40:32.142+0100\"), x: 20, y: 30 }) RETURN a" } ] }'

curl -f \
    -X POST "${NEO4J_BASE}/db/data/transaction/commit" \
    --user "${NEO4J_USER}:${NEO4J_PASS}" \
    -H "Accept: application/json; charset=UTF-8" \
    -H "Content-Type: application/json" \
    -d '{ "statements" : [ { "statement" : "CREATE (a:C { name: \"CC\", title: \"tCC\", dt: datetime(\"2019-07-01T18:40:32.142+0100\"), x: 30, y: 40 }) RETURN a" } ] }'

    
curl -f \
    -X POST "${NEO4J_BASE}/db/data/transaction/commit" \
    --user "${NEO4J_USER}:${NEO4J_PASS}" \
    -H "Accept: application/json; charset=UTF-8" \
    -H "Content-Type: application/json" \
    --data-binary @- << EOF
{
  "statements" : [ {
    "statement" : "MATCH (a:A), (b:B), (c:C) WHERE a.name = \"AA\" AND b.name = \"BB\" AND c.name = \"CC\" CREATE p=(a)-[:A_TO_C]->(c)<-[:B_TO_C]-(b) RETURN p",
    "resultDataContents": [ "row", "graph" ]
  } ]
}
EOF

