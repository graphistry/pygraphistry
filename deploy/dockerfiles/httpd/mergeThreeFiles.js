#!/usr/bin/env node
fs=require('fs');
console.log(JSON.stringify(Object.assign({},JSON.parse(fs.readFileSync(process.argv[2])),JSON.parse(fs.readFileSync(process.argv[3])),JSON.parse(fs.readFileSync(process.argv[4])))).replace('{','{"VIZ_LISTEN_PORT": %(process_num)d,'))

