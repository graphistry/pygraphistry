'use strict';

module.exports = function(grunt) {
    grunt.initConfig({
        pkg: grunt.file.readJSON('package.json'),

        jshint: {
            main: ['src/**/*.js'],
            options: {
                jshintrc: '.jshintrc',
                force: true
            }
        },

        browserify: {
            main: {
                src: ['src/main.sc.js'],
                dest: 'dist/StreamGL.sc.js',
                options: {
                    bundleOptions: { debug: true },
                    transform: ['brfs'],
                    watch: true,
                    keepAlive: false,
                    external: ['render-config'],
                    postBundleCB: function(err, src, next) {
                        global['browserifyDidRun'] = true;
                        next(err, src);
                    },
                    preBundleCB: function(browserifyInstance) {
                        // On "update", limit jshint to checking only updated files
                        if(!global['browserifyDidSetWatchers']) {
                            global['browserifyDidSetWatchers'] = true;
                            browserifyInstance.on('update', function(files) {
                                grunt.config.set("jshint.main", files);
                            });
                        }
                    },
                    force: true
                }
            },

            SCRenderer: {
                src: ['src/renderer.config.sc.js'],
                dest: 'dist/render-config.sc.js',
                options: {
                    transform: ['brfs'],
                    watch: true,
                    keepAlive: false,
                    alias: ['src/renderer.config.sc.js:render-config'],
                    postBundleCB: function(err, src, next) {
                        next(err, src);
                    },
                    bundleOptions: {
                        debug: true,
                    },
                }
            }
        },

        exorcise: {
            bundle: {
                files: {
                    'dist/StreamGL.sc.map': ['dist/StreamGL.sc.js'],
                }
            }
        },

        watch: {
            all: {
                files: ['dist/StreamGL.sc.js'],
                tasks: ['jshint', 'maybeExorcise'],
                options: {
                    spawn: false
                }
            },

            livereload: {
                files: ['dist/*.js', 'index.html', './*.css'],
                options: {
                    livereload: 35729
                }
            },

            configFiles: {
                files: [ 'Gruntfile.js' ],
                options: {
                    reload: true
                }
            }
        },

        clean: {
            main: ['dist', 'doc']
        }
    });

    grunt.loadNpmTasks('grunt-browserify');
    grunt.loadNpmTasks('grunt-contrib-jshint');
    grunt.loadNpmTasks('grunt-contrib-clean');
    grunt.loadNpmTasks('grunt-contrib-watch');
    grunt.loadNpmTasks('grunt-exorcise');

    grunt.registerTask('default', ['jshint', 'browserify', 'exorcise']);
    grunt.registerTask('live', ['default', 'watch']);

    grunt.registerTask('maybeExorcise', 'Run Exorcise as long as browserify has run first', function() {
        if(global['browserifyDidRun']) {
            grunt.log.oklns("Running exorcise becuase browserify has run before");
            grunt.task.run('exorcise');

            global['browserifyDidRun'] = false;
        } else {
            grunt.log.errorlns("Not running exorcise becuase browserify did NOT run before");
        }
    });
}
