'use strict';

module.exports = function(grunt) {
    grunt.initConfig({
        pkg: grunt.file.readJSON('package.json'),

        jshint: {
            main: ['src/**/*.js'],
            options: {
                jshintrc: '../../../.jshintrc',
                force: true
            }
        },

        browserify: {
            SCMain: {
                src: ['src/main.sc.js'],
                dest: 'dist/StreamGL.sc.js',
                options: {
                    bundleOptions: { debug: true },
                    transform: ['brfs'],
                    watch: true,
                    keepAlive: false,
                    external: ['render-config'],
                    postBundleCB: function(err, src, next) {
                        global['browserifyDidRunSC'] = true;
                        next(err, src);
                    },
                    preBundleCB: function(browserifyInstance) {
                        // On "update", limit jshint to checking only updated files
                        if(!global['browserifyDidSetWatchersSC']) {
                            global['browserifyDidSetWatchersSC'] = true;
                            browserifyInstance.on('update', function(files) {
                                grunt.config.set("jshint.main", files);
                            });
                        }
                    },
                    force: true
                }
            },

            GraphMain: {
                src: ['src/main.graph.js'],
                dest: 'dist/StreamGL.graph.js',
                options: {
                    bundleOptions: { debug: true },
                    transform: ['brfs'],
                    watch: true,
                    keepAlive: false,
                    external: ['render-config'],
                    postBundleCB: function(err, src, next) {
                        global['browserifyDidRunGraph'] = true;
                        next(err, src);
                    },
                    preBundleCB: function(browserifyInstance) {
                        // On "update", limit jshint to checking only updated files
                        if(!global['browserifyDidSetWatchersGraph']) {
                            global['browserifyDidSetWatchersGraph'] = true;
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
            },

            GraphRenderer: {
                src: ['src/renderer.config.graph.js'],
                dest: 'dist/render-config.graph.js',
                options: {
                    transform: ['brfs'],
                    watch: true,
                    keepAlive: false,
                    alias: ['src/renderer.config.graph.js:render-config'],
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
            SC: {
                files: {
                    'dist/StreamGL.sc.map': ['dist/StreamGL.sc.js'],
                }
            },

            graph: {
                files: {
                    'dist/StreamGL.graph.map': ['dist/StreamGL.graph.js'],
                }
            }
        },

        watch: {
            SC: {
                files: ['dist/StreamGL.sc.js'],
                tasks: ['jshint', 'maybeExorciseSC'],
                options: {
                    spawn: false
                }
            },

            graph: {
                files: ['dist/StreamGL.graph.js'],
                tasks: ['jshint', 'maybeExorciseGraph'],
                options: {
                    spawn: false
                }
            },

            // livereload: {
            //     files: ['dist/*.js', 'index.html', './*.css'],
            //     options: {
            //         livereload: 35729
            //     }
            // },

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

    grunt.registerTask('maybeExorciseSC', 'Run Exorcise as long as browserify has run first', function() {
        if(global['browserifyDidRunSC']) {
            grunt.log.oklns("Running exorcise becuase browserify has run before");
            grunt.task.run('exorcise:SC');

            global['browserifyDidRunSC'] = false;
        } else {
            grunt.log.errorlns("Not running exorcise becuase browserify did NOT run before");
        }
    });

    // I am really not a fan of Grunt...
    grunt.registerTask('maybeExorciseGraph', 'Run Exorcise as long as browserify has run first', function() {
        if(global['browserifyDidRunGraph']) {
            grunt.log.oklns("Running exorcise becuase browserify has run before");
            grunt.task.run('exorcise:graph');

            global['browserifyDidRunGraph'] = false;
        } else {
            grunt.log.errorlns("Not running exorcise becuase browserify did NOT run before");
        }
    });
}
