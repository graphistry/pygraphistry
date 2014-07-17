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
                src: ['src/main.js'],
                dest: 'dist/<%= pkg.name %>.js',
                options: {
                    bundleOptions: { debug: true },
                    transform: ['brfs'],
                    watch: false,
                    keepAlive: false,
                    didRun: false,
                    postBundleCB: function(err, src, next) {
                        global['browserifyDidRun'] = true;
                        next(err, src);
                    }
                }
            }
        },

        exorcise: {
            bundle: {
                files: {
                    'dist/<%= pkg.name %>.map': ['dist/<%= pkg.name %>.js'],
                }
            }
        },

        jsdoc : {
            dist : {
                src: ['src/*.js'],
                options: {
                    destination: 'doc'
                }
            }
        },

        watch: {
            all: {
                files: ['dist/<%= pkg.name %>.js'],
                tasks: ['jshint', 'maybeExorcise'],
                options: {
                    event: ['added', 'deleted', 'changed'],
                    spawn: false,
                    livereload: 35729
                },
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
    grunt.loadNpmTasks('grunt-jsdoc');

    grunt.registerTask('default', ['jshint', 'browserify', 'jsdoc', 'exorcise']);

    grunt.registerTask('live', 'Set a global variable.', function() {
        grunt.config.set("browserify.main.options.watch", true);
        grunt.task.run('default', "watch");
    });

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
