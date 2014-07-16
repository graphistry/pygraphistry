'use strict';

module.exports = function(grunt) {
    grunt.initConfig({
        pkg: grunt.file.readJSON('package.json'),

        jshint: {
            main: ['src/**/*.js'],
            options: {
                jshintrc: '.jshintrc',
                force: grunt.option('dev')
            }
        },

        browserify: {
            main: {
                src: ['src/main.js'],
                dest: 'dist/<%= pkg.name %>.js',
                options: {
                    // external: []
                    bundleOptions: {
                        debug: true
                    },
                    watch: grunt.option('dev'),
                    keepAlive: grunt.option('dev')
                }
            }
        },

        clean: {
            main: ['dist']
        }
    });

    grunt.loadNpmTasks('grunt-browserify');
    grunt.loadNpmTasks('grunt-contrib-jshint');
    grunt.loadNpmTasks('grunt-contrib-clean');

    grunt.registerTask('default', ['jshint', 'browserify']);
}
