var path = require('path');
var webpack = require('webpack');
var vizAppPackage = require('../package.json');
var AssetsPlugin = require('assets-webpack-plugin');
var graphistryConfig = require('@graphistry/config')();
var WebpackDashboard = require('webpack-dashboard/plugin');
var NPMInstallPlugin = require('npm-install-webpack-plugin');
var WebpackVisualizer = require('webpack-visualizer-plugin');
var ExtractTextPlugin = require('extract-text-webpack-plugin');
var WebpackNodeExternals = require('webpack-node-externals');
var StringReplacePlugin = require('string-replace-webpack-plugin');
var FaviconsWebpackPlugin = require('favicons-webpack-plugin');
var ClosureCompilerPlugin = require('webpack-closure-compiler');
var child_process = require('child_process');

var argv = process.argv.slice(2);
while (argv.length < 2) {
    argv.push(0);
}

module.exports = [
    clientConfig,
    serverConfig,
    apiConfig,
];

const commitId = child_process.execSync('git rev-parse --short HEAD').toString().trim();
const revName = child_process.execSync('git name-rev --name-only HEAD').toString().trim();
const buildNumber = process.env.BUILD_NUMBER;
const buildDate = Date.now();

const versionDefines = {
    __RELEASE__: JSON.stringify(graphistryConfig.RELEASE),
    __GITCOMMIT__: `"${commitId}"`,
    __GITBRANCH__: `"${revName}"`,
    __BUILDDATE__: `${buildDate}`,
    __BUILDNUMBER__: buildNumber ? `"${buildNumber}"` : undefined,
}

function commonConfig(
    isDevBuild = process.env.NODE_ENV === 'development',
    isFancyBuild = argv[1] === '--fancy'
) {
    return {
        postcss,
        amd: false,
        profile: isDevBuild,
        // Create Sourcemaps for the bundle
        devtool: 'source-map',
        // devtool: isDevBuild ? 'source-map' : 'hidden-source-map',
        resolve: {
            unsafeCache: true,
            alias: {
                'viz-client': path.resolve('./src/viz-client'),
                'viz-shared': path.resolve('./src/viz-shared'),
                'doc-worker': path.resolve('./src/doc-worker'),
                'etl-worker': path.resolve('./src/etl-worker'),
                'viz-worker': path.resolve('./src/viz-worker'),
                '@graphistry/falcor': path.resolve(isDevBuild ?
                    './node_modules/@graphistry/falcor/dist/falcor.all.js' :
                    './node_modules/@graphistry/falcor/dist/falcor.all.min.js'
                )
            }
        },
        module: {
            // preLoaders: [{
            //     test: /\.jsx?$/,
            //     exclude: /src\//,
            //     loader: 'source-map'
            // }],
            loaders: loaders(isDevBuild, isFancyBuild),
            noParse: [
                /\@graphistry\/falcor-query-syntax\/lib\/paths\-parser\.js$/,
                /\@graphistry\/falcor-query-syntax\/lib\/route\-parser\.js$/
            ]
        },
        plugins: plugins(isDevBuild, isFancyBuild),
        stats: {
            // Nice colored output
            colors: true
        }
    };
}

function clientConfig(
    isDevBuild = process.env.NODE_ENV === 'development',
    isFancyBuild = argv[1] === '--fancy'
) {

    var config = commonConfig(isDevBuild, isFancyBuild);

    config.node = { fs: 'empty', global: false };

    config.target = 'web';

    config.entry = {
        client: './src/viz-client/index.js',
        vendor: [
            'react-ace',
            'simpleflakes',
            'socket.io-client',
            'lodash', 'underscore',
            'rxjs', 'rxjs-gestures',
            // 'falcor-http-datasource',
            'moment', 'moment-timezone',
            'debug', 'pegjs', 'pegjs-util',
            'rc-switch', 'rc-color-picker',
            'redux', 'recompose', 'redux-observable',
            'react', 'react-dom', 'react-redux',
            'react-bootstrap', 'react-overlays',
            '@graphistry/falcor',
            '@graphistry/falcor-json-graph',
            '@graphistry/falcor-path-syntax',
            '@graphistry/falcor-path-utils',
            '@graphistry/falcor-query-syntax',
            '@graphistry/falcor-react-redux',
            '@graphistry/falcor-router',
            '@graphistry/falcor-socket-datasource',
            '@graphistry/rc-slider',
        ]
    };

    config.output = {
        path: path.resolve('./www'),
        pathinfo: isDevBuild,
        publicPath: '',
        filename: 'viz-client.js'
    };

    config.module.loaders = [
        ...config.module.loaders,
        {
            test: /\.css$/,
            loader: isDevBuild ? 'style!css!postcss' : ExtractTextPlugin.extract({
                loader: 'css!postcss'
            })
        },
        {
            test: /\.less$/,
            loader: isDevBuild ?
                'style!css?module&-minimize&localIdentName=[local]_[hash:6]!postcss!less' :
                ExtractTextPlugin.extract({
                    loader: 'css?module&minimize&localIdentName=[local]_[hash:6]!postcss!less'
                })
        }
    ];

    config.resolve.alias['@graphistry/common'] = path.resolve('./src/viz-client/client-logger');
    config.resolve.alias['dtrace-provider'] = path.resolve('./src/viz-client/client-logger/empty-shim.js');
    config.resolve.alias['fs'] = path.resolve('./src/viz-client/client-logger/empty-shim.js');
    config.resolve.alias['safe-json-stringify'] = path.resolve('./src/viz-client/client-logger/empty-shim.js');
    config.resolve.alias['mv'] = path.resolve('./src/viz-client/client-logger/empty-shim.js');

    config.plugins = [
        ...config.plugins,
        new webpack.optimize.CommonsChunkPlugin({
            name: 'vendor',
            minChunks: Infinity,
            filename: 'vendor.bundle.js'
        }),
        new AssetsPlugin({ path: path.resolve('./www') }),
        new webpack.DefinePlugin(
            Object.assign(
                {},
                {
                    global: 'window',
                    DEBUG: isDevBuild,
                    __DEV__: isDevBuild,
                    __CLIENT__: true,
                    __SERVER__: false,
                    __VERSION__: JSON.stringify(vizAppPackage.version),
                    __RELEASE__: JSON.stringify(graphistryConfig.RELEASE),
                    'process.env.NODE_ENV': '"production"'// || JSON.stringify(process.env.NODE_ENV),
                },
                versionDefines
            )
        ),
        new WebpackVisualizer({
            filename: `${config.output.filename}.stats.html`
        })
    ];

    if (!isDevBuild) {
        config.plugins = [
            ...config.plugins,
            new ClosureCompilerPlugin({
                compiler: {
                    language_in: 'ECMASCRIPT5',
                    language_out: 'ECMASCRIPT5',
                    compilation_level: 'SIMPLE',
                    rewrite_polyfills: false,
                    use_types_for_optimization: false,
                    warning_level: 'QUIET',
                    jscomp_off: '*',
                    jscomp_warning: '*',
                    source_map_format: 'V3',
                    create_source_map: `${config.output.path}/${
                                          config.output.filename}.map`,
                    output_wrapper: `%output%\n//# sourceMappingURL=${config.output.filename}.map`
                },
                concurrency: 3,
            })
        ];
    }

    return config;
}

function serverConfig(
    isDevBuild = process.env.NODE_ENV === 'development',
    isFancyBuild = argv[1] === '--fancy'
) {

    var config = commonConfig(isDevBuild, isFancyBuild);

    config.node = {
        console: true,
        __filename: true,
        __dirname: true
    };

    config.target = 'node';

    config.entry = { server: './src/viz-server/index.js' };

    config.output = {
        path: path.resolve('./www'),
        pathinfo: isDevBuild,
        filename: 'viz-server.js',
        libraryTarget: 'commonjs2'
    };

    config.externals = [
        // native modules will be excluded, e.g require('react/server')
        WebpackNodeExternals(),
        // these assets produced by assets-webpack-plugin
        /^.+assets\.json$/i,
    ];

    config.module.loaders = [
        ...config.module.loaders,
        {
            test: /\.less$/,
            loader: `css/locals?module&localIdentName=[local]_[hash:6]!postcss!less`
        }
    ];

    config.plugins = [
        ...config.plugins,
        new FaviconsWebpackPlugin({
            logo: './src/viz-server/static/img/logo_g.png',
            emitStats: true, statsFilename: 'favicon-assets.json'
        }),
        new webpack.BannerPlugin({
            raw: true,
            entryOnly: true,
            banner: `require('source-map-support').install({ environment: 'node' });`
        }),
        new webpack.DefinePlugin(
            Object.assign(
                {},
                {
                    window: 'global',
                    DEBUG: isDevBuild,
                    __DEV__: isDevBuild,
                    __CLIENT__: false,
                    __SERVER__: true,
                    'process.env.NODE_ENV': '"production"',
                },
                versionDefines
            )
        ),
        new WebpackVisualizer({
            filename: `${config.output.filename}.stats.html`
        }),
        new webpack.optimize.UglifyJsPlugin(serverUglifyJSConfig())
    ];

    return config;
}

function apiConfig(
    isDevBuild = process.env.NODE_ENV === 'development',
    isFancyBuild = argv[1] === '--fancy'
) {
    var config = commonConfig(isDevBuild, isFancyBuild);

    config.entry = { api: './src/api-client/index.js' };
    config.output = {
        publicPath: '',
        pathinfo: isDevBuild,
        path: path.resolve('./www'),
        libraryTarget: 'umd',
        umdNamedDefine: true,
        library: 'GraphistryJS',
        filename: 'graphistryJS.js'
    };

    config.plugins = [
        ...config.plugins,
        new webpack.DefinePlugin({
            global: 'window',
            DEBUG: isDevBuild,
            __DEV__: isDevBuild,
            __CLIENT__: false,
            __SERVER__: false,
            __VERSION__: JSON.stringify(vizAppPackage.version),
            __RELEASE__: JSON.stringify(graphistryConfig.RELEASE),
            'process.env.NODE_ENV': '"production"'//JSON.stringify(process.env.NODE_ENV)
        }),
        new WebpackVisualizer({
            filename: `${config.output.filename}.stats.html`
        })
    ];

    if (!isDevBuild) {
        config.plugins = [
            ...config.plugins,
            new ClosureCompilerPlugin({
                compiler: {
                    language_in: 'ECMASCRIPT5',
                    language_out: 'ECMASCRIPT5',
                    compilation_level: 'SIMPLE',
                    rewrite_polyfills: false,
                    use_types_for_optimization: false,
                    warning_level: 'QUIET',
                    jscomp_off: '*',
                    jscomp_warning: '*',
                    source_map_format: 'V3',
                    create_source_map: `${config.output.path}/${
                                          config.output.filename}.map`,
                    output_wrapper: `%output%\n//# sourceMappingURL=${config.output.filename}.map`
                },
                concurrency: 3,
            })
        ];
    }

    return config;
}

function loaders(isDevBuild) {
    return [
        babel(isDevBuild),
        { test: /\.json$/, loader: 'json' },
        { test: /\.glsl$/, loader: 'webpack-glsl' },
        { test: /\.proto$/, loader: 'proto-loader' },
        { test: /\.pegjs$/, loader: 'pegjs-loader?cache=true&optimize=size' },
        { test: /\.(hbs|handlebars)$/, loader: 'handlebars-loader' },
        { test: /\.eot(\?v=\d+\.\d+\.\d+)?$/, loader: "url?&name=[name]_[hash:6].[ext]" },
        { test: /\.svg(\?v=\d+\.\d+\.\d+)?$/, loader: "url?&name=[name]_[hash:6].[ext]&limit=10000&mimetype=image/svg+xml" },
        { test: /\.woff(\?v=\d+\.\d+\.\d+)?$/, loader: "url?&name=[name]_[hash:6].[ext]&limit=10000&mimetype=application/font-woff" },
        { test: /\.woff2(\?v=\d+\.\d+\.\d+)?$/, loader: "url?&name=[name]_[hash:6].[ext]&limit=10000&mimetype=application/font-woff" },
        { test: /\.ttf(\?v=\d+\.\d+\.\d+)?$/, loader: "url?&name=[name]_[hash:6].[ext]&limit=10000&mimetype=application/octet-stream" },
        // match everything except [
        //   hb, js, jsx, json, css, scss, less,
        //   html, glsl, pegjs, proto, handlebars
        // ] You can add more.
        { test: /\.(?!(hb|jsx?|json|s?css|less|html?|glsl|woff|woff2|ttf|eot|svg|pegjs|proto|handlebars)$)([^.]+$)/, loader: 'url?limit=10000&name=[name]_[hash:6].[ext]' },
        { test: /PEGUtil.js$/,
            include: /node_modules\/pegjs-util/,
            loader: StringReplacePlugin.replace({ // from the 'string-replace-webpack-plugin'
                replacements: [{
                    pattern: /typeof define\.amd !== (\"|\')undefined(\"|\')/ig,
                    replacement: function(/*match, p1, offset, string*/) {
                        return false;
                    }
                }]
            })
        }
    ];
    function babel(isDevBuild) {
        return {
            test: /\.(js|es6|mjs|jsx)$/,
            exclude: /(node_modules(?!\/rxjs))/,
            loader: 'babel-loader',
            query: {
                babelrc: false,
                cacheDirectory: true, // cache into OS temp folder by default
                plugins: ['transform-runtime'],
                presets: [
                    // !isDevBuild ? 'es2016' :
                    ['es2015', { modules: false, loose: true }],
                    'react',
                    'stage-0'
                ]
            }
        };
    }
}

function plugins(isDevBuild, isFancyBuild) {

    var plugins = [
        new StringReplacePlugin(),
        // new webpack.NamedModulesPlugin(),
        // Avoid publishing files when compilation fails
        new webpack.NoErrorsPlugin(),
        new webpack.ProvidePlugin({ React: 'react' }),
        new webpack.LoaderOptionsPlugin({
            // debug: false,
            // minimize: true,
            debug: isDevBuild,
            minimize: !isDevBuild,
            quiet: isDevBuild,
            progress: !isDevBuild
        }),
        // use this for universal server client rendering
        new ExtractTextPlugin({ allChunks: true, filename: 'styles.css' }),
    ];

    if (isDevBuild) {
        // plugins.push(new NPMInstallPlugin());
        plugins.push(new webpack.HotModuleReplacementPlugin());
        if (isFancyBuild) {
            plugins.push(new WebpackDashboard());
        } else {
            // Report progress for non-fancy dev builds
            plugins.push(new webpack.ProgressPlugin());
        }
    } else {
        // Report progress for prod builds
        plugins.push(new webpack.ProgressPlugin());
        plugins.push(new webpack.optimize.OccurrenceOrderPlugin(true));
        // Webpack deduping is currently broken :(
        // plugins.push(new webpack.optimize.DedupePlugin());
        plugins.push(new webpack.optimize.AggressiveMergingPlugin());
    }

    return plugins;
}

function postcss(webpack) {
    return [
        require('postcss-font-awesome'),
        require('autoprefixer')
    ];
}

function serverUglifyJSConfig() {
    return {
        mangle: false,
        beautify: true,
        sourceMap: true,
        compress: {
            /* (default: true) -- join consecutive simple statements using the comma operator. May be set to a positive integer to specify the maximum number of consecutive comma sequences that will be generated. If this option is set to true then the default sequences limit is 200. Set option to false or 0 to disable. The smallest sequences length is 2. A sequences value of 1 is grandfathered to be equivalent to true and as such means 200. On rare occasions the default sequences limit leads to very slow compress times in which case a value of 20 or less is recommended. */
            sequences: false,
            /* -- rewrite property access using the dot notation, for example foo["bar"] → foo.bar */
            properties: false,
            /* -- remove unreachable code */
            dead_code: true,
            /* -- remove debugger; statements */
            drop_debugger: false,
            /* (default: false) -- apply "unsafe" transformations (discussion below) */
            unsafe: false,
            /* (default: false) -- Reverse < and <= to > and >= to allow improved compression. This might be unsafe when an at least one of two operands is an object with computed values due the use of methods like get, or valueOf. This could cause change in execution order after operands in the comparison are switching. Compression only works if both comparisons and unsafe_comps are both set to true. */
            unsafe_comps: false,
            /* -- apply optimizations for if-s and conditional expressions */
            conditionals: false,
            /* -- apply certain optimizations to binary nodes, for example: !(a <= b) → a > b (only when unsafe_comps), attempts to negate binary nodes, e.g. a = !b && !c && !d && !e → a=!(b||c||d||e) etc. */
            comparisons: false,
            /* -- attempt to evaluate constant expressions */
            evaluate: false,
            /* -- various optimizations for boolean context, for example !!a ? b : c → a ? b : c */
            booleans: false,
            /* -- optimizations for do, while and for loops when we can statically determine the condition */
            loops: false,
            /* -- drop unreferenced functions and variables */
            unused: true,
            /* -- hoist function declarations */
            hoist_funs: false,
            /* (default: false) -- hoist var declarations (this is false by default because it seems to increase the size of the output in general) */
            hoist_vars: false,
            /* -- optimizations for if/return and if/continue */
            if_return: false,
            /* -- join consecutive var statements */
            join_vars: false,
            /* -- small optimization for sequences, transform x, x into x and x = something(), x into x = something() */
            cascade: false,
            /* -- default false. Collapse single-use var and const definitions when possible. */
            collapse_vars: false,
            /* -- default false. Improve optimization on variables assigned with and used as constant values. */
            // reduce_vars: false,
            /* -- display warnings when dropping unreachable code or unused declarations etc. */
            warnings: false,
            /* -- negate "Immediately-Called Function Expressions" where the return value is discarded, to avoid the parens that the code generator would insert. */
            negate_iife: false,
            /* -- the default is false. If you pass true for this, UglifyJS will assume that object property access (e.g. foo.bar or foo["bar"]) doesn't have any side effects. */
            pure_getters: false,
            /* -- default null. You can pass an array of names and UglifyJS will assume that those functions do not produce side effects. DANGER: will not check if the name is redefined in scope. An example case here, for instance var q = Math.floor(a/b). If variable q is not used elsewhere, UglifyJS will drop it, but will still keep the Math.floor(a/b), not knowing what it does. You can pass pure_funcs: [ 'Math.floor' ] to let it know that this function won't produce any side effect, in which case the whole statement would get discarded. The current implementation adds some overhead (compression will be slower). */
            pure_funcs: null,
            /* -- default false. Pass true to discard calls to console.* functions. */
            drop_console: false,
            /* -- default true. Prevents the compressor from discarding unused function arguments. You need this for code which relies on Function.length. */
            keep_fargs: true,
            /* -- default false. Pass true to prevent the compressor from discarding function names. Useful for code relying on Function.prototype.name. See also: the keep_fnames mangle option. */
            keep_fnames: true,
            /* -- default 1. Number of times to run compress. Use an integer argument larger than 1 to further reduce code size in some cases. Note: raising the number of passes will increase uglify compress time. */
            passes: 1,
        }
    };
}
