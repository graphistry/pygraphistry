module.exports = function override(webpackConfig) {
  webpackConfig.module.rules.push({
    test: /\.m?js$/,
    resolve: {
      fullySpecified: false,
    },
  });

  return webpackConfig;
};
