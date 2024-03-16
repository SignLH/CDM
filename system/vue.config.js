'use strict'
const {
    buildConfig
} = require('./src/settings.js')
const buildBaseConfig = buildConfig.build
module.exports = {
    ...buildBaseConfig,
    lintOnSave: false,
    productionSourceMap: false,
    //https://blog.csdn.net/weixin_58099903/article/details/129881336
    parallel: false,
    devServer: {
        port: 9934,
        open: true,
        overlay: {
            warnings: false,
            errors: true
        },
        disableHostCheck: true,
        proxy: {
            ['/api']: {
                target: `http://localhost:8080`,
                changeOrigin: true,
                pathRewrite: {
                    ['^/api']: ''
                }
            }
        },
    },
    configureWebpack: {
        module: {
            rules: [{
                test: /\.mjs$/,
                type: "javascript/auto",
                resolve: {
                    mainFields: ["exports", "module", "browser", "main"],
                },
            }],
        },
    },
}
