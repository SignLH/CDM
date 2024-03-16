/**
 * 该配置统一打包路径，设置一遍即可
 * @type {string}
 */
const BASE = "";

module.exports = {

    appName: '东平镇综合智慧管理平台',

    //打包配置
    buildConfig: {
        base: `${BASE}/`,

        //打包时的基本配置，打包时改一下
        build: {
            publicPath: `./`,
            outputDir: `../BuildProd/DongpingDataScreen`,
            assetsDir: 'static',
        },
    },
}
