/**
 * 腾讯服务器： TC
 * 政务服务器： ZF
 */
window.ENV = "ZF"

window.CONFIG = {
    TC:{
        ApiBaseUrl: 'https://dptest.milkway.tech:2333',
        ImageBaseUrl: 'https://dptest.milkway.tech:2337/',
        DigitalTwinCloudUrl: 'ws://127.0.0.1:3000',
        LocalServiceUrl: 'https://dpdb.milkway.tech'
    },
    ZF:{
        ApiBaseUrl: 'https://dpcy.cmzw.gov.cn:8101',
        ImageBaseUrl: 'https://dpcy.cmzw.gov.cn:8101/',
        DigitalTwinCloudUrl: 'ws://127.0.0.1:3000',
        LocalServiceUrl: 'https://dpcy.cmzw.gov.cn:8443/dp'
    }
}
