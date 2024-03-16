import request from '@/utils/request'


//获取界面基础数据显示
export function getPageBaseData(params) {
    return request({
        url: `/mangeapi/basic/display`,
        method: 'get',
        params
    })
}

//获取热点停车
export function getPageParkList(params) {
    return request({
        url: `/mangeapi/basic/parklist`,
        method: 'get',
        params
    })
}

//获取重点监控
export function getMonitorList(params) {
    return request({
        url: `/mangeapi/basic/cameralist`,
        method: 'get',
        params
    })
}

//获取案件列表
export function getCaseList(params) {
    return request({
        url: `/mangeapi/basic/caselist`,
        method: 'get',
        params
    })
}

//打开监控
export function openMonitors(data) {
    return request({
        baseURL:'https://10.244.77.60:443',
        url: `/avitx_gw100/v1/server/openStream`,
        method: 'put',
        data
    })
}

//关闭监控
export function closeMonitors(data) {
    return request({
        baseURL:'https://10.244.77.60:443',
        url: `/avitx_gw100/v1/server/closeStream`,
        method: 'put',
        data
    })
}


