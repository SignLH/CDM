import request from '@/utils/request'
//2、获取案件汇总
export function getTotalCount(params) {
    return request({
        url: `/mangeapi/basic/totalevent`,
        method: 'get',
        params
    })
}
//3、获取 问题云图
export function getQueCloud(params) {
    return request({
        url: `/mangeapi/basic/words`,
        method: 'get',
        params
    })
}
//4、获取 趋势图
export function getTrendList(params) {
    return request({
        url: `/mangeapi/basic/trendlist`,
        method: 'get',
        params
    })
}

//5、获取 单兵设备
export function getDeviceList(data) {
    return request({
        url: `/mangeapi/basic/device`,
        method: 'post',
        data
    })
}

//7、获取 巡检记录
export function getInspectionList(params) {
    return request({
        url: `/mangeapi/basic/inspection`,
        method: 'get',
        params
    })
}

//8、获取 实地督查
export function getSuperviseInfo(params) {
    return request({
        url: `/mangeapi/basic/supervise`,
        method: 'get',
        params
    })
}

//9、获取 巡察案件
export function getInspectioncase(params) {
    return request({
        url: `/mangeapi/basic/inspectioncase`,
        method: 'get',
        params
    })
}