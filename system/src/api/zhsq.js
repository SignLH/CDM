import request from '@/utils/request'

//获取 今日情况和案件汇总
export function getTotalEvent(params) {
    return request({
        url: `/mangeapi/basic/totalevent/count`,
        method: 'get',
        params
    })
}

//获取界面 智能设备统计
export function getTotalDevice(params) {
    return request({
        url: `/mangeapi/basic/totaldevice`,
        method: 'get',
        params
    })
}


//获取界面 人员力量
export function geCitypower(params) {
    return request({
        url: `/mangeapi/basic/citypower`,
        method: 'get',
        params
    })
}

//获取界面 人员力量2
export function geCitypower2(params) {
    return request({
        url: `/mangeapi/basic/gis`,
        method: 'get',
        params
    })
}