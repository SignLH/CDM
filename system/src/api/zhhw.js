import request from '@/utils/request'

//获取 一周统计
export function getYztjData(params) {
    return request({
        url: `/mangeapi/basic/weektotal`,
        method: 'get',
        params
    })
}

//获取界面 获取车辆列表
export function getCarList(params) {
    return request({
        url: `/mangeapi/basic/carinfo`,
        method: 'get',
        params
    })
}


//获取界面 获取车辆称重（轨迹）
export function getCarWeight(params) {
    return request({
        url: `/mangeapi/basic/carlist`,
        method: 'get',
        params
    })
}