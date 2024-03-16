import request from '@/utils/request'

//系统登录
export function login(data) {
    return request({
        url: `/authapi/login`,
        method: 'post',
        data
    })
}
//获取 驾驶舱导航信息
export function getSceneInfo() {
    return request({
        url: `/mangeapi/basic/scene`,
        method: 'get'
    })
}