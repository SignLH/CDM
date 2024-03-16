import request from '@/utils/request'

//获取界面 网格
export function getGridData(params) {
    return request({
        url: `/mangeapi/basic/gis`,
        method: 'get',
        params
    })
}

//获取界面 指挥体系 人员列表
export function getPersonData(params) {
    return request({
        url: `/mangeapi/basic/duty`,
        method: 'get',
        params
    })
}


//获取界面 过程监管
export function getGcjgData(params) {
    return request({
        url: `/mangeapi/basic/intelligentdiscovery`,
        method: 'get',
        params
    })
}


//获取 园区税收
export function getTaxrevenueList(params) {
    return request({
        url: `/mangeapi/basic/taxrevenue`,
        method: 'get',
        params
    })
}

//获取 企业迁出迁入
export function getCompanyinout(params) {
    return request({
        url: `/mangeapi/basic/companyinout`,
        method: 'get',
        params
    })
}

//获取 获取党建信息
export function getPartyAllInfo(params) {
    return request({
        url: `/mangeapi/basic/gpartycommittee`,
        method: 'get',
        params
    })
}

//获取 独居老人信息
export function getSolitaryElderly(params) {
    return request({
        url: `/mangeapi/basic/oldpeopleinfo`,
        method: 'get',
        params
    })
}