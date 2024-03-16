import request from '@/utils/request'


//拉取党建活动列表
export function getPartyActivity(params) {
  return request({
    url: `/mangeapi/basic/activity`,
    method: 'get',
    params
  })
}

//党建分析 每个年龄段数据 趋势汇总
export function getPartyAnalyze(params) {
  return request({
    url: `/mangeapi/basic/partyanaly`,
    method: 'get',
    params
  })
}

//党建信息
export function getPartyInfoTotal(params) {
  return request({
    url: `/mangeapi/basic/party/total`,
    method: 'get',
    params
  })
}
