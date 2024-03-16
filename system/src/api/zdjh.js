import request from '@/utils/request'

//获取安置帮教人员
export function getResettlePeople(params) {
  return request({
    url: `/mangeapi/basic/fetchResettlePeople`,
    method: 'get',
    params
  })
}

//获取重点管控人员
export function getKeyPeople(params) {
  return request({
    url: `/mangeapi/basic/fetchKeynotePeople`,
    method: 'get',
    params
  })
}

//获取独居老人
export function getALonePeople(params) {
  return request({
    url: `/mangeapi/basic/fetchOldPeople`,
    method: 'get',
    params
  })
}

//民族宗教
export function getReligiousData(params) {
  return request({
    url: `/mangeapi/basic/religious`,
    method: 'get',
    params
  })
}

//拉取教众列表
export function getReligiousListData(params) {
  return request({
    url: `/mangeapi/basic/religiouslist`,
    method: 'get',
    params
  })
}
