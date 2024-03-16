import request from '@/utils/request'

//获取汽车模型id
export function getTrackCarId() {
  if (process.env.NODE_ENV === 'development') {
    return ''
  } else {
    return ''
  }
}

//获取镇级指挥网格列表
export function getMapGridList(params) {
  return request({
    url: `/mangeapi/basic/gis`,
    method: 'get',
    params
  })
}

//拉取gis 详细信息
export function getGisInfo(params) {
  return request({
    url: `/mangeapi/basic/gis/info`,
    method: 'get',
    params
  })
}

//获取党建信息包括人员
export function getPartyCommitteeInfo(params) {
  return request({
    url: `/mangeapi/basic/gpartycommittee`,
    method: 'get',
    params
  })
}

//获取智能设备
export function getIntelligentDevices(data) {
  return request({
    url: `/mangeapi/basic/device`,
    method: 'post',
    data
  })
}

//获取重点监控列表
export function getCameraDevices(params) {
  return request({
    url: `/mangeapi/basic/cameralist`,
    method: 'get',
    params
  })
}

//人员力量
export function getPeoplePower(params) {
  return request({
    url: `/mangeapi/basic/citypower`,
    method: 'get',
    params
  })
}

//获取清运车移动轨迹(包含车辆重量 单位： 吨)
export function getCarTrackList(params) {
  return request({
    url: `/mangeapi/basic/carlist`,
    method: 'get',
    params
  })
}

//获取巡检线路
export function getInspectionGis(params) {
  return request({
    url: `/mangeapi/basic/inspectiongis`,
    method: 'get',
    params
  })
}

//根据建筑id拉取楼栋和楼栋内房间信息
export function getBuildInfo(params) {
  return request({
    url: `/mangeapi/basic/buildinfo`,
    method: 'get',
    params
  })
}

//根据房间id拉取人员列表
export function getBuildPeopleInfo(data) {
  return request({
    url: `/mangeapi/basic/people`,
    method: 'post',
    data
  })
}

//根据gis_id 拉取gis地块，团队人员
export function getGridTeamInfo(params) {
  return request({
    url: `/mangeapi/basic/gisteam`,
    method: 'get',
    params
  })
}

//拉取天气信息（包含天气，降雨，天气预警）
export function getWeatherInfo(params) {
  return request({
    url: `/mangeapi/basic/getweather`,
    method: 'get',
    params
  })
}
//党建值日安排
export function getPartyduty(params) {
  return request({
    url: `/mangeapi/basic/partyduty`,
    method: 'get',
    params
  })
}
