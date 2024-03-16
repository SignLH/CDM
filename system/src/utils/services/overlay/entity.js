
import {mapService} from "@/utils/app-base";
import LOG from "@/utils/self-log";
import {setCameraStatus} from "@/utils/services/camera";
import store from '@/store'


/**
 * 区分id
 * 作用 为了区分添加地图实体的类型 SuffixForOverlayId为其分组名 好处：
 * 1.创建实体需要穿id id组成为 SuffixForOverlayId对应类型+被创建数据真实id 监听事件下可以通过前缀分辨是什么类型数据 以及获取对应真实id
 * 2.创建时按照组名为 SuffixForOverlayId创建可以统一删除同组数据
 */
export const SuffixForOverlayId={
  //类型值长度 绝对不能改动
  length:8,
  //获取对应类型分组名称
  getGroupValueById(id){
    return '/MEntity/'+id.substring(0,SuffixForOverlayId.length)
  },
  poi:'type001@',
  polygon:'type002@',
  polyline:'type003@',
  emissive:'type004@',
  carTrack:'type005@',
  poiPeople:'type006@',
  poiParty:'type007@',
  poiDevice:'type008@',
  poiMonitorDevice:'type009@',
  inspectionRoute:'type010@',
  poiServiceStation:'type011@',
  //巡检点位
  poiXjdw:'type012@',
  //文明实践站站点
  poiCivilizationPractice:'type013@',
  //退伍军人服务站
  poiVeteransServiceStation:'type014@',
  //廉洁文化宣传阵地
  poiHonestCulturePropagandaPositions:'type015@',
  //网格名称poi
  poiGird:'type016@',
}

/**
 * 删除覆盖物
 * @param client
 * @param entityIdList
 * @returns {Promise<void>}
 */
export async function deleteEntityByIdList(client, entityIdList = []) {
  LOG.info(["delete entity id list=", entityIdList])
  for (let id of entityIdList) {
    try {
      const entity = await client.getEntityById(id);
      await entity.setClient(null);
    } catch (e) {
      LOG.warn([e])
    }
  }
}

/**
 * 删除覆盖物By组名
 * @param client
 * @param groupName
 * @param deleteGlobalData 删除其他全局变量
 * @returns {Promise<boolean>}
 */
export async function deleteEntityByGroup(client, groupName,deleteGlobalData=true) {
  try {
    // const group = await mapService.client.getModelGroupByGroupPath(groupName)
    // group.destroy()
    // clear window
    if (deleteGlobalData){
      store.state.map.haveAddedJson=null
    }

    const response = await client.request('GetEntityIds', {
      group: groupName                         // 资源列表中从根目录开始的路径
    })

    const groupIdList= response.result
    const orderList=[]
    //创建批量请求
    groupIdList.forEach(e=>{
      orderList.push({
        method: 'DestroyEntity',  // 必填，API 方法名
        params: {        // 可选，API 请求的参数对象
          id: e
        }
      })
    })
    await client.batchRequest(orderList);
    LOG.info(['成功删除 groupName= '+groupName+' '+orderList.length+'个实体'])
    return true
  } catch (e) {
    LOG.warn([e])
    return false
  }
}


export async function updateEntityByGroup(client, groupName,isVisible) {
  try {
    const response = await client.request('UpdateEntityStatusByGroup', {
      group:groupName,                         // 必填，输入项，资源列表中组的名称（从根目录开始的路径）
      visible: isVisible,                              // 可选，设置可见性
      locked: false                                // 可选，设置是否锁定
    })
    // const group = await client.getModelGroupByGroupPath(groupName);
    // await group.setVisible(isVisible);
    LOG.warn([response])
    return response
  } catch (e) {
    LOG.warn([e])
    return false
  }
}

/**
 * 聚焦覆盖物
 */
export async function focusEntityById(client, id,distance=500, degree =  -30.0,yaw=0) {
  try{
    const model = await client.getEntityById(id);
    // if (id.substring(SuffixForOverlayId.length)===SuffixForOverlayId.polyline){
    //   model.focus(distance, -Math.PI / 6)
    //   return true
    // }
    // {"pitch":-29.999958038330078,"yaw":-81.34032440185547,
    //   "coordinate":[121.53587433735262,31.67602410590838,99.60281814905606],"duration":2}
    const coordinate=[model.coordinate.longitude-0.008*(distance/500),model.coordinate.latitude,distance]
    return await setCameraStatus(client,{
      pitch:degree,                                  // 镜头俯仰角
      yaw,                                    // 镜头偏航角
      coordinate,             // 镜头位置，[经度,纬度,高度(单位: m)]，仅CIM模式下支持，指定 coordinate 时可不指定 position
      duration:2                                   // 可选，单位（秒），默认为 2；如设置为 0，则为瞬移
    })
  }catch (e) {
    LOG.warn(['聚焦目标失败',e])
    return false
  }
}

/**
 * 设置所有组内实体不可见 不可点击
 */
export async function focusEntityById1(client, id,distance=50000, degree = -Math.PI / 6) {
  let model=''
  try {
    model = await client.getEntityById(id);
  }catch (e) {
    console.error("聚焦覆盖物"+id+"失败")
    return
  }
  model.focus(distance,degree)
}

//高亮建筑
//https://diva.sheencity.cn/docs/index.html#/api/rendermode?id=%e5%8f%a0%e8%89%b2%e9%ab%98%e4%ba%ae%e6%a8%a1%e5%bc%8f
//https://diva.sheencity.cn/diva-sdk/docs/enums/diva_sdk.RenderMode.html 类型
export async function setEntityRenderMode(client,id,mode){
  try {
    const response = await client.request('SetEntityRenderMode', {
      id,  // 目标对象的 entityId
      mode:mode     // 叠色高亮模式
    })
  }catch (e) {
    LOG.warn([e])
  }
}
