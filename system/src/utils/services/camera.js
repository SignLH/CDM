//聚焦至实体
import { mapService } from "@/utils/app-base";
import LOG from "@/utils/self-log";


/**
 * 调用预设漫游轨
 * @param client
 * @param trackName 轨1，轨2，轨3
 * @param callback
 * @returns {Promise<void>}
 */
export async function playCameraTrack(client, trackName, callback) {
  // 先获得绑定的 camera 对象
  const camera = await client.getCamera();

  // 再获得想要播放漫游轨 cameraTrack
  const track = await camera.getTrack(trackName);  // 通过指定的漫游轨名称或索引获得对应的 cameraTrack 对象

  track.loop = false;                           // 是否循环播放，默认为 false
  track.applyPresetCamera = true;               // 是否应用预设的镜头位置，默认为 true
  track.applyPresetEnvironment = true;          // 是否应用预设的天空后期环境效果，默认为 true
  track.applyPresetTransitionAnimation = true;  // 是否应用过渡动画效果，默认为 true

  await track.play();    // 播放当前漫游轨


  // 方式二，设置处理器
  track.onfinish = (event) => {
    // LOG.info(['事件触发', event);
    // LOG.info(['漫游轨索引号', event.detail.index);
    track.onfinish = null;      // 取消侦听
    LOG.info(['漫游轨名称', event.detail.name]);
    if (callback)
      callback()
  };

}

/**
 * 改变地图场景
 * @param sceneName
 * @param camera
 * @returns {Promise<void>}
 */
export async function changeMapScene(sceneName = '', camera = true) {
  const options = {
    camera,        // 可选，默认为 true，是否应用预设的镜头位置
    environment: true,  // 可选，默认为 true，是否应用预设的天空后期环境效果
    visibility: true,    // 可选，默认为 true，是否应用物体可见性状态，注意仅对快照场景中确实存在预设数据的情况有效
  }
  try {
    // 通过名称应用场景，参数为 DIVA 编辑器中场景的名称，默认名称为空，则无法利用该接口，需对场景右键进行重命名。
    await mapService.client.applyScene(sceneName, options);
  } catch (e) {
    LOG.warn(['应用场景 failed', e])
  }
}


/**
 * 获取镜头参数
 * @param client
 * @returns {Promise<*>}
 */
export async function getCameraInfo(client) {
  /*
   {
    pitch: 30.0,                                  // 镜头俯仰角
    yaw: 10.5,                                    // 镜头偏航角
    position: [1.0,1.5,1.0],                      // 镜头位置，[x, y, z]，单位：cm
    //coordinate: [116.445,39.918,0],             // 镜头位置，[经度,纬度,高度(单位: m)]，仅CIM模式下支持，指定 coordinate 时可不指定 position
    duration: 2                                   // 可选，单位（秒），默认为 2；如设置为 0，则为瞬移
}
   */
  try{
    const res = await client.request('GetCamera')
    return res.result
  }catch (e) {
    LOG.warn([e,'getCameraInfo error'])
    return {}
  }
}

/**
 * 设置镜头参数
 * @param client
 * @param data
 * @returns {Promise<*>}
 */
export async function setCameraStatus(client, data) {
  /*
   {
    pitch: 30.0,                                  // 镜头俯仰角
    yaw: 10.5,                                    // 镜头偏航角
    position: [1.0,1.5,1.0],                      // 镜头位置，[x, y, z]，单位：cm
    //coordinate: [116.445,39.918,0],             // 镜头位置，[经度,纬度,高度(单位: m)]，仅CIM模式下支持，指定 coordinate 时可不指定 position
    duration: 2                                   // 可选，单位（秒），默认为 2；如设置为 0，则为瞬移
}
   */
  try {
    await client.request('SetCamera', data)
    return true
  } catch (e) {
    LOG.warn([e])
    return false
  }
}
export async function cameraFocusCoordinate(client,longitude,latitude ,distance=800, degree =  -30.0,yaw=0) {
  /*
   {
    pitch: 30.0,                                  // 镜头俯仰角
    yaw: 10.5,                                    // 镜头偏航角
    position: [1.0,1.5,1.0],                      // 镜头位置，[x, y, z]，单位：cm
    //coordinate: [116.445,39.918,0],             // 镜头位置，[经度,纬度,高度(单位: m)]，仅CIM模式下支持，指定 coordinate 时可不指定 position
    duration: 2                                   // 可选，单位（秒），默认为 2；如设置为 0，则为瞬移
}
   */
  const coordinate=[longitude-0.008*(distance/500),latitude*1,distance]
  await setCameraStatus(mapService.client, {
    pitch: degree,                                  // 镜头俯仰角
    yaw,                                    // 镜头偏航角
    coordinate,             // 镜头位置，[经度,纬度,高度(单位: m)]，仅CIM模式下支持，指定 coordinate 时可不指定 position
    duration: 2                                   // 可选，单位（秒），默认为 2；如设置为 0，则为瞬移
  })
}

/**
 * 多实体聚焦算法
 * @param client
 * @param boundingBoxList
 * @param isPrevent 是否阻止 2023年5月22日11:25:52后  默认阻止
 * @returns {Promise<*>}
 */
export async function overallFocusFun(client, boundingBoxList = [],isPrevent=true) {
  if (boundingBoxList.length === 0 || isPrevent) {
    LOG.info(['多实体聚焦 isPrevent',isPrevent])
    return
  }
  LOG.info('boundingBoxList', boundingBoxList);
  //format data
  for (let i = 0; i < boundingBoxList.length; i++) {
    const data = boundingBoxList[i]
    boundingBoxList[i] = [data.x, data.y, data.z]
  }
  let maxV = [...boundingBoxList[0]], minV = [...boundingBoxList[0]]
  boundingBoxList.forEach(e => {
    for (let i = 0; i < e.length; i++) {
      if (maxV[i] < e[i]) {
        maxV[i] = e[i]
      }
      if (minV[i] > e[i]) {
        minV[i] = e[i]
      }
    }
  })
  const centerList = []
  let distance = 0
  for (let i = 0; i < 3; i++) {
    centerList.push((minV[i] + maxV[i]) * 1.0 / 2)
    const dist = maxV[i] - minV[i]
    if (distance < dist) distance = dist
  }
  // LOG.info([boundingBoxList)
  // LOG.info([maxV,minV,centerList)
  distance = distance * 1.5
  try {
    const res = await client.request('FocusOnCoord', {
      position: centerList,                // 聚焦目标位置，[x, y, z]，单位：cm
      //coordinate: [116.445,39.918,0],       // 聚焦目标位置，[经度,纬度,高度(单位: m)]，仅CIM模式下支持，指定 coordinate 时可不指定 position
      distance: distance,                        // 镜头到聚焦目标对象 boundingBox 中心的距离，单位 cm
      pitch: -30.0                            // 镜头俯仰角，取值范围 [-89.9, 89.9]
    })
    return true
  } catch (e) {
    LOG.warn(e)
    return false
  }
}
