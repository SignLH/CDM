import {POI, POI2D, POIIcon} from '@sheencity/diva-sdk';
import {Vector3} from '@sheencity/diva-sdk-math';
import {Emissive} from '@sheencity/diva-sdk';
import "@/utils/services/overlay/overlay.model";
import Vue from 'vue'
import {deleteEntityByIdList, SuffixForOverlayId} from "@/utils/services/overlay/entity";

import html2canvas from 'html2canvas';
import LOG from "@/utils/self-log";


/**
 * 添加poi
 * @param client
 * @param id
 * @param title
 * @param resourceName
 * @param coordinate
 * @param icon
 * @param bgColor
 * @param isFocus
 * @param eventCallback
 * @returns {Promise<void>}
 */
export async function addPOI3DNew(client, id, title = 'poi test', resourceName = 'POI圆形标签',
                                  coordinate, icon, bgColor, isFocus, eventCallback) {
  // await deleteEntityByIdList(client, [id]).catch(e => Vue.prototype.LOG.warn(e))
  const response = await client.request('CreateEntity', {
    id, //对象标识符，必须全局唯一，建议使用 GUID
    type: "poi",               // 指定类型
    resourceName: resourceName,     // DIVA 资源库中资源的名称
    name: id,
    group: SuffixForOverlayId.getGroupValueById(id),
    visible: true,             // 可选，可见性，默认 true
    locked: false,             // 可选，锁定状态，默认 false
    coordinate,
    scale: [0.5, 0.5, 0.5],   // [scaleX,scaleY,scaleZ] 可选，默认[1.0,1.0,1.0]
    property: {                // 覆盖物类型的特有属性
      icon,        // 可选，图标类型；icon参数必有其一，如都未指定将创建失败
      title,      // 可选，标题文字，默认无标题（注：部分 POI 资源不支持标题文字）
      autoSize: true,       // 可选，默认关闭；开启时以屏幕运算，无论远近像素大小固定；关闭时以三维空间运算，近大远小
      backgroundColor: bgColor,      // 可选，[r,g,b]，整体颜色，取值范围[0,0,0]~[255,255,255]，默认 [0,0,0]
      opacity: 0.5           // 可选，不透明度，值域[0, 1.0]，默认 1.0
    }
  })

  // 侦听事件
  if (eventCallback) {
    client.request('AddEventListener', {
      entityId: id,    // 需要添加事件的 entityId
      type: "click"           // 事件类型：click 单击；dblclick 双击；mousedown 按下；mouseup 抬起
    }).then(res => {
      LOG.warn([res])
      const id = res.target
      eventCallback(id)
    })
  }

  // if (isFocus)
  //   await focusEntityById(client, id, 50000)
}

/**
 * 添加点位
 * @param client map instance
 * @param id id
 * @param title 名称
 * @param resourceName 样式名称
 * @param coordinate 坐标
 * @param icon 图标
 * @param bgColor 颜色
 * @param isFocus 废弃参数
 * @param eventCallback 点击事件回调
 * @return {Promise<*>}
 */
export async function addPOI(client, id, title = 'poi test', resourceName = 'POI圆形标签',
                             coordinate, icon, bgColor, isFocus, eventCallback) {
  // await deleteEntityByIdList(client, [id]).catch(e => Vue.prototype.LOG.warn(e))

  if (coordinate[0] === 0 || coordinate[1] === 0) {
    LOG.warn(['不正常的地理位置 coordinate=', coordinate])
    return
  }

  const poiJson = {
    id,
    name: id,
    group: SuffixForOverlayId.getGroupValueById(id),
    resource: {
      name: resourceName,
    },
    coordinate,
    scale: new Vector3(0.5, 0.5, 0.5),
    clickable: true,
    visible: true,
    icon,
    title,
    autoSize: true, //可选，默认关闭；开启时以屏幕运算，无论远近像素大小固定；关闭时以三维空间运算，近大远小
    backgroundColor: bgColor,
    opacity: 1,
  }
  const poi = new POI(poiJson)
  await poi.setClient(client).catch(e => {
    LOG.warn([e])
    LOG.warn(['添加失败 请检查一下颜色 地理位置 图片url',poiJson])
    LOG.warn(['添加失败 经纬度：',coordinate])
  });

  // LOG.warn(['添加成功 经纬度：',coordinate])


  // 侦听事件
  if (eventCallback) {
    poi.addEventListener('click', async (event) => {
      const id = event.detail.target
      // console.log(id,eventCallback)
      eventCallback(id)
    });
  }

  const res = await poi.getBoundingInfo()
  return res

  // if (isFocus)
  //   await focusEntityById(client, id, 50000)
}

export async function addPOI2D(client, id, title = 'poi test', resourceName = 'POI圆形标签',
                               coordinate, icon, bgColor, isFocus, eventCallback) {
  // await deleteEntityByIdList(client, [id]).catch(e => Vue.prototype.LOG.warn(e))

  const poi2d = new POI2D({
    id,
    name: id,
    group: SuffixForOverlayId.getGroupValueById(id),
    resource: {
      name: 'POI圆形2D',
    },
    coordinate,
    scale: new Vector3(0.5, 0.5, 0.5),
    clickable: true,
    visible: true,
    icon,
    // title,
    autoSize: true, //可选，默认关闭；开启时以屏幕运算，无论远近像素大小固定；关闭时以三维空间运算，近大远小
    backgroundColor: bgColor,
    opacity: 1,
    lineLength: 100,
  })
  await poi2d.setClient(client).catch(e => LOG.warn([e]));

  // 侦听事件
  if (eventCallback) {
    poi2d.addEventListener('click', async (event) => {
      const id = event.detail.target
      eventCallback(id)
    });
  }

  // if (isFocus)
  //   await focusEntityById(client, id, 50000)
}

export async function addEmissive(client, id, resourceName = '悬浮标记04',
                                  coordinate, color = '#fff', isFocus = false) {
  await deleteEntityByIdList(client, [id]).catch(e => Vue.prototype.LOG.warn([e]))

  const emissive = new Emissive({
    name: id,
    group: '/MEntity/' + id.substr(0, SuffixForOverlayId.length),
    resource: {
      name: resourceName,
    },
    coordinate,
    rotation: new Vector3(0, 0, 0),
    scale: new Vector3(5, 5, 5),
    clickable: true,
    visible: true,
    emissionColor: color,
    emissionStrength: 0,
    speed: 0,
  });

  await emissive.setClient(client).catch(e => Vue.prototype.LOG.warn([e]))

  // if (isFocus)
  //   await emissive.focus(50000, -Math.PI / 6)
}

export async function addPOICustomPoi(client, id, title = 'poi test', resourceName = 'POI圆形标签',
                                      coordinate, icon, bgColor, isFocus, eventCallback) {
  // await deleteEntityByIdList(client, [id]).catch(e => Vue.prototype.LOG.warn(e))

  const poi2d = new POI2D({
    id,
    name: id,
    group: SuffixForOverlayId.getGroupValueById(id),
    resource: {
      name: 'POI圆形2D',
    },
    coordinate,
    scale: new Vector3(0.5, 0.5, 0.5),
    clickable: true,
    visible: true,
    icon,
    // title,
    autoSize: true, //可选，默认关闭；开启时以屏幕运算，无论远近像素大小固定；关闭时以三维空间运算，近大远小
    backgroundColor: bgColor,
    opacity: 1,
    lineLength: 100,
  })

  getCustomPoiBase64(title, bgColor, icon, async (url) => {
    const response = await client.request('CreateEntity', {
      id,        // 对象标识符，必须全局唯一，建议使用 GUID
      type: "image2d",                    // 指定类型
      name: id,                    // 可选，默认自动生成，资源列表中的名称，必须全局唯一
      group: SuffixForOverlayId.getGroupValueById(id),    // 可选，默认资源列表根目录（无组），指定资源列表中的目录名称，如不存在将自动创建
      visible: true,                      // 可选，可见性，默认 true
      locked: false,                      // 可选，锁定状态，默认 false
      coordinate,   // 可选，[经度,纬度,高度(单位: m)]，仅CIM模式下支持，指定 coordinate 时可不指定 position
      scale: [0.5, 0.5, 0.5],                // 可选，[scaleX,scaleY,scaleZ]，默认[1.0,1.0,1.0]
      property: {
        // color: [255, 255, 0],     // 可选，叠加颜色，[r,g,b]，取值范围[0,0,0]~[255,255,255]，默认[255,255,255]即不叠色
        opacity: 0.9,       // 可选，不透明度，默认为 1.0
        url,               // 必填，显示的图片，支持 http/https、file协议和本地路径，支持base64图片编码
      }
    });
  })

  // 侦听事件
  if (eventCallback) {
    // poi2d.addEventListener('click', async (event) => {
    //   const id = event.detail.target
    //   eventCallback(id)
    // });
  }

  // if (isFocus)
  //   await focusEntityById(client, id, 50000)
}

export function getCustomPoiBase64(title, color, icon, callback) {
  const id = 'custom-poi'
  const poiDom = document.getElementById(id)

  html2canvas(poiDom, {}).then(canvas => {
    let dataURL = canvas.toDataURL("image/png", 1)
    callback && callback(dataURL)
  })
}

/**
 * poi聚合  个数小于<40不触发聚合
 * @param client
 * @param id
 * @param idList
 * @returns {Promise<void>}
 */
export async function addPOICluster(client, id, idList = []) {
  await client.request("DestroyPOICluster", {id}).catch(e => {
  })
  if (idList.length>=40){
    const result = await client.request('CreatePOICluster', {
      id,         // 必填，当前 entityId
      group: SuffixForOverlayId.getGroupValueById(id),
      level: 10,                  // 必填，聚合层级数, 最高为 10, 最低为 2
      entityIds: idList,
      minClusterCount: 20,        // 可选，触发聚合的最低数量，默认为 2
      styles: [                  // 可选,
        {
          maxCount: 10,           // 必填, 当聚合点下聚合的POI数量低于 maxCount 时，使用当前样式
          textColor: [255, 255, 255],   // 可选, 文字颜色，默认为 [0, 0, 0]
          color: [2, 126, 248],     // 可选, 叠加颜色，默认为 [255, 255, 255]
          opacity: 1.0,           // 可选, 不透明度, 默认为 1
          // icon: "x://xxx.png",    // 可选, 自定义图标, 支持 http/https、file 协议和本地路径，支持 base64 图片编码
          scale: [1.0, 1.0]       // 可选, 聚合标记缩放，默认为 [1, 1], 默认高度为 1920x1080 分辨率下的 100px, 宽度根据图片比例计算
        },
        {
          maxCount: 60,           // 必填, 当聚合点下聚合的POI数量低于 maxCount 时，使用当前样式
          textColor: [255, 255, 255],   // 可选, 文字颜色，默认为 [0, 0, 0]
          color: [2, 126, 248],     // 可选, 叠加颜色，默认为 [255, 255, 255]
          opacity: 1.0,           // 可选, 不透明度, 默认为 1
          // icon: "x://xxx.png",    // 可选, 自定义图标, 支持 http/https、file 协议和本地路径，支持 base64 图片编码
          scale: [1.0, 1.0]       // 可选, 聚合标记缩放，默认为 [1, 1], 默认高度为 1920x1080 分辨率下的 100px, 宽度根据图片比例计算
        },
        {
          maxCount: 500,           // 必填, 当聚合点下聚合的POI数量低于 maxCount 时，使用当前样式
          textColor: [255, 255, 255],   // 可选, 文字颜色，默认为 [0, 0, 0]
          color: [2, 126, 248],     // 可选, 叠加颜色，默认为 [255, 255, 255]
          opacity: 1.0,           // 可选, 不透明度, 默认为 1
          // icon: "x://xxx.png",    // 可选, 自定义图标, 支持 http/https、file 协议和本地路径，支持 base64 图片编码
          scale: [1.0, 1.0]       // 可选, 聚合标记缩放，默认为 [1, 1], 默认高度为 1920x1080 分辨率下的 100px, 宽度根据图片比例计算
        }
      ]
    });
    LOG.info(['聚合回调', result])
  }else {
    LOG.info(['poi个数小于40阈值，取消聚合回调'])
  }

}
