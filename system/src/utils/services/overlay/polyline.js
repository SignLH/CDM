import {Polyline, PolylineInterpMode, Material} from '@sheencity/diva-sdk';
import {Vector3} from '@sheencity/diva-sdk-math';
import {POI, POIIcon} from '@sheencity/diva-sdk';
import {deleteEntityByIdList, SuffixForOverlayId} from "@/utils/services/overlay/entity";
import {getRandomColor} from "@/utils/color";
import {Model} from '@sheencity/diva-sdk';
import LOG from "@/utils/self-log";

const polyline = new Polyline({
  id: 'polyline_test1',
  name: 'polyline_test1',
  group: 'group_test',
  visible: true,
  clickable: true,
  positions: [
    new Vector3(5510, 5510, 100),
    new Vector3(0, 0, 100),
    new Vector3(-5510, -5510, 100),
  ],
  // coordinates: [
  //   [116.46011605252606, 39.91999816888632, 0],
  //   [116.46011605237283, 39.91990810564861, 0],
  //   [116.46023302057945, 39.91999816870935, 0],
  // ],
  style: '三角',
  interpMode: PolylineInterpMode.Linear,
  width: 300,
  material: new Material({
    color: '#0064c8',
    emissive: 1.0,
    opacity: 0.2,
    speed: 0.5,
  }),
});


/**
 * 创建轨迹路线  包括创建两条路径和三个点
 * @param client
 * @param carId  要保存 删除要使用它
 * @param coordinates  终点 --- 汽车点 --- 起点
 * @param isUseCoord
 * @returns {Promise<void>}
 */
export async function addCarPolyline(client, carId = 'testCar1', coordinates = [
  [121.53, 31.67, 10], [121.528, 31.67, 10], [121.528, 31.668, 10]]) {
  const carIdList = [
    //线
    `${SuffixForOverlayId.polyline}1-${carId}`, `${SuffixForOverlayId.polyline}2-${carId}`,
    //点
    `${SuffixForOverlayId.poi}1-${carId}`,
    `${SuffixForOverlayId.poi}2-${carId}`,
    `${SuffixForOverlayId.poi}1-${carId}`
  ]

  //删除 如果有
  await deleteEntityByIdList(client, carIdList)


  //----创建----
  //创建线路
  const polylineCarStyle = [
    //已走过
    {
      id: carIdList[0],
      name: carIdList[0],
      group: '/MEntity/' + SuffixForOverlayId.polyline,
      visible: true,
      clickable: true,
      coordinates: [coordinates[0], coordinates[1]],
      style: '自发光',
      interpMode: PolylineInterpMode.Linear,
      width: 300,
      material: new Material({
        color: getRandomColor(),
        emissive: 1.0,
        opacity: 0.2,
        speed: 0.5,
      }),
    },
    //未走过
    {
      id: carIdList[1],
      name: carIdList[1],
      group: '/MEntity/' + SuffixForOverlayId.polyline,
      visible: true,
      clickable: true,
      coordinates: [coordinates[1], coordinates[2]],
      style: '三角',
      interpMode: PolylineInterpMode.Linear,
      width: 300,
      material: new Material({
        color: getRandomColor(),
        emissive: 1.0,
        opacity: 0.2,
        speed: 0.5,
      }),
    },
  ]

  for (let polylineCarStyleElement of polylineCarStyle) {
    const polyline = new Polyline(polylineCarStyleElement)
    await polyline.setClient(client).catch(e => LOG.warn([e]))
  }

  const poiList = [
    //start
    {
      id: carIdList[2],
      name: carIdList[2],
      group: '/MEntity/' + SuffixForOverlayId.poi,
      resource: {
        name: 'POI水滴',
      },
      coordinate: coordinates[0],
      // coordinate: [116.445, 39.918, 0],
      scale: new Vector3(0.5, 0.5, 0.5),
      clickable: true,
      visible: true,
      icon: POIIcon.Location,
      autoSize: true, //可选，默认关闭；开启时以屏幕运算，无论远近像素大小固定；关闭时以三维空间运算，近大远小
      backgroundColor: '#2af4da',
      opacity: 1,
    },
    //car
    {
      id: carIdList[3],
      name: carIdList[3],
      group: '/MEntity/' + SuffixForOverlayId.poi,
      resource: {
        name: 'POI水滴',
      },
      coordinate: coordinates[1],
      // coordinate: [116.445, 39.918, 0],
      scale: new Vector3(0.5, 0.5, 0.5),
      clickable: true,
      visible: true,
      icon: POIIcon.Location,
      autoSize: true, //可选，默认关闭；开启时以屏幕运算，无论远近像素大小固定；关闭时以三维空间运算，近大远小
      backgroundColor: '#ff0000',
      opacity: 1,
    },
    //end
    {
      id: carIdList[4],
      name: carIdList[4],
      group: '/MEntity/' + SuffixForOverlayId.poi,
      resource: {
        name: 'POI水滴',
      },
      coordinate: coordinates[2],
      // coordinate: [116.445, 39.918, 0],
      scale: new Vector3(0.5, 0.5, 0.5),
      clickable: true,
      visible: true,
      icon: POIIcon.Location,
      autoSize: true, //可选，默认关闭；开启时以屏幕运算，无论远近像素大小固定；关闭时以三维空间运算，近大远小
      backgroundColor: '#117e7c',
      opacity: 1,
    },
  ]

  for (let poiData of poiList) {
    const poi = new POI(poiData)
    await poi.setClient(client).catch(e => LOG.warn([e]))
  }
}

/**
 * 添加路线动画
 */
export async function addPolylineAnimal(client, id, color, coordinates, duration) {
  const model = new Model({
    id: id,
    name:id,
    group: SuffixForOverlayId.getGroupValueById(id),
    resource: {name: '环卫清洁车01'},
    coordinate:coordinates[0],
    visible: true,
    clickable: true,
    scale: new Vector3(10, 10, 10),
  });
  await model.setClient(client);

  await client.request('SetPathAnimation', {
    id,                      // 目标对象的 id
    duration,                   // 必填，动画持续时间，单位（秒）
    loop: true,                       // 可选，是否循环，默认false
    path: {
      coordinates: coordinates,  // 仅CIM模式下可用，指定coordinates时可不指定positions，元素格式 [经度,纬度,高度(单位: m)]
    },
    interpMode: "curve",              // 可选，插值算法，"curve"、"linear" 默认 "curve",
    relativeRotation: [0, 0, 0],     // 可选，相对路线的旋转角度 [roll,pitch俯仰角,yaw偏航角(0沿路线，180反向)]
    property: {
      action: {                       // 特殊，仅当实体类型为 character 时可支持
        name: "跑"                    // 路径动画的特定动作，name 选项："走"、"跑"、"骑单车"
      }
    },
    vertical: false                   // [>=4.5.2] 可选，是否垂直于地面, 默认 false
  });

}

/**
 * 添加路线
 */
export async function addPolyline(client, id, color,
                                  coordinates = [[121.53, 31.67, 10], [121.528, 31.67, 10]],
                                  style='管道',opacity=1) {
  //删除 如果有
  await deleteEntityByIdList(client, [id])
  //----创建----
  //创建线路
  const polyline = new Polyline({
    id: id,
    name: id,
    group: SuffixForOverlayId.getGroupValueById(id),
    visible: true,
    clickable: true,
    coordinates,
    style,
    interpMode: PolylineInterpMode.Linear,
    width: 3000,
    material: new Material({
      color,
      emissive: 1.0,
      speed: 0.5,
      opacity
    }),
  })
  await polyline.setClient(client).catch(e => LOG.warn([e]))
}
