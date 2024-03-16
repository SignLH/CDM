import {Material, Polygon, Vector3} from '@sheencity/diva-sdk';
import {getRandomColor} from "@/utils/color";
import {focusEntityById, SuffixForOverlayId} from "@/utils/services/overlay/entity"
import LOG from "@/utils/self-log";

const polygon = new Polygon({
  id: 'polygon_test',
  name: 'polygon_test',
  group: 'group_test',
  visible: true,
  clickable: true,
  // positions: [
  //   new Vector3(10065.2734375,-2998.427734375, 100),
  //   new Vector3(12469.18359375,30971.5703125, 100),
  //   new Vector3(-22624.1015625,-13298.5576171875, 100),
  //   new Vector3(-12468.0546875,-27346.666015625, 100),
  // ],
  coordinates: [
    [116.46011605252606, 39.91999816888632, 0],
    [116.46011605237283, 39.91990810564861, 0],
    [116.46023302057945, 39.91999816870935, 0],
  ],
  style: '围栏渐变',
  height: 30000,
  material: new Material({
    color: '#0064c8',
    emissive: 1,
    opacity: 0.2,
    speed: 0.5,
    scale: 1.0,
  }),
  definition: [
    {enabled: false, type: 'bottom', material: new Material({opacity: 0.5})},
    {enabled: true, type: 'side', material: new Material({opacity: 1})},
    {enabled: true, type: 'top' },
  ],
});

/**
 * 添加网格
 */
export async function addGridPolygon_bak(client, gisstr, id, color,gridName, isFocus = false, eventCallback,height=8000) {
  // await deleteEntityByIdList(client, [id]).catch(e => LOG.warn([e]))
  // 获取地图位置
  const split = gisstr.split(';');
  const coordinates = []

  split.forEach(e => {
    let splitList2 = e.split(',').map(e => Number(e));
    splitList2.push(-1)
    coordinates.push(splitList2)
  })
  if (!color) {
    color = getRandomColor()
  }
  const polygon = new Polygon({
    id: id,
    name: id,
    group: SuffixForOverlayId.getGroupValueById(id),
    visible: true,
    clickable: true,
    coordinates,
    style: '围栏管道平移',
    height,
    material: new Material({
      color,
      emissive: 1,
      opacity: 0.2,
      speed: 0.5,
      scale: 1.0,
    }),
    definition: [
      {enabled: true, type: 'bottom', material: new Material({emissive: 0, color, opacity: 0})},
      {enabled: true, type: 'side', material: new Material({opacity: 1})},
      {enabled: false, type: 'top', material: new Material({emissive: 0, opacity: 0})},
    ],
  })

  // 侦听事件
  if (eventCallback) {
    polygon.addEventListener('click', async (event) => {
      const id = event.detail.target
      eventCallback(id)
    });
  }

  await polygon.setClient(client).catch(e => LOG.warn([e]))
  if (isFocus)
    await focusEntityById(client, id, 400000)

  return polygon
}

/**
 * 添加网格
 */
export async function addGridPolygon(client, gisstr, id, color,gridName, isFocus = false, eventCallback,height=8000) {
  // await deleteEntityByIdList(client, [id]).catch(e => LOG.warn([e]))
  // 获取地图位置
  const split = gisstr.split(';');
  const coordinates = []

  split.forEach(e => {
    let splitList2 = e.split(',').map(e => Number(e));
    splitList2.push(-1)
    coordinates.push(splitList2)
  })
  if (!color) {
    color = getRandomColor()
  }
  const { result } = await client.request("LngLatToWorld", {  //经纬度坐标组转为笛卡尔坐标组
    coordinates,
  });
  const a = result.positions;      //获取转化后的笛卡尔坐标组
  const position = a[0];         //取笛卡尔坐标组中的第一个点作为polygon原点
  const repositions = a.map((v) =>                         //计算并得到坐标组中相对原点的相对位置坐标
    Vector3.FromArray(v.map((v, i) => v - position[i]))
  );
  //创建polygon，其他参数不变，传入原点position，以及相对位置坐标组repositionF

  const polygon = new Polygon({
    id: id,
    name: id,
    group: SuffixForOverlayId.getGroupValueById(id),
    visible: true,
    clickable: true,
    position: Vector3.FromArray(position),   //增加polygon模型原点参数并传入
    positions: repositions,                 //传入相对模型原点位置
    style: '围栏管道平移',
    height,
    material: new Material({
      color,
      emissive: 1,
      opacity: 0.2,
      speed: 0.5,
      scale: 1.0,
    }),
    definition: [
      {enabled: true, type: 'bottom', material: new Material({emissive: 0, color, opacity: 0})},
      {enabled: true, type: 'side', material: new Material({opacity: 1})},
      {enabled: false, type: 'top', material: new Material({emissive: 0, opacity: 0})},
    ],
  })

  // 侦听事件
  if (eventCallback) {
    polygon.addEventListener('click', async (event) => {
      const id = event.detail.target
      eventCallback(id)
    });
  }

  await polygon.setClient(client).catch(e => LOG.warn([e]))
  if (isFocus)
    await focusEntityById(client, id, 400000)

  return polygon
}
