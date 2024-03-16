import LOG from "@/utils/self-log";
import {AppBaseConfig, getDigitalTwinCloudUrl} from "@/utils/app-base";

/**
 * 通过 SDK 可以给场景内指定的 Entity 创建弹出窗口。
 *
 * 创建成功后窗口将立即呈现；
 * 弹窗包含关闭按钮，用户可通过按钮关闭窗口，窗口关闭后将自动销毁；
 * 一个 Entity 目前仅支持创建唯一的 WebWidget，创建时，如果 Entity 已存在 WebWidget，将会自动销毁前一个，再创建当前设置的 WebWidget；
 * 如果 url 为 rtmp 协议或者以 mp4 结尾时，将会默认切换为内置播放器启动弹窗，其他将作为浏览器弹窗。
 *
 * //销毁弹窗
 * await model.setWebWidget(null);
 *
 */
/**
 * 创建弹窗
 * @param model
 * @param url 链接
 * @param isCenter 是否居中
 * @param width 大小
 * @param height 大小
 * @param closable 是否显示原生关闭按钮
 */
export function setWebWidget(model, url, isCenter = true,
                             width=400,height=260,closable=true) {
  // const model = await client.getEntityById('xxxxxx');
  const getOffset = () => {
    return {x: - width/2, y: -400}
  }

  //修改url添加传递地图地址
  url+=`&${AppBaseConfig.DigitalTwinCloudUrl}=${getDigitalTwinCloudUrl()}`

  const data = {
    width,                    // 弹窗宽度，单位（px）
    height,                   // 弹窗高度，单位（px）
    position: 'right',              // 可选，"left" "right"，指定弹出界面在模型的哪侧
    angle: 10,                     // 可选，以竖直方向为基准，竖直连线向左/右偏移角度，单位：角度
    transparent: true,             // 可选，是否启用透明度，开启后配合内嵌网页背景为透明以支持异形窗口
    closable,                // 可选，是否启用右上方关闭按钮
    dismissable: false,            // 可选，是否启用单击窗口外自动关闭（右键或左键长按拖动不会触发关闭）
    mouseInput: true,              // [>=v0.2.3] 可选，是否接收鼠标事件，默认 false
    keyboardInput: true,           // [>=v0.2.3] 可选，是否接收键盘事件，默认 false，单独开启键盘事件不生效，需同时开启鼠标事件
    line: {
      visible: true,               // 可选，连线是否显示
      color: '#00f',               // 可选，连线颜色
      horizontalLineLength: 0,    // 可选，水平连线长度，单位：像素
      verticalLineLength: 300,      // 可选，竖直连线长度，单位：像素
      connectorVisible: true,      // 可选，连接点是否显示
    },
    offset:isCenter?getOffset():null
  }
  if (!isCenter) {
    data.offset = null
  }

  LOG.info(['set window url=',url])
  model.setWebWidget(new URL(url), data)
}


export async function clearWindow(client, id) {
  try {
    const model = await client.getEntityById(id)
    model.setWebWidget(null)
  }catch (e) {
    LOG.warn([e])
  }
}


