import {Diva} from '@sheencity/diva-sdk';
import {getDigitalTwinCloudUrl} from "@/utils/app-base";
import LOG from "@/utils/self-log";

export class DivaService {
  /**
   * divaClient
   * @type {import('@sheencity/diva-sdk').DivaClient}
   */
  client;

  /**
   * 初始话 webRtc 链接
   * @param element (HTMLElement) 视频加载的 dom 元素
   * @param callback
   * @param isReceiveVideo
   */
  async init(element, callback, isReceiveVideo = true) {
    let diva;
    if (this.isEmbeddedMode() && false) {
      // 使用内嵌模式
      diva = new Diva({
        mode: 'embedded',
        // apiKey: 'xxx',
        container: element,
      });
    } else {
      // 使用云渲染模式
      diva = new Diva({
        mode: 'cloud',
        // apiKey: 'xxx',
        container: element,
        url: new URL(getDigitalTwinCloudUrl()),
        // 可选，日志类型；'resolved'：在控制台以字符串形式打印；'raw'：以二进制形式打印；'none'：不打印。默认：'resolved'
        logType: 'none',
        // [>= sdk v1.1.0]可选，是否禁用浏览器键的默认行为；包含 F1~F12 及 Tab 键。默认：true
        suppressBrowserKeys: false,
        //[>= sdk v1.2.0]可选，是否接收视频流传输；默认：true
        receiveVideo: isReceiveVideo,
      });
    }
    try {
      this.client = await diva.init();
      LOG.info(["加载地图成功"])
      callback(true)
    } catch (e) {
      LOG.info(["加载地图失败", e])
      callback(false)
    }
    // LOG.info(['client is', this.client);
    // diva.init().then(res => {
    //   this.client = res
    // }).catch(e => {
    // })
  }

  /**
   * 判断是否启用内嵌模式
   * @returns 内嵌模式下返回 true
   */
  isEmbeddedMode() {
    return window.navigator.userAgent.includes('Mars');
  }
}
