import {DivaService} from "@/utils/services/diva.service";

//API地址 根路径
export function getApiBaseUrl() {
  if (process.env.NODE_ENV === 'development') {
    return 'https://dptest.milkway.tech:2333'
  } else {
    return window.CONFIG[window.ENV].ApiBaseUrl
  }
}

//图片地址 根路径
export function getImageBaseUrl() {
  if (process.env.NODE_ENV === 'development') {
    return 'https://dptest.milkway.tech:2337/'
  } else {
    return window.CONFIG[window.ENV].ImageBaseUrl
  }
}

export const mapService = new DivaService();
// export const data = new DataService();

//存储在LocalStorage名称
export const AppBaseConfig = {
  DigitalTwinCloudUrl: 'DigitalTwinCloudUrl'
}

//底图地址
export function getDigitalTwinCloudUrl() {
  //获取用户设置
  const useDigitalTwinCloudUrl = localStorage.getItem(AppBaseConfig.DigitalTwinCloudUrl)
  if (process.env.NODE_ENV === 'development') {
    return useDigitalTwinCloudUrl || 'ws://192.168.43.244:3000'
  } else {
    return useDigitalTwinCloudUrl || window.CONFIG[window.ENV].DigitalTwinCloudUrl
  }
}

//本机服务器地址
export function getLocalServiceUrl() {
  if (process.env.NODE_ENV === 'development') {
    return 'http://192.168.43.181:9934'
  } else {
    return window.CONFIG[window.ENV].LocalServiceUrl
  }
}

//当镜头高度小于miniWorldHeight 网格大小改变 阻止网格弹窗  单位cm
export const miniWorldHeight = 30000

// export const MQ_SERVICE = 'ws://127.0.0.1:61614/stomp' // mq服务地址(默认监听消息端口)
// export const MQ_USERNAME = 'xxx' // mq连接用户名
// export const MQ_PASSWORD = 'xxx' //mq连接密码
