const getters = {
  size: state => state.app.size,
  pageName: state => state.app.pageName,
  leftOpen: state => state.app.leftOpen,
  rightOpen: state => state.app.rightOpen,
  screenIds: state => state.app.screenIds,
  pageMonitorList: state => state.app.pageMonitorList,
  haveAddedJson: state => state.map.haveAddedJson,
  monitorVideoUrl: state => state.map.monitorVideoUrl,
  poiDeviceShow: state => state.map.poiDeviceShow,
  poiClickId: state => state.map.poiClickId,
  currentTime: state => state.time.currentTime,
  currentScene: state => state.time.currentScene,
}
export default getters
