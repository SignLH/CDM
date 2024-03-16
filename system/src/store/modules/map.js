import {addPOI, addPOICluster} from "@/utils/services/overlay/poi";
import {getLocalServiceUrl, mapService, miniWorldHeight} from "@/utils/app-base";
import {deleteEntityByGroup, focusEntityById, SuffixForOverlayId} from "@/utils/services/overlay/entity";
import {getGroupNameObjById, POIIcon} from "@/utils/services/overlay/overlay.model";
import {getRandomColor, XjdwTypeColor} from "@/utils/color";
import {
  getBuildInfo,
  getCameraDevices,
  getCarTrackList,
  getInspectionGis,
  getIntelligentDevices,
  getMapGridList,
  getPartyCommitteeInfo
} from "@/api/map";
import {addGridPolygon} from "@/utils/services/overlay/polygon";
import {addPolyline, addPolylineAnimal} from "@/utils/services/overlay/polyline";
import {setWebWidget} from "@/utils/services/web-widget";
import {cameraFocusCoordinate, getCameraInfo, overallFocusFun, setCameraStatus} from "@/utils/services/camera";
import {clickEventListener, wheelEventListener} from "@/utils/services/event-listener";
import LOG from "@/utils/self-log";

const state = {
  //监控实体
  monitorVideoUrl: null,
  monitorDevicesMap: new Map(),
  currentWindowId: null,

  //have been added entity
  haveAddedEntity: new Map(),
  //have been added entity json  用于其他组件监听新添加的地图实体数据 type是SuffixForOverlayId元素 value对应数据数组
  haveAddedJson: {type: null, value: null},
  //智能设备映射 根据id 映射 entity 突出一个重点
  poiDeviceMap: new Map(),
  //智能设备实体数据 数据改变根据类型显示弹窗
  poiDeviceShow: null,
  //poi点击 id
  poiClickId: null,
}

const mutations = {
  Set_Monitor_Video_Url(state, value) {
    LOG.warn(['Set_Monitor_Video_Url', value])
    state.monitorVideoUrl = value
  },
  SetCurrentWindowId(state, value) {
    state.currentWindowId = value
  },
  Update_Monitor_Devices_Map(state, value) {
    state.monitorDevicesMap.set(value.id, value)
  },
  UpdateHaveAddedEntity(state, v = {key: '', value: ''}) {
    const haveAddedEntity = state.haveAddedEntity
    if (haveAddedEntity.get(v.key) == null) {
      haveAddedEntity.set(v.key, [])
    }
    haveAddedEntity.get(v.key).push(v.value)
  },

  UpdatePoiDeviceMap(state, v = {key: '', value: ''}) {
    state.poiDeviceMap.set(v.key, v.value)
  },
  ClearPoiDeviceMap(state,) {
    state.poiDeviceMap.clear()
  },
  SetPoiDeviceShow(state, value) {
    state.poiDeviceShow = value
  },
}


const actions = {

  SetMonitorVideoUrl({commit}, data) {
    commit('Set_Monitor_Video_Url', data)
  },

  /**
   * 撒点党群服务点
   * @param commit
   * @param state
   * @constructor
   */
  async GetPartyServicePosition({state, commit}) {
    state.haveAddedEntity.set(SuffixForOverlayId.polygon, [])
    getPartyCommitteeInfo({type: '党群服务点'}).then(async res => {
      const boxList = []
      if (res.result && res.result.length > 0) {
        for (let i = 0; i < res.result.length; i++) {
          let party = res.result[i]
          let gisStr = ((party.gis_map[0] || {}).gisstr_wgs || '0,0').split(',')
          const gisList = [Number(gisStr[0]), Number(gisStr[1]), 0]
          if (gisList[0] === 0 || gisList[0] === 1) {
            LOG.warn('getPartyCommitteeInfo，过滤空值经纬度')
            continue
          }
          const resPoi = await addPOI(mapService.client, SuffixForOverlayId.poiServiceStation + party.id,
            party.name, 'POI圆形标签', gisList, getLocalServiceUrl() + '/img/dh.png', '#ff1e1e', true,
            async (id) => {
              const trueId = id.substring(SuffixForOverlayId.length)
              const data = res.result.find(value => value.id === trueId)

              let url = getLocalServiceUrl() + '/#/service-station-poi?obj='
                + JSON.stringify({name: data.name}) + '&token=' + sessionStorage.getItem('LoginToken')

              const model = await mapService.client.getEntityById(id);
              setWebWidget(model, url, true, 600, 400)
            })
          boxList.push(resPoi.boundingBox.center)
        }
        if (res.result.length > 1) {
          await overallFocusFun(mapService.client, boxList)
        } else {
          let id = SuffixForOverlayId.poiServiceStation + res.result[0].id
          await focusEntityById(mapService.client, id)
        }
      } else {
        console.error("获取网络数据失败！")
      }
    })
  },

  /**
   *撒点党建信息包括人员
   * @param commit
   * @param name
   * @constructor
   */
  async GetPartyCommitteeInfo({commit}, name = '东平镇党委') {
    await deleteEntityByGroup(mapService.client, '/MEntity/' + SuffixForOverlayId.poiParty)
    getPartyCommitteeInfo({name}).then(async res => {
      if (res.status) {
        for (let party in res.result) {
          await addPOI(mapService.client, SuffixForOverlayId.poiParty + party.id, party.name, 'POI圆形标签',
            [party.latitude, party.longitude, 0], POIIcon.Toilet, '#ff1e1e', true)
          for (let resultKey in party.member) {
            const id = SuffixForOverlayId.poi + resultKey.id
            await addPOI(mapService.client, id, resultKey.wcusername, 'POI圆形标签',
              [resultKey.latitude, resultKey.longitude, 0], POIIcon.Location, getRandomColor(), false)
          }
        }

      } else {
        console.error("获取网络数据失败！")
      }
    })
  },

  /**
   * 撒点人员力量
   * @param commit
   * @param types
   * @constructor
   */
  async GetPeoplePower({commit},) {
    await deleteEntityByGroup(mapService.client, '/MEntity/' + SuffixForOverlayId.poiPeople)
    const boxList = [], poiIdList = []
    getMapGridList({gistype2: '责任网格'}).then(async res => {
      for (let i = 0; i < (res.result || []).length; i++) {
        let resultKey = res.result[i]
        const color = resultKey.colour || getRandomColor()
        const id = SuffixForOverlayId.poiPeople + resultKey.id
        const resPoi = await addPOI(mapService.client, id, resultKey.gisname,
          'POI圆形标签', [Number(resultKey.longitude), Number(resultKey.latitude), 0]
          , getLocalServiceUrl() + '/img/people-power.png', color, false,
          async (id) => {
            const trueId = id.substring(SuffixForOverlayId.length)
            const data = res.result.find(value => value.id === trueId)
            try {
              const model = await mapService.client.getEntityById(id);
              model.focus(400000, -Math.PI / 6)
              let queryData = {gistype2: data.gistype2, gisname: data.gisname}
              let url = getLocalServiceUrl() + '/#/lqld-wangge?obj='
                + JSON.stringify(queryData) + '&token=' + sessionStorage.getItem('LoginToken')
              setWebWidget(model, url)
              LOG.info(['addGridPolygon setWebWidget url=', url])
            } catch (e) {
              LOG.warn([e])
            }
          })

        poiIdList.push(id)
        boxList.push(resPoi.boundingBox.center)
      }

      await overallFocusFun(mapService.client, boxList)
      //添加 聚合
      await addPOICluster(mapService.client, SuffixForOverlayId.poiDevice + 'poiCluster', poiIdList)
    })
  },

  /**
   * 撒点智能设备
   * @param commit
   * @param device_type 设备类型
   * @param gisname 社区名称
   * @param isCameraRange 是否获取从当前镜头点位信息
   * @constructor
   */
  async GetIntelligentDevices({commit}, {device_type = ['视频监控'], gisname = null, isCameraRange = true}) {
    //清空缓存
    commit('ClearPoiDeviceMap')
    await deleteEntityByGroup(mapService.client, '/MEntity/' + SuffixForOverlayId.poiDevice)
    let queryData = {
      device_type,
      gisname,
      curpage: 1,
      pagesize: 100
    }
    if (isCameraRange) {
      const cameraInfo = await getCameraInfo(mapService.client)
      const coordinate = cameraInfo.coordinate
      queryData.longitude = coordinate[0]
      queryData.latitude = coordinate[1]
      queryData.distance = 5 // km 正方形区域
    }
    getIntelligentDevices(queryData).then(async res => {
      const dataList = res.result || []
      const poiIdList = []
      const boxList = []
      for (let i = 0; i < dataList.length; i++) {
        const data = dataList[i]
        let style = {
          icon: POIIcon.Building,
          color: '#00b4ff',
        }
        if (data.device_type === '单兵') {
          style = {
            icon: POIIcon.AlarmStation,
            color: '#ff1e1e',
          }
          //设备离线
          if (data.device_status != 1) {
            style.color = '#d2cfcf'
          }
        } else if (data.device_type === '视频监控') {
          style = {
            icon: POIIcon.Camera,
            color: '#409EFF',
          }
        }
        const poiId = SuffixForOverlayId.poiDevice + data.id;
        const coordinate = [Number(data.longitude), Number(data.latitude), 0]
        if (!coordinate[0] || !coordinate[1]) {
          LOG.warn(['已过滤空经纬度poi'])
          continue
        }
        let res = {}
        try {
          if (data.device_type === '视频监控') {
            commit('Update_Monitor_Devices_Map', data)
            res = await addPOI(mapService.client, poiId,
              data.device_name, 'POI圆形标签', coordinate,
              style.icon, style.color, i === dataList.length - 1,
              async (id) => {
                //地图点击视频监控poi change ；monitorVideoUrl
                const datum = state.monitorDevicesMap.get(id.substring(SuffixForOverlayId.length));
                await focusEntityById(mapService.client, id)
                commit('Set_Monitor_Video_Url', {...datum})
              })
          } else {
            res = await addPOI(mapService.client, poiId,
              data.device_name, 'POI圆形标签', coordinate,
              style.icon, style.color, i === dataList.length - 1)
          }
          poiIdList.push(poiId)
          boxList.push(res.boundingBox.center)

          //加入缓存
          commit('UpdatePoiDeviceMap', {key: poiId, value: data})
        } catch (e) {
          LOG.warn(['添加poi失败', e])
        }
      }
      await overallFocusFun(mapService.client, boxList)
      //添加 聚合
      await addPOICluster(mapService.client, SuffixForOverlayId.poiDevice + 'poiCluster', poiIdList)
    })
  },

  /**
   * 撒点重点监控
   * @param state
   * @param commit
   * @param obj
   * @constructor
   */
  async GetCameraDevices({state, commit}, obj) {
    const scene_id = (obj || {}).scene_id
    await deleteEntityByGroup(mapService.client, '/MEntity/' + SuffixForOverlayId.poiMonitorDevice)
    getCameraDevices({scene_id}).then(async res => {
      const dataList = res.result || []
      for (let i = 0; i < dataList.length; i++) {
        const data = dataList[i]
        commit('Update_Monitor_Devices_Map', data)
        await addPOI(mapService.client, SuffixForOverlayId.poiMonitorDevice + data.id,
          data.device_name, 'POI圆形标签',
          [Number(data.longitude), Number(data.latitude), 0],
          POIIcon.Camera, '#409EFF', i === dataList.length - 1,
          async (id) => {
            const datum = state.monitorDevicesMap.get(id.substr(SuffixForOverlayId.length));
            await focusEntityById(mapService.client, id)
            commit('Set_Monitor_Video_Url', {
              ...datum
            })
          })
      }
    })
  },

  /**
   * 添加网格
   * @param commit
   * @param gistype1 api接口字段
   * @param gistype2 api接口字段
   * @param isDelBefore  是否删除上一次地图构建的实体数据
   * @param gisname api接口字段
   * @param isShowWindow 是否显示弹窗
   * @param gridType   1网格（首页左上角）     2社区（智慧社区左上角）  3智慧党建
   * @param is_community 字段 is_community = 1 全部， = 0，首页撒全部只撒6个居委
   * @param distance  聚焦距离
   * @param girdList  不为空不请求接口直接渲染此数据
   * @param sub_gis_flag  是否获取节点下的所有叶子 sub_gis_flag ,默认 false ，不返回
   * @param poiClickCallback poi网格名称自定义点击事件回调 默认null不处理
   * @param deleteGlobalData haveAddedJson搭配使用
   * @param preventWindows 是否阻止点击显示弹窗
   */
  async GetMapGridList({state, commit}, {
    isDelBefore = true, gistype1 = '', gistype2 = '', gisname = '',
    gridType = 1, is_community = 1, distance = 16000, girdList = null,
    sub_gis_flag = false, poiClickCallback = null, deleteGlobalData = true, preventWindows = true
  }) {
    // LOG.info(['isShowWindow',gistype1,gistype2,isShowWindow)
    if (isDelBefore) {
      await deleteEntityByGroup(mapService.client, '/MEntity/' + SuffixForOverlayId.polygon, deleteGlobalData)
      await deleteEntityByGroup(mapService.client, '/MEntity/' + SuffixForOverlayId.poiGird, deleteGlobalData)
    }

    const doRenderGirds = async (girdList) => {
      for (let i = 0; i < girdList.length; i++) {
        let resultKey = girdList[i]

        const color = resultKey.colour || getRandomColor()
        const id = SuffixForOverlayId.polygon + resultKey.id
        const polygon = await addGridPolygon(mapService.client, resultKey.gisstr_wgs, id, color,
          resultKey.gisname, false,
          //handle click event
          async (id) => {
            const cameraInfo = await getCameraInfo(mapService.client);
            const height = cameraInfo.position[2]
            if (preventWindows && height < miniWorldHeight) {
              LOG.warn(['height<miniWorldHeight 阻止点击事件'])
              return
            }

            const trueId = id.substring(SuffixForOverlayId.length)
            const data = girdList.find(value => value.id === trueId)
            try {
              const model = await mapService.client.getEntityById(id);
              // model.focus(400000, -Math.PI / 6)
              preventWindows && cameraFocusCoordinate(mapService.client, data.longitude * 1 + 0.004, data.latitude)

              let queryData = {gistype2: data.gistype2, gisname: data.gisname}
              let url = getLocalServiceUrl() + '/#/lqld-wangge?obj='
                + JSON.stringify(queryData) + '&token=' + sessionStorage.getItem('LoginToken')
              if (gridType === 3) {
                queryData = {gisname: data.gisname}
                url = getLocalServiceUrl() + '/#/dj-wangge2?obj='
                  + JSON.stringify(queryData) + '&token=' + sessionStorage.getItem('LoginToken')
                setWebWidget(model, url, true, 600, 400)
              } else {
                setWebWidget(model, url)
              }
            } catch (e) {
              LOG.warn([e])
            }
          }, preventWindows ? 8000 : 800)

        // console.log(polygon)

        commit("UpdateHaveAddedEntity", {
          key: SuffixForOverlayId.polygon, value: polygon
        })

        let poiId = SuffixForOverlayId.poiGird + resultKey.id
        await addPOI(mapService.client, poiId, resultKey.gisname,
          'POI圆形标签', [Number(resultKey.longitude), Number(resultKey.latitude), 0]
          , POIIcon.SolarEnergy, color, false, poiClickCallback)
        if (i === girdList.length - 1) {
          await focusEntityById(mapService.client, poiId, distance)
        }
      }
    }

    if (girdList) {
      await doRenderGirds(girdList)
    } else {
      getMapGridList({gistype2, gisname, sub_gis_flag, is_community}).then(async res => {
        const polygon = res.result || []
        await doRenderGirds(polygon)
        //具体请看components/GeneralDialogGroup
        state.haveAddedJson = {type: SuffixForOverlayId.polygon, value: res.result}
      })
    }
  },

  /**
   * 散点 gis api 下的点位信息
   * @param state
   * @param commit
   * @param isDelBefore  是否删除上一次地图构建的实体数据
   * @param gistype1
   * @param gistype2
   * @param gisname
   * @param icon 图标
   * @param isFocus
   * @return {Promise<void>}
   * @constructor
   */
  async GetGisApi({state, commit}, {
    isDelBefore = true, gistype1 = '',
    gistype2 = '', gisname = '', icon = null, isFocus = false
  }) {
    // LOG.info(['isShowWindow',gistype1,gistype2,isShowWindow)
    let preStr = ''
    switch (gistype2) {
      case '巡检点位':
        preStr = SuffixForOverlayId.poiXjdw;
        break
      case '文明实践站站点':
        preStr = SuffixForOverlayId.poiCivilizationPractice;
        break
      case '廉洁文化宣传阵地':
        preStr = SuffixForOverlayId.poiHonestCulturePropagandaPositions;
        break
    }
    if (isDelBefore) {
      await deleteEntityByGroup(mapService.client, '/MEntity/' + preStr)
    }
    getMapGridList({gistype2, gisname}).then(async res => {
      const poiIdList = [], poiList = res.result || []
      for (let i = 0; i < poiList.length; i++) {
        let resultKey = poiList[i]

        const color = resultKey.colour || getRandomColor()
        const poiId = preStr + resultKey.id
        const coordList = resultKey.gisstr_wgs.split(',')
        if (resultKey.gistype != 3) {
          LOG.info(['过滤非poi', resultKey])
          continue
        }
        if (coordList == null || coordList.length !== 2) {
          LOG.info(['过滤不正常的经纬度', resultKey])
          continue
        }
        resultKey.longitude = coordList[0]
        resultKey.latitude = coordList[1]
        try {
          await addPOI(mapService.client, poiId, resultKey.gisname,
            'POI圆形标签', [Number(resultKey.longitude), Number(resultKey.latitude), 0]
            , icon || POIIcon.Location, color, isFocus, async (id) => {
              const type = id.substring(0, SuffixForOverlayId.length)
              const trueId = id.substring(SuffixForOverlayId.length)
              switch (type) {
                case SuffixForOverlayId.poiCivilizationPractice:
                  try {
                    const model = await mapService.client.getEntityById(id);
                    model.focus(50000, -Math.PI / 6)
                    let queryData = {gis_id: trueId}
                    let url = getLocalServiceUrl() + '/#/civilization-practice?obj='
                      + JSON.stringify(queryData) + '&token=' + sessionStorage.getItem('LoginToken')
                    setWebWidget(model, url, true, 600, 400)
                    LOG.info(['addGridPolygon setWebWidget url=', url])
                  } catch (e) {
                    LOG.warn([e])
                  }
                  break
                case SuffixForOverlayId.poiHonestCulturePropagandaPositions:
                  try {
                    const model = await mapService.client.getEntityById(id);
                    model.focus(50000, -Math.PI / 6)
                    let queryData = {gis_id: trueId}
                    let url = getLocalServiceUrl() + '/#/ljwh?obj='
                      + JSON.stringify(queryData) + '&token=' + sessionStorage.getItem('LoginToken')
                    setWebWidget(model, url, true, 400, 300, false)
                    LOG.info(['addGridPolygon setWebWidget url=', url])
                  } catch (e) {
                    LOG.warn([e])
                  }
                  break

              }
            })
          if (isFocus) {
            await focusEntityById(mapService.client, poiId, 1000)
          }
          poiIdList.push(poiId)
        } catch (e) {
          LOG.warn([e])
        }

      }
      //添加 聚合
      await addPOICluster(mapService.client, preStr + 'poiCluster', poiIdList)
    })
  },

  //巡检点位撒点
  async XjdwPoiRender({state, commit}) {
    // LOG.info(['isShowWindow',gistype1,gistype2,isShowWindow)
    let preStr = SuffixForOverlayId.poiXjdw
    await deleteEntityByGroup(mapService.client, '/MEntity/' + preStr)
    let poiIdList = [], poiList = []
    //点位类别：、、、、、、、、、、、、、、、
    //get all data by xjdwType
    for (let type of XjdwTypeColor) {
      let {result} = await getMapGridList({gistype2: type.name})
      result = result || []
      result.map(e => {
        e.XjdwType = type;
        return e
      })
      poiList = poiList.concat(result)
    }

    for (let i = 0; i < poiList.length; i++) {
      let resultKey = poiList[i]
      const color = resultKey.XjdwType.color
      const poiId = preStr + resultKey.id
      const coordList = resultKey.gisstr_wgs.split(',')
      if (resultKey.gistype != 3) {
        LOG.info(['过滤非poi', resultKey])
        continue
      }
      if (coordList == null || coordList.length !== 2) {
        LOG.info(['过滤不正常的经纬度', resultKey])
        continue
      }
      resultKey.longitude = coordList[0]
      resultKey.latitude = coordList[1]
      try {
        await addPOI(mapService.client, poiId, resultKey.gisname,
          'POI圆形标签', [Number(resultKey.longitude), Number(resultKey.latitude), 0]
          , POIIcon.Location, color, false)
        poiIdList.push(poiId)
      } catch (e) {
        LOG.warn([e])
      }
    }

    state.haveAddedJson = {type: preStr, poiList}
    //添加 聚合
    await addPOICluster(mapService.client, preStr + 'poiCluster', poiIdList)
  },

  /**
   * <em>全局事件注册</em>
   * 设置距离调整网格高度  一标三实  建筑详情
   */
  InitMapListener({state, commit}) {
    //注册全局滚轮事件--设置距离调整网格高度
    wheelEventListener.push(async (eventData) => {
      const gridList = state.haveAddedEntity.get(SuffixForOverlayId.polygon)
      // LOG.info(['cameraInfo', gridList)
      if (gridList == null || gridList.length === 0) {
        return
      }
      const cameraInfo = await getCameraInfo(mapService.client)
      LOG.info(['cameraInfo', cameraInfo])
      const height = cameraInfo.position[2]
      for (const polygon of gridList) {
        let toHeight = height < miniWorldHeight ? 800 : 8000
        if (polygon.height !== toHeight) {
          polygon.height = toHeight
          try {
            await polygon.update();
          } catch (e) {
            LOG.warn([e])
          }
        }
      }
    })
    //注册全局点击事件--一标三实 建筑详情 现状未来
    clickEventListener.push(async (eventData) => {
      // LOG.info([eventData)
      //id
      const entityId = eventData.detail.entityId

      const cameraInfo = await getCameraInfo(mapService.client)
      const height = cameraInfo.position[2]
      //是否比阈值高度要小
      const isLessThanMiniWorldHeight = height < miniWorldHeight
      //一标三实
      const ybss = async () => {
        if (isLessThanMiniWorldHeight) {
          const model = await mapService.client.getEntityById(entityId);
          let url = getLocalServiceUrl() + '/#/ybss?obj=' + JSON.stringify({modelId: entityId})
            + '&token=' + sessionStorage.getItem('LoginToken')
          setWebWidget(model, url, true, 600, 400, false)
        }
      }

      //建筑详情
      const jzxq = async () => {
        const model = await mapService.client.getEntityById(entityId);
        let url = getLocalServiceUrl() + '/#/construction-detail?obj=' +
          JSON.stringify({modelId: entityId})
          + '&token=' + sessionStorage.getItem('LoginToken')
        setWebWidget(model, url, true, 330, 480, false)
      }

      //现状未来
      const nameObj = getGroupNameObjById(entityId);
      const xzwl = async () => {
        if (nameObj) {
          // await clearWindow(mapService.client, state.currentWindowId)
          const model = await mapService.client.getEntityById(entityId);
          commit('SetCurrentWindowId', entityId)
          model.focus(20000, -Math.PI / 4)
          setWebWidget(model, getLocalServiceUrl() +
            '/#/now-with-future?nameObj=' + JSON.stringify(nameObj) + '&entityId=' + entityId,
            false, 180, 155)
        }
      }
      //小于高度
      if (isLessThanMiniWorldHeight) {
        xzwl()
      }

      //获取绑定信息
      getBuildInfo({model_id: entityId}).then(async res => {
        const resData = (res.result || [])[0] || {}
        switch (resData.gistype2) {
          case '社区':
            await ybss()
            break
          case '地标':
            // await setEntityRenderMode(mapService.client, entityId, RenderMode.Highlight)
            //如果大于阈值高度或者无现状未来显示
            if (!isLessThanMiniWorldHeight || !nameObj) {
              await jzxq()
            }
            break
          default:
            // await ybss()
            LOG.info([entityId + '没有绑定点击事件'])
        }
      })

      PoiBindInfo(entityId)
    })

    //根据id 获取绑定poi操作
    const PoiBindInfo = async (entityId) => {
      //设置poi id
      state.poiClickId = entityId

      if (!entityId) {
        return
      }

      const type = entityId.substring(0, SuffixForOverlayId.length)
      const tureId = entityId.substring(SuffixForOverlayId.length)

      switch (type) {
        case SuffixForOverlayId.poiDevice:
          const poiDevice = state.poiDeviceMap.get(entityId)
          if (poiDevice) {
            commit('SetPoiDeviceShow', poiDevice)
          }
          break
        case SuffixForOverlayId.poiXjdw:
          focusEntityById(mapService.client, entityId)
          const model = await mapService.client.getEntityById(entityId);
          let url = getLocalServiceUrl() + '/#/xjdw?obj=' + JSON.stringify({modelId: entityId})
            + '&token=' + sessionStorage.getItem('LoginToken')
          setWebWidget(model, url, true, 400, 300, false)
          break
      }
    }
  },
  /**
   * 清运车移动轨迹
   * @param commit
   * @param licenseplate api传参
   * @returns {Promise<void>}
   * @constructor
   */
  async GetCarTrackList({commit}, licenseplate = '沪A-DF88675') {
    await deleteEntityByGroup(mapService.client, '/MEntity/' + SuffixForOverlayId.polyline)

    const coordinateList = []

    getCarTrackList({licenseplate}).then(async res => {
      //反转 变成开始--->终点
      const dataList = (res.result || []).reverse()
      for (let i = 0; i < dataList.length; i++) {
        let data = dataList[i]
        const getCoord = (data) => {
          return [Number(data.longitude), Number(data.latitude), 5]
        }
        coordinateList.push(getCoord(data))
      }
      await addPolyline(mapService.client, SuffixForOverlayId.polyline + licenseplate + 1,
        '#00A2F1', coordinateList)
      //修改高度
      coordinateList.map(e => {
        e[2] = 4;
        return e;
      })
      await addPolyline(mapService.client, SuffixForOverlayId.polyline + licenseplate + 2,
        '#00ffe0', coordinateList, '自发光', 0.5)
      const carId = SuffixForOverlayId.polyline + licenseplate + "Car"
      await addPolylineAnimal(mapService.client, carId, '#ff0000', coordinateList, 50)
      LOG.warn(['开始聚焦'])
      await focusEntityById(mapService.client, carId, 5000)
      setTimeout(async () => {
        await focusEntityById(mapService.client, carId, 1000)
      }, 2000)

    })

  },

  /**
   * 清除所有覆盖物
   * @param commit
   * @returns {Promise<void>}
   * @constructor
   */
  async ClearAllEntity({commit}) {
    state.haveAddedEntity = new Map()
    await deleteEntityByGroup(mapService.client, '/MEntity')
  },
  //单兵 定位
  async GetIntelligentDevicesPosition({commit}, item) {
    LOG.info(['item', item])
    if (item == null || !item.latitude || !item.longitude) {
      return
    }
    const id = SuffixForOverlayId.poiDevice + item.id
    const success = await focusEntityById(mapService.client, id)
    if (!success) {
      const style = {
        icon: POIIcon.AlarmStation,
        color: item.device_status == 1 ? '#ff1e1e' : '#d2cfcf',
      }
      const data = item
      await addPOI(mapService.client, SuffixForOverlayId.poiDevice + data.id,
        data.device_name, 'POI圆形标签',
        [Number(data.longitude), Number(data.latitude), 0],
        style.icon, style.color,)
      await focusEntityById(mapService.client, id)
    }

  },
  //巡检记录点击
  async GetInspectionRouter({commit}, id) {
    await deleteEntityByGroup(mapService.client, '/MEntity/' + SuffixForOverlayId.inspectionRoute)

    // LOG.info(['id',id)
    const coordinateList = []

    getInspectionGis({id}).then(async res => {
      const dataList = (res.result || [])
      for (let i = 0; i < dataList.length; i++) {
        let data = dataList[i]
        const getCoord = (data) => {
          return [Number(data.longitude), Number(data.latitude), 5]
        }
        coordinateList.push(getCoord(data))
      }
      await addPolyline(mapService.client, SuffixForOverlayId.inspectionRoute + id,
        '#f10008', coordinateList)
      await focusEntityById(mapService.client, SuffixForOverlayId.inspectionRoute + id
        , 500000)
    })
  },
  /**
   * 管道显示
   * @param commit
   * @param id 实体id
   * @param isVisible 是否显示
   * @param isFocus 是否聚焦
   * @param focusId 自定义聚焦id 否则默认 id参数
   * @return {Promise<void>}
   * @constructor
   */
  async SetEntityVisible({commit}, {
    id = '5806879D4C0F9F4718655F927D693EFD', isVisible = true, isFocus = true
    , focusId = '06BE557E4DDFC221D0A11D8374DAE4BF'
  }) {
    try {
      const model = await mapService.client.getEntityById(id);
      if (isFocus) {
        const modelFocus = focusId ? await mapService.client.getEntityById(focusId) : model;
        modelFocus.focus(20000, -Math.PI / 3)

      }
      model.setVisible(isVisible)
    } catch (e) {
      LOG.warn([e])
    }
  },
  //暂时无用
  async SetDynamicPathVisible({commit}, {pathIdList = [0, 1, 2, 3], isVisible = false}) {
    for (let i = 0; i < pathIdList.length; i++) {
      const data = pathIdList[i]
      await mapService.client.setPathVisibility(data, isVisible);
    }
  }
}

export default {
  namespaced: true,
  state,
  mutations,
  actions
}
