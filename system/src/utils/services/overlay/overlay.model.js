import {v4} from 'uuid';

//场景

//场景
export const SceneNameAssemble = {
  /* 施工五福院、建成五福院、施工综合服务中心、建成综合服务中心、

  白天-入口、鸟瞰一、鸟瞰二、鸟瞰三、绿色东平、镇政府、五福院、综合服务中心、
  党群服务中心、社区事务所、东平市人民公园、农工商超市、长江小学、长江幼儿园、
  德科公司、东平养老院、市容环境事务所、社区服务中心、消防救援站、汽车站、瀛都新村、
  瀛丰嘉苑、长晶新村、明华一村、明华二村、桂林新村、长江学校、垃圾处理站、德科公司、
  民远技术学院、 民远技术学院2、
  N、E、W、S、夜景-入口、绿色东平-夜
   */
  //灯光组名
  light:'夜景',
  morning: {"pitch":-24.403438568115234,"yaw":-43.65771484375,"coordinate":[121.53153359091996,31.672266011142785,150.99875571104744],"duration":2},
  noon: {"pitch":-35.856201171875,"yaw":128.5750274658203,"coordinate":[121.53507904409508,31.67889682923468,143.45849524864687],"duration":2},
  evening: {"pitch":-24.403554916381836,"yaw":-43.658409118652344,"coordinate":[121.53153359091996,31.672266011142785,150.99875571104744],"duration":2},
  birdView1: {"pitch":-31.411806106567383,"yaw":-96.81289672851562,"coordinate":[121.5360625770132,31.667644888997984,322.28876684427047],"duration":2},
  birdView2: {"pitch":-38.26921844482422,"yaw":-32.58132553100586,"coordinate":[121.52843283286848,31.672513576862777,451.4282460946213],"duration":2},
  birdView3: {"pitch":-26.203487396240234,"yaw":-13.306812286376953,"coordinate":[121.5286481679902,31.676722449390677,208.90152125495155],"duration":2},
  default: {"pitch":-27.128997802734375,"yaw":-94.60236358642578,"coordinate":[121.53852552876242,31.66082447140575,749.9365748281772],"duration":2},
  //智慧党建
  ZHDJ:{"pitch":-35.782257080078125,"yaw":-172.88827514648438,"coordinate":[121.53876018885414,31.67219456476722,482.7396212835672],"duration":2},
  //重点监护
  ZDJH:{"pitch":-32.96209716796875,"yaw":137.75062561035156,"coordinate":[121.5386467344494,31.6814896878904,482.445513748284],"duration":2},
  //数治戎耀
  SZRY:{"pitch":-20.04738998413086,"yaw":-173.31838989257812,"coordinate":[121.5362941247123,31.676967103825348,103.65955464951435],"duration":2},

  N: {"pitch":-27.363218307495117,"yaw":-87.61309814453125,"coordinate":[121.53426745447753,31.66588803387474,337.5477447018738],"duration":2},
  S: {"pitch":-26.50396728515625,"yaw":90.71827697753906,"coordinate":[121.53463162121894,31.683446312784298,333.46469872350923],"duration":2},
  W: {"pitch":-29.6038818359375,"yaw":-177.4909210205078,"coordinate":[121.54177317372223,31.67667753294489,329.96625194459466],"duration":2},
  E: {"pitch":-33.351470947265625,"yaw":6.138081073760986,"coordinate":[121.52775824232818,31.677563320963973,354.52405916816457],"duration":2},
  TownCenter:{"pitch":-32.96209716796875,"yaw":137.75074768066406,"coordinate":[121.53552214988838,31.67886144158553,169.08678432841202],"duration":2},
  TownGovernment:{"pitch":-29.027673721313477,"yaw":-97.5140380859375,"coordinate":[121.5371515049435,31.67631250753402,81.06037518243667],"duration":2},
  partyServiceCenter:{"pitch":-35.782283782958984,"yaw":-172.8883056640625,"coordinate":[121.53461044316842,31.672815886822193,140.02572073338314],"duration":2},
  // //消防站
  // XiaoFangZhan: '消防救援站',
  // //公交站
  // GongJiaoZhan: '汽车站',
  //五福院
  WuFuYuanNow: {"pitch":-24.90838623046875,"yaw":-48.42433166503906,"coordinate":[121.53502416297754,31.677729567939632,77.79555151643834],"duration":2},
  WuFuYuanNowFuture: '建成五福院',
  //综合服务
  ZhongHeFuWuZhongXinNow: {"pitch":-32.70248031616211,"yaw":-56.36267852783203,"coordinate":[121.53535743861967,31.676807029785955,160.3788549640045],"duration":2}  ,
  ZhongHeFuWuZhongXinFuture: '建成综合服务',
  //德科公司
  DeKeGongSiNow: {"pitch":-32.901580810546875,"yaw":-124.03532409667969,"coordinate":[121.53380334614529,31.673337429069573,153.9338114238408],"duration":2},
  DeKeGongSiFuture1: '民远技术学院',
  DeKeGongSiFuture2: '民远技术学院2',
}

/**
 * 返回对应id的地名现状未来 实体类显示或隐藏 通过按组显示和隐藏
 * @param id
 * @returns {{future2: string, future: string, now: string}|{future: string, now: string}|null}
 */
export function getGroupNameObjById(id) {
  const idWithSceneNameList = [
    //德科公司
    {
      sceneName: {now: '现状对比/德科公司', future: '现状对比/明远技术学院-01', future2: '现状对比/明远技术学院-02'},
      idList: [
        //now
        'E62D10D94005EFA1DFD8078089312FA8', 'F7867E2A4E9DE7310F70C6AE2E5F156C',
        'ACB5096E4C616F6621CED39E03756B6E', '7BCFE7F54613B835667CBCB0B7139F0B',
        //future1
        "123DC97846080BD2F2E154B967FB723D","17F987E24A43374C914B98BFBABF0B4A",
        "7208FB694919C267E671AF8A4D2641BA","92529EB94FD4F984DFAF01A07A423116",
        //future2
        'ECB557F64E283217BF8C6E937F00333B', '6208D5864BBF691779C201939CC4E4FA',
        '2CE5B1A14C2674CF7EA90A9AEC792796', '76EDF0444621ED82211AB19113329F15',
        '954A975F4643037B2C80B1897B393C5E',
      ]
    },
    //综合服务中心
    {
      sceneName: {now: '现状对比/施工效果/综合服务中心', future: '现状对比/建成效果/建成综合服务中心'},
      idList: [
        //now
        'D8D4CB284BFDC747C9D9F98E291478DF', 'EC3B63C34E900CEB7C55B9824F6F457B',
        '41899E49430588985CB9E7AFC5137D14', '7758940249798EA431C39D825DF2F853',
        //future
        '60B0E07049C037E566C7F1A3719E73C9',
      ]
    },
    //五福院
    {
      sceneName: {now: '现状对比/施工效果/五福院', future: '现状对比/建成效果/建成五福院'},
      idList: [
        //now
        '501B2B2A4258EB8CF199A4BD810E74C0', 'E37A26F1473AD2D439099E9FE32D304A',
        "29885AAE48BDFA085E14A3A36B088664",
        //future
        '0C1C6A2D44663B820DC5218017EFE660', 'D8D4CB284BFDC747C9D9F98E291478DF',
        'D3FF6C0640E156A8714302A2ED0A7104','07B5D7D545916E02E125A3A2A4B5DB6F',
        '7207F773419672275FDFBDAE1BFC2B66', '2AC341C44B27530A9A796E842CA81F44',
        '3F687C904EFC38B11697F89FBF978611', '3569E6AF467F4735A90BFC95D67C6B9A',
        'ECAA02994A5ED59579BC8BA3F2642BD0', 'F062917D4CEC57810076AEBEF1E4EE57',
      ]
    },
  ]
  for (let item of idWithSceneNameList) {
    const isFind = item.idList.findIndex(e => e === id) !== -1;
    if (isFind) {
      return {
        ...item.sceneName
      }
    }

  }
  return null
}

export const OverlayType = {
  POI: 'poi',
  Marker: 'Marker',
  Emissive: 'emissive',
}

//https://diva.sheencity.cn/docs/index.html#/sdk/rendermode?id=%e5%8f%a0%e8%89%b2%e9%ab%98%e4%ba%ae%e6%a8%a1%e5%bc%8f
export const RenderMode={
  Alarm: "alarm", //警报模式（红色闪烁）
  Default: "default",
  Emission: "emission", //自发光模式
  Hidden: "hidden",
  Highlight: "highlight",
  Translucence: "translucence", //半透明模式 （蓝色半透明）
}

export const POIIcon = {
  Camera: 'camera',
  Location: 'location',
  TrafficLight: 'trafficLight',
  TrashCan: 'trashCan',
  StreetLamp: 'streetLamp',
  BusStation: 'busStation',
  Exit: 'exit',
  Restaurant: 'restaurant',
  Parking: 'parking',
  Dock: 'dock',
  Subway: 'subway',
  Supermarket: 'supermarket',
  Mall: 'mall',
  Toilet: 'toilet',
  Building: 'building',
  AlarmStation: 'alarmStation',
  SolarEnergy: 'solarenergy'
}

export const POIIconType = {
  type1: 'POI文字标签',
  type2: 'POI圆形标签',
  type3: 'POI水滴',
}

export const EmissionType = {
  type1: '悬浮标记01',
  type2: '圆形区域轮廓02',
  type3: '雷达标记',
  type4: '地面标记01',
  type5: '圆形区域轮廓01',
  type6: '事故标记',
  type7: '悬浮标记02',
  type8: '圆形区域轮廓03',
}

export class Overlay {
  constructor() {
    this.corrdinateX = 0;
    this.corrdinateY = 0;
    this.corrdinateZ = 0;
    this.content = '';
    this.color = '';
    this.rotationX = 0;
    this.rotationY = 0;
    this.rotationZ = 0;
    this.scale = 1;
    this.opacity = 1;
    this.id = v4();
  }
}

export class POIOverlay extends Overlay {
  constructor() {
    // @ts-ignore
    super(...arguments);
    this.type = OverlayType.POI;
    this.icon = '';
    this.iconType = '';
  }
}

export class MarkerOverlay extends Overlay {
  constructor() {
    // @ts-ignore
    super(...arguments);
    this.type = OverlayType.Marker;
    this.title = '';
    this.align = '';
    this.borderWidth = 0;
    this.borderColor = '';
  }
}

export class EmissiveOverlay extends Overlay {
  constructor() {
    // @ts-ignore
    super(...arguments);
    this.type = OverlayType.Emissive;
    this.emission = 0;
    this.speed = 0;
    this.icon = '';
  }
}
