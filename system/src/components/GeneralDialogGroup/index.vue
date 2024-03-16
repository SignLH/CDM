<!--
*@general-dialog-group 通用弹窗组
*@author yuge
*@date 2023/7/7 15:34
-->
<template>
  <div class="general-dialog-group-main">
    <!--    监控弹窗-->
    <monitor-dialog v-model="monitorWinShow" :title="monitorWinTitle" :entity="monitorWinChoose"/>

    <!--    门磁-->
    <magnetic-door-dialog v-model="isShowDialogMagneticDoor" :entity="chooseEntity"/>

    <!--    火警报警器-->
    <fire-alarm-dialog v-model="isShowDialogFireAlarm" :entity="chooseEntity"/>

    <!--    地图浏览器弹窗-->
    <map-gird-show v-model="isShowMapGird" :base-data="chooseEntity"/>
    <map-xjdw-show v-model="isShowMapXjdw" :base-data="chooseEntity"/>
    <map-duty-gird v-model="isShowDutyGird" :base-data="chooseEntity"/>
  </div>
</template>

<script>
import {mapGetters} from 'vuex'
import MonitorDialog from "./components/MonitorDialog.vue";
import MagneticDoorDialog from "./components/MagneticDoorDialog.vue";
import {focusEntityById, SuffixForOverlayId} from "@/utils/services/overlay/entity";
import {mapService} from "@/utils/app-base";
import FireAlarmDialog from "./components/FireAlarmDialog.vue";
import MapGirdShow from "./components/Map/MapGirdShow.vue";
import MapXjdwShow from "./components/Map/MapXjdwShow.vue";
import MapDutyGird from "@/components/GeneralDialogGroup/components/Map/MapDutyGird.vue";

export default {
  name: "general-dialog-group",
  props: {},
  components: {MapDutyGird, MapXjdwShow, MapGirdShow, FireAlarmDialog, MagneticDoorDialog, MonitorDialog},
  data() {
    return {
      //重点监控
      monitorWinShow: false,
      monitorWinTitle: '重点监控',
      monitorWinChoose: null,

      //智能设备 弹窗
      isShowDialogMagneticDoor: false,
      isShowDialogFireAlarm: false,

      //map handler view
      isShowMapGird: false,
      isShowMapXjdw: false,
      isShowDutyGird: false,

      chooseEntity: null,
    }
  },
  computed: {
    ...mapGetters([
      'monitorVideoUrl',
      'poiDeviceShow',
      'haveAddedJson',
    ]),
  },
  watch: {
    monitorVideoUrl(val) {
      if (val != null && val.video_address) {
        //开启重点监控
        this.openMonitorWinTitle({from: val.from ? val.from : 'dialog', ...val})
      }
    },
    poiDeviceShow(v) {
      if (v) {
        let id = SuffixForOverlayId.poiDevice + v.id
        focusEntityById(mapService.client, id)
        switch (v.device_type) {
          case '门磁':
            this.chooseEntity = v
            this.isShowDialogMagneticDoor = true
            break
          case '火警报警器':
            this.chooseEntity = v
            this.isShowDialogFireAlarm = true
            break
        }
      }
    },
    haveAddedJson: {
      handler(newV) {
        // is null, close all,need add if add new dialog
        if (!newV) {
          this.isShowMapGird = false
          this.isShowMapXjdw = false
          this.isShowDutyGird = false
          return
        }
        switch (newV.type) {
          //责任网格
          case SuffixForOverlayId.polygon:
            const polygon = newV.value
            //judgement logic: 只有一个网格是责任网格
            if (polygon && polygon.length === 1 && polygon[0].gistype2 === '责任网格') {
              this.chooseEntity = polygon[0]
              this.isShowDutyGird = true
            } else {
              this.chooseEntity = null
              this.isShowMapGird = false
            }
            break

          case SuffixForOverlayId.poiXjdw:
            //judgement logic: 只有一个网格并且存在子网格
            this.chooseEntity = newV.value
            this.isShowMapXjdw = true
            break
        }
      },
      deep: true
    }
  },
  mounted() {
  },
  methods: {
    //开启重点监控
    openMonitorWinTitle(item) {
      this.monitorWinShow = true;
      this.monitorWinTitle = item.device_name || '重点监控'
      this.monitorWinChoose = item
    },
  },
  destroyed() {
  }
}
</script>

<style scoped lang="scss">
.general-dialog-group-main {
  height: 0;
  width: 0;
}
</style>
