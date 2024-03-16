<template>
  <div>
    <div class="dialog-modal" v-if="isVisible && modal"></div>
    <transition name="dialog">
      <div class="m-map-main" v-if="isVisible && baseData">
        <dv-border-box-12>
          <div class="m-map-main-inner">
            <div class="dialog-head">
              <div class="dialog-head-title">{{ baseData.gisname }}</div>
              <cross-sign @click.native="closeThis"/>
            </div>
            <div class="dialog-body">
              <div class="gird-map-btn" :class="{'gird-map-btn-selected':chooseLevel===1}"
                   @click="showGird(1)">责任网格
              </div>
              <div class="gird-map-btn" :class="{'gird-map-btn-selected':chooseLevel===2}"
                   @click="showGird(2)">二级网格
              </div>
              <div class="gird-map-btn" :class="{'gird-map-btn-selected':chooseLevel===3}"
                   @click="showGird(3)">微网格
              </div>
              <div v-if="!!chooseEntity" class="gird-map-btn-tip" @dblclick="chooseEntity=null"
              title="双击清空选择">
                {{ chooseEntity.gisname }}
              </div>
            </div>
          </div>
        </dv-border-box-12>
      </div>
    </transition>
  </div>
</template>

<script>
import visible from "@/mixins/edit-dialog"
import {mapGetters} from 'vuex'
import CrossSign from "@/components/CrossSign/index.vue";
import {SuffixForOverlayId} from "@/utils/services/overlay/entity";
import LOG from "@/utils/self-log";

export default {
  name: "MapDutyGird",
  components: {CrossSign},
  props: {
    modal: {
      type: Boolean,
      default: false,
    },
    baseData: {
      type: Object,
      default: () => ({})
    }
  },
  mixins: [visible],
  watch: {
    baseData(v){
      this.chooseEntity=null
      this.chooseLevel=1
    }
  },
  data() {
    return {
      chooseLevel: 1,
      chooseEntity: null,
    }
  },
  mounted() {
    this.appendToBody()
  },
  methods: {
    closeThis() {
      this.isVisible = false
    },
    showGird(level) {
      this.chooseLevel = level
      const item = this.baseData || {}
      const gird2List = item.sub_gis || []
      switch (level) {
        case 1:
          this.$store.dispatch('map/GetMapGridList', {
            girdList: [item],
            deleteGlobalData: false,
            distance: 3000
          })
          break
        case 2:
          this.$store.dispatch('map/GetMapGridList', {
            girdList: gird2List,
            deleteGlobalData: false,
            distance: 3000,
            poiClickCallback: this.handlerGirdLevel2Click
          })
          break
        case 3:
          let showGird3List = []
          if (this.chooseEntity) {
            showGird3List = this.chooseEntity.sub_gis
          } else {
            for (let e of gird2List) {
              const gird3s = e.sub_gis || []
              showGird3List = showGird3List.concat(gird3s)
            }
          }
          this.$store.dispatch('map/GetMapGridList', {
            girdList: showGird3List,
            deleteGlobalData: false,
            distance: 170,
            preventWindows:false
          })
          break
      }
    },
    handlerGirdLevel2Click(id) {
      LOG.info(['点击二级网格 --id ' + id, !!id])
      const item = this.baseData || {}
      const gird2List = item.sub_gis || []
      let chooseGird3Id = id.substring(SuffixForOverlayId.length)
      for (let e of gird2List) {
        if (e.id === chooseGird3Id) {
          if (e.gistype2.indexOf('二级网格') !== -1) {
            this.chooseEntity = e
          }
          break
        }
      }
    },
    /**
     * 添加组件到body
     */
    appendToBody() {
      this.$nextTick(() => {
        const body = document.querySelector("body");
        if (body.append) {
          body.append(this.$el);
        } else {
          body.appendChild(this.$el);
        }
      });
    },
  },
}
</script>

<style lang='scss' scoped>
/* 设置持续时间和动画函数 */
.dialog-enter-active,
.dialog-leave-active {
  transition: all .3s ease;
}

.dialog-enter,
.dialog-leave-to {
  transform: translateY(-150px);
  opacity: 0;
}

.dialog-modal {
  position: fixed;
  top: 0;
  right: 0;
  left: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  z-index: 9998;
}

.m-map-main {
  position: fixed;
  top: 9%;
  width: 500px;
  left: calc(50% - 250px);
  box-sizing: border-box;
  z-index: 9999;
  border-radius: 16px;

  .m-map-main-inner {
    left: calc(50% - 300px);
    padding: 16px;
    box-sizing: border-box;

    .dialog-head {
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 10px;

      .dialog-head-title {
        font-size: 25px;
        font-family: YouSheBiaoTiHei;
        color: #ffffff;
      }
    }

    .dialog-body {
      width: 100%;
      height: 40px;
      margin-bottom: 16px;
      display: flex;
      justify-content: space-around;
      position: relative;
    }
  }

}

.gird-map-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 145px;
  border-radius: 8px;
  color: white;
  cursor: pointer;
  background-image: linear-gradient(135deg, #5b247a, #1bcedf);

}

.gird-map-btn-tip {
  position: absolute;
  bottom: -20px;
  right: 0;
  color: white;
  cursor: pointer;
  font-size: 12px;
  text-align: center;
}

.gird-map-btn:hover, .gird-map-btn-selected {
  border: rgba(255, 255, 255, .7) 1px solid;
}
</style>
