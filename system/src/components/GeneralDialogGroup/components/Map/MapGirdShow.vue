<template>
  <div>
    <div class="dialog-modal" v-if="isVisible && modal"></div>
    <transition name="dialog">
      <div class="m-map-main" v-if="isVisible && baseData">
        <dv-border-box-12 :key="showWhat">
          <div class="m-map-main-inner">
            <div class="dialog-head">
              <div class="dialog-head-title">{{ baseData.gisname }}</div>
              <div style="display: flex">
                <custom-button type="warn" v-if="showWhat!=='gird'" style="height: 20px;width: 60px; margin-right: 8px;"
                               @click.native="showWhat='gird'">返回</custom-button>
                <cross-sign @click.native="closeThis"/>
              </div>
            </div>
            <div class="dialog-body" v-if="showWhat==='gird'">
              <auto-horizontal-scroll>
                <div class="gird-map-item" v-for="(item,i) in baseData.sub_gis" :key="i"
                @click="showGirdDetail(item)">
                  <div class="gird-map-item-inner">
                    <div class="point" :style="{background:item.colour||'#fff'}"></div>
                    <div class="gird-name" :title="item.gisname">{{ item.gisname }}</div>
                  </div>
                </div>
              </auto-horizontal-scroll>
            </div>
            <div class="dialog-body" v-else style="height: 400px">
              <div class="grid-inner">
                <service-station style="width: 100%; height: 100%" :obj-data="chooseItem"/>
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
import CrossSign from "@/components/CrossSign";
import UnitNumber from '@/components/UnitNumber'
import SpotLabel from '@/components/SpotLabel'
import AutoHorizontalScroll from "@/components/AutoHorizontalScroll.vue";
import ServiceStation from "@/views/window-poi/service-station/index.vue";
import CustomButton from "@/components/custom-button.vue";

export default {
  name: "MapGirdShow",
  components: {CustomButton, ServiceStation, AutoHorizontalScroll, CrossSign, UnitNumber, SpotLabel},
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
    isVisible(val) {
      if (val) {
        this.showWhat='gird'
      }
    }
  },
  data() {
    return {
      //gird or girdDetail
      showWhat:'gird',
      chooseItem:null
    }
  },
  mounted() {
    this.appendToBody()
  },
  methods: {
    closeThis() {
      this.isVisible = false
    },
    showGirdDetail(item){
      this.chooseItem=item
      this.showWhat='girdDetail'
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
  width: 600px;
  left: calc(50% - 300px);
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
    }
  }

}

.gird-map-item {
  display: inline-block;
  margin-left: 16px;
  height: 100%;
  width: 145px;
  border-radius: 8px;
  color: white;
  cursor: pointer;
  background-image: linear-gradient(135deg, #5b247a, #1bcedf);

  .gird-map-item-inner{
    height: 100%;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 4px;

    .point {
      display: inline-block;
      background: #FFFFFF;
      border-radius: 100%;
      height: 18px;
      width: 18px;
      margin:0 5px;
    }

    .gird-name {
      flex: 1;
      overflow-x: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;//禁止换行
      display: inline-block;
      line-height: 32px;
      font-size: 16px;
      font-weight: 500;
    }
  }

}

.gird-map-item:hover{
  border: rgba(255, 255, 255) 1px solid;
}


.grid-inner {
  height: 100%;
  width: 100%;
  padding: 8px;

  .grid-inner-top {
    width: 100%;
    height: 24px;
    padding: 0 5px;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    color: #ffffff;

    .title {
      font-family: YouSheBiaoTiHei;
      font-size: 20px;
    }

    .grid {
      font-size: 14px;
      margin-bottom: 3px;
      margin-right: 30px;
    }
  }


  .grid-inner-bottom {
    width: 100%;
    height: calc(100% - 36px);
    padding: 5px;

    .user-info {
      width: 100%;
      height: 80px;
      display: flex;

      .usin-item {
        width: 50%;
        height: 100%;
        display: flex;

        .header {
          width: 70px;
          height: 100%;
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;

          .back {
            z-index: 0;
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
          }

          .icon {
            position: relative;
            z-index: 1;
            width: 30px;
            height: auto;
          }

          .h-img {
            position: relative;
            z-index: 1;
            width: 64px;
            height: 74px;
          }
        }

        .info {
          flex: 1;
          margin-left: 8px;

          .name {
            font-size: 22px;
            line-height: 1;
            font-family: YouSheBiaoTiHei;
            color: #46FEFE;
          }

          .role {
            margin: 8px 0;
            display: flex;
            align-items: center;

            .icon {
              height: 10px;
              width: auto;
            }

            .text {
              margin: 0 5px;
              font-size: 11px;
              color: #ffffff;
              line-height: 14px;
            }
          }

          .action {
            display: flex;
            align-items: center;
            margin-top: 10px;

            .icon {
              width: 23px;
              height: 23px;
              margin-right: 10px;
              cursor: pointer;
            }
          }
        }
      }
    }

    .grid3-main{
      width: 100%;
      height: 54px;
      padding: 8px;
      align-items: center;
      border-radius: 8px;
      margin-top: 8px;
      background: #00000020;

      overflow: hidden;

      .cell{
        display: inline-block;
        background: #00000042;
        //width: 100px
        text-align: center;
        padding: 8px 15px;
        border-radius: 8px;
        margin-left: 8px;
      }
    }

    .user-list-view {
      margin-top: 8px;
      height: calc(100% - 80px);
      width: 100%;
      padding: 8px;
      background: #00000042;
    }

    .user-list {
      display: flex;
      flex-wrap: wrap;

      .list-line {
        margin-right: 20px;
        width: calc(50% - 20px);
        display: flex;
        align-items: center;
        padding: 4px;
        border-bottom: 1px dashed #0a5dc9a8;

        .line-item {
          flex: 1;
          font-size: 11px;
          color: #ffffff;
          overflow: hidden;
          word-break: keep-all;
          text-overflow: ellipsis;
        }

        .line-icon {
          flex-shrink: 0;
          width: 10px;
          height: auto;
          margin-right: 4px;
        }

      }
    }
  }
}
</style>
