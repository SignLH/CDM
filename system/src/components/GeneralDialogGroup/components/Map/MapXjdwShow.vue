<template>
  <div>
    <div class="dialog-modal" v-if="isVisible && modal"></div>
    <transition name="dialog">
      <div class="m-map-main" v-if="isVisible && baseData">
        <dv-border-box-12>
          <div class="m-map-main-inner">
            <div class="dialog-head">
              <div class="dialog-head-title">巡检点位</div>
              <cross-sign @click.native="closeThis"/>
            </div>
            <div class="dialog-body">
              <auto-horizontal-scroll>
                <div class="xjdw-map-item" v-for="(item,i) in XjdwTypeColor()" :key="i">
                  <div class="xjdw-map-item-inner">
                    <div class="point" :style="{background:item.color||'#fff'}"></div>
                    <div class="gird-name" :title="item.name">{{ item.name }}</div>
                  </div>
                </div>
              </auto-horizontal-scroll>
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
import {XjdwTypeColor} from "@/utils/color";

export default {
  name: "MapXjdwShow",
  components: {AutoHorizontalScroll, CrossSign, UnitNumber, SpotLabel},
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
    // isVisible(val) {
    //   if (val) {
    //     console.log(this.baseData)
    //   }
    // }
  },
  data() {
    return {}
  },
  mounted() {
    this.appendToBody()
  },
  methods: {
    XjdwTypeColor() {
      return XjdwTypeColor
    },
    closeThis() {
      this.isVisible = false
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

.xjdw-map-item {
  display: inline-block;
  margin-left: 16px;
  height: 100%;
  width: 155px;
  border-radius: 8px;
  color: white;
  cursor: pointer;
  background-image: linear-gradient(135deg, #5b247a, #1bcedf);

  .xjdw-map-item-inner{
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
</style>
