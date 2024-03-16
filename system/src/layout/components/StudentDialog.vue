<template>
  <div>
    <div class="dialog-modal" v-if="isVisible"></div>
    <transition name="dialog">
      <div class="stu-dialog-main" v-if="isVisible" :style="{ left: `calc(50% - ${width} / 2)`, width, top }">
        <dv-border-box-13>
          <div class="m-dialog-main-inner">
            <div class="dialog-head">
              <div class="dialog-head-title">{{ title }}</div>
              <cross-sign @click.native="isVisible=!isVisible"/>
            </div>
            <div class="dialog-body">
              <div class="upper-main">
                <div class="flex-around" >
                  <city-image-block type="contracted" title="已做答问题" value="23" unit="个"/>
                  <city-image-block type="contracted" title="未做答问题" value="27" unit="个"/>
                  <city-image-block type="contracted" title="回答正确" value="12" unit="个"/>
                  <city-image-block type="contracted" title="部分正确" value="5" unit="个"/>
                  <city-image-block type="contracted" title="回答错误" value="6" unit="个"/>
                  <city-image-block type="contracted" title="本题尝试次数" value="21" unit="次"/>
                  <city-image-block type="contracted" title="正确率" value="52" unit="%"/>
                  <city-image-block type="contracted" title="下一次作答准确率预估" value="49" unit="%"/>
                </div>
              </div>
              <div class="lower-main">
                <student-chart1 style="width: 49%"/>
                <div class="hr-main"></div>
                <student-chart2 style="width: 49%"/>
              </div>

            </div>
          </div>
        </dv-border-box-13>
      </div>
    </transition>
  </div>
</template>

<script>

import visible from "@/mixins/edit-dialog"
import CrossSign from "@/components/CrossSign";
import CityImageBlock from "@/components/CityImageBlock/index.vue";
import StudentChart1 from "@/layout/components/StudentChart1.vue";
import StudentChart2 from "@/layout/components/StudentChart2.vue";

export default {
  name: "StudentDialog",
  components: {StudentChart2, StudentChart1, CityImageBlock, CrossSign},
  computed: {},
  props: {
    title: {
      type: String,
      default: '学生ID：20232002的知识空间',
    },
    width: {
      type: String,
      default: '1200px',
    },
    top: {
      type: String,
      default: '8%'
    },
    entity: {
      type: Object,
      default: null
    }
  },
  mixins: [visible],
  data() {
    return {
      isClose: false,
    }
  },
  watch: {
    isVisible(val) {
      if (val) {
        if (this.entity) {
        }
      } else {
      }
    }
  },
  mounted() {
    this.appendToBody()
  },
  methods: {
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
    }
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

.stu-dialog-main {
  position: fixed;
  top: 15%;
  width: 1400px;
  left: calc(50% - 300px);
  min-height: 700px;
  box-sizing: border-box;
  backdrop-filter: blur(10px);
  z-index: 9999;
  border-radius: 16px;
  //padding-bottom: 66px;

  .m-dialog-main-inner {
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
        font-size: 20px;
        font-family: YouSheBiaoTiHei;
        color: #ffffff;
      }
    }

    .dialog-body {
      width: 100%;
      height: 700px;
      position: relative;
    }
  }

}

.upper-main{
  margin: 16px 0
}

.hr-main{
  height: 90%;
  width: 1px;
  background-color: rgba(255, 255, 255, 0.4);
}

.lower-main{
  width: 100%;
  display: flex;
  height: calc(100% - 100px);
  justify-content: space-between;
}
</style>
