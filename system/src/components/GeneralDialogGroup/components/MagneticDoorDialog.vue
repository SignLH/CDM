<template>
  <div>
    <div class="dialog-modal" v-if="isVisible && entity != null"></div>
    <transition name="dialog">
      <div class="m-dialog-main" v-if="isVisible" :style="{ left: `calc(50% - ${width} / 2)`, width, top }">
        <dv-border-box-11 :title="entity.device_name" :titleWidth="270" :color="['#43beb9', 'green']">
          <div class="m-dialog-main-inner">
            <div class="dialog-head">
              <div class="dialog-head-title">{{ entity.device_type }}</div>
              <cross-sign @click.native="closeThis"/>
            </div>
            <div class="dialog-body mc-main">
              <div class="mc-description">
                {{showData.location_description}}
                <div class="gender-unit" v-if="showData.device_status==1">正常</div>
                <div class="gender-unit lost" v-else>异常</div>
              </div>
              <custom-form class="mc-form" title-width="100px" :form-data="formData" null-hided>
                <template #lastData>
                  {{updatedAtL}}
                  <span class="last-data online" v-if="lastState.action=='opened'">开</span>
                  <span class="last-data" v-else-if="lastState.action=='closed'">关</span>
                  <span class="last-data">-</span>
                </template>
              </custom-form>
            </div>
          </div>
        </dv-border-box-11>
      </div>
    </transition>
  </div>
</template>

<script>

import visible from "@/mixins/edit-dialog"
import CrossSign from "@/components/CrossSign";
import VideoMonitor from "@/components/VideoMonitor";
import {openMonitors, closeMonitors} from '@/api/page'
import {mapGetters} from 'vuex'
import CustomForm from "@/components/CustomForm.vue";

export default {
  name: "MagneticDoorDialog",
  components: {CustomForm, VideoMonitor, CrossSign},
  computed: {
    ...mapGetters([
      'pageMonitorList'
    ]),
  },
  props: {
    title: {
      type: String,
      default: '标题',
    },
    width: {
      type: String,
      default: '500px',
    },
    top: {
      type: String,
      default: '15%'
    },
    entity: {
      type: Object,
      default: null
    }
  },
  mixins: [visible],
  data() {
    return {
      showData:{},
      lastState:{},
      updatedAtL:'',
      formData:[],
    }
  },
  watch: {
    isVisible(val) {
      if (val) {
        this.handleOpen()
      }
    }
  },
  mounted() {
    this.appendToBody()
  },
  methods: {
    //处理窗口打开逻辑
    handleOpen() {
      let data=this.entity
      this.showData=data
      try{
        if (data.laststate){
          this.lastState=JSON.parse(data.laststate)
          if (this.lastState.info){
            const info=this.lastState.info
            this.updatedAtL=info.updatedAtL
          }
        }
      }catch (e) {
        console.warn(e)
      }
      this.LOG.info(['dialog open',this.showData])
      this.formData=[
        {title:'住户门牌',value:data.extended},
        // {title:'出生日期',value:'-'},
        {title:'地址',value:data.address},
        {title:'联系方式',value:data.phone},
        {title:'第一监护人',value:data.hzname},
        {type:'slot',title:'最新数据',value:'lastData'},
      ]
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

.m-dialog-main {
  position: fixed;
  top: 15%;
  width: 600px;
  left: calc(50% - 300px);
  height: 370px;
  box-sizing: border-box;
  //border: 1px #0080fe solid;
  z-index: 9999;
  border-radius: 16px;
  //padding-bottom: 66px;

  .m-dialog-main-inner {
    width: 100%;
    height: 100%;
    left: calc(50% - 300px);
    padding: 16px;
    padding-top: 0;
    box-sizing: border-box;

    .dialog-head {
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 10px;
      height: 50px;
      .dialog-head-title {
        font-size: 20px;
        font-family: YouSheBiaoTiHei;
        color: #ffffff;
      }
    }

    .dialog-body {
      background-image: linear-gradient(135deg, rgba(32, 100, 17, 0.66) 0%, rgba(0, 0, 0, 0.7) 100%);
      border-radius: 8px;
      width: 100%;
      height: calc(100% - 50px);

      padding: 16px;
    }
  }
}

//门磁
.mc-main{
  width: 100%;
  height: 100%;
  overflow: hidden;
  color: #ffffff;

  .mc-description{
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    line-height: 36px;

    .gender-unit{
      line-height: 34px;
      display: inline-block;
      padding: 0 8px;
      border-radius:4px;
      background-color: rgba(11, 255, 166, 0.8);
      font-family: YouSheBiaoTiHei;
      font-weight: normal;
    }
    .female{
      background-color: rgba(239, 75, 76, 0.8);
    }
    .lost{
      background-color: rgba(128, 128, 128, 0.86);
    }
  }

  .mc-form{
    height: calc(100% - 44px - 8px);
    margin-top: 8px;

    .online{
      color: #1efa9f !important;
      background: #1efa9f50;
      text-shadow: 0 0 1px #0BFFA650  !important;
    }
    .last-data{
      color: #9ba29e;
      padding: 2px;
      border-radius: 5px;
      box-shadow: 0 0 11px #9ba29e50 ;
    }
  }
}

</style>
