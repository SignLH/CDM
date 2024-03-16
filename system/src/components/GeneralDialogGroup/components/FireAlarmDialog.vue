<template>
  <div>
    <div class="dialog-modal" v-if="isVisible && entity != null"></div>
    <transition name="dialog">
      <div class="m-dialog-main" v-if="isVisible" :style="{ left: `calc(50% - ${width} / 2)`, width, top }">
        <dv-border-box-11 :title="entity.device_name+'（'+deviceCommercialType+'）'" :titleWidth="270"
                          :color="[ '#9b0202','#000000',]">
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
              <custom-form class="mc-form" title-width="70px" :form-data="formData" null-hided>
                <template #lastData>
                  <div class="last-data online" v-if="showData.device_status==1">在线</div>
                  <div class="last-data" v-else>离线</div>
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
  name: "FireAlarmDialog",
  components: {CustomForm, VideoMonitor, CrossSign},
  computed: {
    ...mapGetters([
      'pageMonitorList'
    ]),
    deviceCommercialType(){
      if (!this.entity){
        return ''
      }
      const commercial=['燃气报警器']
      for (let string of commercial) {
        if (this.entity.device_type.search(string)!==-1){
          return '商用'
        }
      }
      return '民用'
    },
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
      this.LOG.info(['dialog open',this.showData])
      this.formData=[
        {title:'地址',value:data.address},
        {title:'类型',value:data.device_type},
        {title:'住户',value:data.hzname},
        {title:'电话',value:data.phone},
        {type:'slot',title:'状态',value:'lastData'},
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
      background-image: linear-gradient(135deg, #4b483f90, #260000);
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
      box-shadow: 0 0 11px #0BFFA650  !important;
    }
    .last-data{
      color: #9ba29e;
      padding: 2px 4px;
      border-radius: 5px;
      box-shadow: 0 0 11px #9ba29e50 ;
    }
  }
}

</style>
