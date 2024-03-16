<template>
  <div>
    <div class="dialog-modal" v-if="isVisible"></div>
    <transition name="dialog">
      <div class="solos-call-main" v-if="isVisible" :style="{ left: `calc(50% - ${width} / 2)`, width, top }">
        <dv-border-box-13>
          <div class="m-dialog-main-inner">
            <div class="dialog-head">
              <div class="dialog-head-title">{{ title }}</div>
              <cross-sign @click.native="closeThis"/>
            </div>
            <div class="dialog-body">
              <div class="loading-tip-view" v-if="isLoading !== 'success'">
                <dv-loading>
                  <span style="color: #0180fa;font-size: 16px;">Loading...</span>
                </dv-loading>
              </div>
              <div ref='id_voice' class="camera-main"/>
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

export default {
  name: "SolosCallDialog",
  components: {CrossSign},
  computed: {},
  props: {
    title: {
      type: String,
      default: '标题',
    },
    width: {
      type: String,
      default: '800px',
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
      hndSession:null,
      hndYsIndex:null,
      fuserCode:null,
      isLoading:'loading',
      isClose:false,
    }
  },
  watch: {
    isVisible(val) {
      if (val) {
        if (this.entity){
          this.isClose=false
          this.$nextTick(() => this.initCall())
        }
      }else {
        this.closeAll()
      }
    }
  },
  mounted() {
    this.appendToBody()
  },
  methods: {
    closeThis() {
      this.isVisible = false
    },
    initCall() {
      this.isLoading='loading'
      const elementDiv=this.$refs.id_voice
      const that=this
      $(function () {
        const initCfgParams = {
          bocxws: true,
          bManualLP: false
        };
        const initParams = {
          url: 'https://cmcy.qiyu-tech.cn:9443',
          calltype: jSW.CallProtoType.AUTO,
          config: initCfgParams,
          oninit: (rc) => {
            if (rc == 0) {
              //库初始化成功
              // 在这之后才能调用其他接口，否则会报错，proto找不到等
              that.hndSession = new jSW.SWSession({
                server: "61.172.176.18",
                port: 9701,
                onopen:  (sess)=>{
                  sess.swLogin({
                    user: "cmcyApi",
                    password: "cmcy@2023"
                  });
                }
              });
              // 注册事件的回调函数
              that.hndSession.swAddCallBack('login', (sender, event, json) => {
                if (json.code == jSW.RcCode.RC_CODE_S_OK) {
                  //初始化语插件界面
                  jSW.GroupCall.Init(elementDiv, that.hndSession, (rc)=>{
                    //在这个回调通知成功后， 就可以调用invite接口了
                    if(rc==0){
                      //openCall();
                      that.$nextTick(() => that.openCall())
                    }
                  });

                } else {
                  console.warn('初始化单兵设备失败！')
                  this.isLoading='error'
                }
              });
            }
          }
        }
        jSW.swInit(initParams);
      });
    },
    openCall() {
      this.LOG.info(['attempt to openCall'])
      const puid=this.entity.device_id
      const name=this.entity.device_name
      if(!puid){
        this.LOG.warn(['attempt to openCall,but puid is ',puid])
        return
      }
      try {
        const szSzPuid = [{puid,name,confid:""}];
        jSW.GroupCall.Invite(szSzPuid, (bresult,szResult) =>{
          if (szResult==1){
            this.LOG.info(['attempt to that.openCall() success!'])
            setTimeout(()=>this.isLoading='success',200)
          }else if (!this.isClose){
            //递归打开
            setTimeout(()=>this.openCall(),200)
          }
        }, "1");
      }catch (e) {
        console.warn('openCall error',e)
        this.isLoading='error'
      }
    },
    closeAll(){
      this.isClose=true
      //关闭销毁语音插件
      jSW.GroupCall.Clear();
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

<style lang='scss' scoped >
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

.solos-call-main {
  position: fixed;
  top: 15%;
  width: 600px;
  left: calc(50% - 300px);
  min-height: 500px;
  box-sizing: border-box;
  //background-color: rgba(11, 62, 95, 0.9);
  //border: 1px #0080fe solid;
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
      height: auto;
      min-height: 400px;
      position: relative;

      .loading-tip-view{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        background: rgba(0, 0, 0, 0.2);
        z-index: 99999;
      }

      .camera-main {
        width: 100%;
        height: 400px;
      }
    }
  }

}
</style>