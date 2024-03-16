<!--
*@VideoMonitor
*@author GYY
*@date 2023/2/25 14:47
-->
<template>
  <div class="iframe-monitor-main" >
    <div class="iframe-monitor-modal" v-if="modal"/>
    <div class="iframe-video-main">
      <iframe width="100%" height="100%" :src="iframeUrl" frameborder="0"/>
    </div>
  </div>
</template>

<script>

import {getLocalServiceUrl} from "@/utils/app-base";

export default {
  name: "VideoMonitor",
  props: {
    srcUrl: {
      type: String,
      default: ''
    },
    modal:{
      type:Boolean,
      default: true
    }
  },
  created() {
  },
  data() {
    return {
      iframeUrl:null
    }
  },
  watch: {
    srcUrl: {
      handler(newValue, _) {
        if (newValue&&newValue.length>5){
          this.iframeUrl=`${getLocalServiceUrl()}/#/monitor?entity=${
            JSON.stringify({videoUrl:newValue})
          }`
          console.log(this.iframeUrl)
        }
      },
      immediate: true
    }
  },
  mounted() {
  },
  methods: {
  },
  destroyed() {
  }
}
</script>

<style lang='scss' scoped>
.iframe-monitor-main {
  width: 100%;
  height: 100%;
  background-color: #000000;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  position: relative;

  .iframe-monitor-modal{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 2;
    background-color: transparent;
  }

  .iframe-video-main {
    width: 100%;
    height: 100%;
    z-index: 1;
  }
}
</style>
