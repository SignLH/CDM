<!--
*@VideoMonitor
*@author GYY
*@date 2023/2/25 14:47
-->
<template>
  <div class="video-monitor-main">
    <div class="img">
      <video-player class="video-player vjs-custom-skin v-player" :ref="id" :id="id" :options="playerOptions"
                    @error="reLoadVideo"></video-player>

    </div>
  </div>
</template>

<script>
import "vue-video-player/src/custom-theme.css";
import {videoPlayer} from "vue-video-player";
import "video.js/dist/video-js.css";
import "videojs-contrib-hls";

export default {
  name: "VideoMonitorOld",
  props: {
    srcUrl: {
      type: String,
      default: ''
    }
  },
  components: {videoPlayer},
  created() {
    this.id = this.randomString(10)
  },
  data() {
    return {
      id: '',
      playerOptions: {
        playbackRates: [0.5, 1.0, 1.5, 2.0], //可选择的播放速度
        autoplay: true, //如果true,浏览器准备好时开始回放。
        muted: true, // 默认情况下将会消除任何音频。
        loop: true, // 视频一结束就重新开始。
        preload: "auto", // 建议浏览器在<video>加载元素后是否应该开始下载视频数据。auto浏览器选择最佳行为,立即开始加载视频（如果浏览器支持）
        language: "zh-CN",
        aspectRatio: "16:9", // 将播放器置于流畅模式，并在计算播放器的动态大小时使用该值。值应该代表一个比例 - 用冒号分隔的两个数字（例如"16:9"或"4:3"）
        fluid: true, // 当true时，Video.js player将拥有流体大小。换句话说，它将按比例缩放以适应其容器。
        sources: [
          {
            type: "application/x-mpegURL",
            src: ''
          },
        ],
        poster: "", //你的封面地址

        // width: document.documentElement.clientWidth,
        // height: "500",
        notSupportedMessage: "加载失败", //允许覆盖Video.js无法播放媒体源时显示的默认信息。
        controlBar: {
          timeDivider: true, //当前时间和持续时间的分隔符
          durationDisplay: true, //显示持续时间
          remainingTimeDisplay: false, //是否显示剩余时间功能
          fullscreenToggle: true, //全屏按钮
        },
      },
    }
  },
  watch: {
    srcUrl: {
      handler(newValue, _) {
        this.$set(this.playerOptions, 'sources', [
          {
            type: "application/x-mpegURL",
            src: newValue
          },
        ])
      },
      immediate: true
    }
  },
  mounted() {
  },
  methods: {
    /**
     * 随机生成字符串
     * @param len 指定生成字符串长度
     */
    randomString(len) {
      len = len || 10;
      const $chars = 'ABCDEFGHJKMNPQRSTWXYZabcdefhijkmnprstwxyz2345678';
      /****默认去掉了容易混淆的字符oOLl,9gq,Vv,Uu,I1****/
      let maxPos = $chars.length;
      let pwd = '';
      for (let i = 0; i < len; i++) {
        pwd += $chars.charAt(Math.floor(Math.random() * maxPos));
      }
      return pwd;
    },

    reLoadVideo() {
      // console.error("reLoadVideo")
      setTimeout(() => {
        this.playerPlay()

      }, 1000)
    },
    playerPlay() {
      try {
        this.$refs[this.id].dispose(() => {
          this.$refs[this.id].initialize()
        })
      } catch (e) {
      }
    },
  },
  destroyed() {
    try {
      this.$refs[this.id].dispose()
    } catch (e) {
    }
  }
}
</script>

<style lang='scss' scoped>
.video-monitor-main {
  width: 100%;
  height: 100%;
  background-color: #000000;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;

  .img {
    width: 100%;
  }
}
</style>
