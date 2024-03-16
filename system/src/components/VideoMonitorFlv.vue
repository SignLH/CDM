<!--
*@VideoMonitor
*@author GYY
*@date 2023/2/25 14:47
-->
<template>
    <div class="video-monitor-main">
        <div class="tip" v-if="videoPlayerLoading">
            <span>初始化中</span>
        </div>
        <div class="video" v-else>
            <video id="monitor_video" class="video" controls autoplay muted width="100%" height="100%">
            </video>
        </div>
    </div>
</template>

<script>

import flv from "flv.js";

export default {
    name: "VideoMonitorFlv",
    props: {
        srcUrl: {
            type: String,
            default: ''
        }
    },
    components: {},

    data() {
        return {
            player: null,
            videoPlayerLoading: false,

            flvStr: ''
        }
    },
    watch: {
        srcUrl: {
            handler(newValue, _) {
                this.flvStr = newValue
                this.videoPlayerLoading = true
                setTimeout(() => {
                    this.videoPlayerLoading = false
                    this.play()
                }, 500)
            },
            immediate: true
        }
    },
    created() {
    },
    methods: {
        play() {
            if (flv.isSupported()) {
                this.player = flv.createPlayer(
                    {
                        type: "flv",
                        isLive: true,
                        url: this.flvStr,
                        hasAudio: false,
                        withCredentials: false,
                    },
                    {
                        enableStashBuffer: false,
                        isLive: true,
                        lazyLoad: true,
                        stashInitialSize: 128
                    }
                );
                this.player.on(flv.Events.ERROR, (errorInfo, errType, errDetail) => {
                    if (errDetail.code == 404) {
                        setTimeout(() => {
                            this.videoPlayerLoading = true
                            this.reLoadPlay()
                        }, 2500)
                    }
                });
                setTimeout(() => {
                    this.loadPlay()
                }, 200)

            } else {
                console.log("不支持的格式");
                return;
            }

        },
        loadPlay() {
            let video = document.getElementById("monitor_video")
            if (this.player == null) {
                return
            }
            try {
                console.log('player attachMediaElement')

                this.player.attachMediaElement(video);

            } catch (e) {

            }
            this.player.load();
            setTimeout(() => {
                this.player.play();
            }, 1000)

        },
        reLoadPlay() {
            if (this.player == null) {
                return
            }
            this.videoPlayerLoading = true
            this.player.pause();
            this.player.unload();
            this.player.detachMediaElement()
            this.player.destroy();
            this.player = null;

            setTimeout(() => {
                this.videoPlayerLoading = false
                this.play()
            }, 200)
        }
    },
    destroyed() {
        try {
            if (this.player) {
                this.player.pause();
                this.player.unload();
                this.player.detachMediaElement()
                this.player.destroy();
                this.player = null;
            }
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

    .video {
        width: 100%;
        height: 100%;
    }

    .tip {
        width: 100%;
        height: 100%;
        color: #efefef;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }
}
</style>
