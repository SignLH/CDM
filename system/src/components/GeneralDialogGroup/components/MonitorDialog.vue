<template>
    <div>
        <div class="dialog-modal" v-if="isVisible && entity != null"></div>
        <transition name="dialog">
            <div class="m-dialog-main" v-if="isVisible" :style="{ left: `calc(50% - ${width} / 2)`, width, top }">
                <dv-border-box-13>
                    <div class="m-dialog-main-inner">
                        <div class="dialog-head">
                            <div class="dialog-head-title">{{ title }}</div>
                            <cross-sign @click.native="closeThis" />
                        </div>
                        <div class="dialog-body">
                            <video-monitor v-if="videoUrl" :src-url="videoUrl" :modal="false"/>
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
import VideoMonitor from "@/components/VideoMonitor";
import { openMonitors, closeMonitors } from '@/api/page'
import { mapGetters } from 'vuex'
export default {
    name: "monitor-dialog",
    components: { VideoMonitor, CrossSign },
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
            videoUrl: null
        }
    },
    watch: {
        isVisible(val) {
            if (val) {
              this.LOG.info(['entity',this.entity])
                this.handleOpenVideoUrl()
            } else {
                this.handleCloseVideoUrl()
            }
        }
    },
    mounted() {
        this.appendToBody()
    },
    methods: {
        //处理窗口打开逻辑
        handleOpenVideoUrl() {
          this.LOG.info(['entity 2',this.entity])
            // 如果是 this.entity.from == 'dialog' 且 pageMonitorList 里不存在 entity；调用打开视频流
            if (this.entity == null) {
                return
            }
            let noIncludeEntity = true

          this.LOG.info(['entity 3 this.pageMonitorList',this.pageMonitorList])
            this.pageMonitorList.forEach(i => {
                if(i.id == this.entity.id){
                    noIncludeEntity = false
                }
            })
          this.LOG.info(['entity 4',this.entity])
            if (this.entity.from && this.entity.from == 'dialog' && noIncludeEntity) {
                openMonitors({
                    mode: 1,
                    type: 0,
                    signal2: [this.entity.mac_address]
                }).then(res=>{
                    this.videoUrl = this.entity.video_address
                })
            }else{
                this.videoUrl = this.entity.video_address
            }

        },
        //处理窗口关闭逻辑
        handleCloseVideoUrl() {
            // 如果是 this.entity.from == 'dialog' 且 pageMonitorList 里不存在 entity；调用关闭视频流
            if (this.entity == null) {
                return
            }
            let noIncludeEntity = true
            this.pageMonitorList.forEach(i => {
                if(i.id == this.entity.id){
                    noIncludeEntity = false
                }
            })
            if (this.entity.from && this.entity.from == 'dialog' && noIncludeEntity) {
                closeMonitors({
                    mode: 1,
                    type: 0,
                    signal2: [this.entity.video_address]
                })
            }
            this.videoUrl = null
        },
        closeThis() {
            this.isVisible = false
            this.$store.dispatch('map/SetMonitorVideoUrl', null)
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
            //height: auto;
            height: 500px;
        }
    }

}
</style>
