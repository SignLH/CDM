
<template>
    <div>
        <div class="dialog-modal" v-if="isVisible && modal"></div>
        <transition name="dialog">
            <div class="m-dialog-main" v-if="isVisible">
                <dv-border-box-13>
                    <div class="m-dialog-main-inner">
                        <div class="dialog-head">
                            <div class="dialog-head-title">独居老人信息</div>
                            <cross-sign @click.native="closeThis" />
                        </div>
                        <div class="dialog-body">
                            <div class="table-head-view table-head">
                            <div class="table-head-item">姓名</div>
                            <div class="table-head-item">性别</div>
                            <div class="table-head-item">家庭住址</div>
                            <div class="table-head-item">联系方式</div>
                            <div class="table-head-item">监护人</div>
                            <div class="table-head-item">监护人电话</div>
                            <div class="table-head-item">设备绑定</div>
                        </div>
                        <div class="table-list">
                            <auto-scroll-list :list-data="dataList" v-if="dataList.length > 0">
                                <div class="table-line-out-view">
                                    <div class="table-line-view" v-for="i in dataList" :key="i.id">
                                        <div class="table-line-item">{{ i.name }}</div>
                                        <div class="table-line-item">{{ i.sex }}</div>
                                        <div class="table-line-item">{{ i.address }}</div>
                                        <div class="table-line-item">{{ i.phone }}</div>
                                        <div class="table-line-item">{{ i.guarder_name }}</div>
                                        <div class="table-line-item">{{ i.guarder_phone }}</div>
                                        <div class="table-line-item">{{ i.isbind == 1? '绑定' :'未绑定' }}</div>
                                    </div>
                                </div>
                            </auto-scroll-list>
                            <div class="table-no-data" v-else>暂无数据</div>
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
import CrossSign from "@/components/CrossSign/index.vue";
import UnitNumber from '@/components/UnitNumber.vue'
import AutoScrollList from '@/components/AutoScrollList.vue'
import { getSolitaryElderly } from '@/api/home'

export default {
    name: "SolitaryElderlyDialog",
    components: { CrossSign, UnitNumber, AutoScrollList },
    props: {
        modal: {
            type: Boolean,
            default: true,
        }
    },
    mixins: [visible],
    watch: {
        isVisible(val) {
            if (val) {
                this.getSolitaryElderly()
            }
        }
    },
    data() {
        return {
            dataList: []
        }
    },
    mounted() {
        this.appendToBody()
    },
    methods: {
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

        //获取全部信息
        getSolitaryElderly() {
            getSolitaryElderly().then(res => {
                this.dataList = res.result || []
            })
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
    top: 10%;
    width: 700px;
    left: calc(50% - 350px);
    min-height: 500px;
    box-sizing: border-box;
    z-index: 9999;
    border-radius: 16px;

    .m-dialog-main-inner {
        left: calc(50% - 350px);
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
            height: 520px;
            display: flex;
            flex-direction: column;

            .table-head {
                flex-shrink: 0;
                width: 100%;
            }

            .table-list {
                width: 100%;
                flex: 1;
                position: relative;
                overflow: hidden;
                background: #01183542;
            }
        }
    }

}
</style>