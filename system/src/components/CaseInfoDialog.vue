
<template>
    <div>
        <div class="dialog-modal" v-if="isVisible && modal"></div>
        <transition name="dialog">
            <div class="m-dialog-main" v-if="isVisible">
                <dv-border-box-13>
                    <div class="m-dialog-main-inner">
                        <div class="dialog-head">
                            <div class="dialog-head-title">案件信息</div>
                            <cross-sign @click.native="closeThis" />
                        </div>
                        <div class="dialog-body">
                            <div class="table-head-view table-head">
                                <div class="table-head-item">案件编号</div>
                                <div class="table-head-item">处置部门</div>
                                <div class="table-head-item">发生地点</div>
                                <div class="table-head-item">问题描述</div>
                                <div class="table-head-item">案件来源</div>
                                <div class="table-head-item">发现时间</div>
                                <div class="table-head-item">工单状态</div>
                            </div>
                            <div class="table-list">
                                <auto-scroll-list :list-data="dataList" v-if="dataList.length > 0">
                                    <div class="table-line-out-view">
                                        <div class="table-line-view" v-for="i in dataList" :key="i.id">
                                            <div class="table-line-item">{{ i.taskid }}</div>
                                            <div class="table-line-item">{{ i.streetname }}</div>
                                            <div class="table-line-item">{{ i.address }}</div>
                                            <div class="table-line-item">{{ i.description }}</div>
                                            <div class="table-line-item">{{ i.infosourcename }}</div>
                                            <div class="table-line-item">{{ i.discovertime | formatDate('yyyy-MM-dd hh:mm')
                                            }}</div>
                                            <div class="table-line-item">{{ i.statusname }}</div>
                                        </div>
                                    </div>
                                </auto-scroll-list>
                                <div class="table-no-data" v-else>暂无数据</div>
                            </div>
                            <div class="table-paginaiton">
                                <pagination :total="page.total" :currentPage="page.current" :pagesize="page.limit" @change="pageChange"></pagination>
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
import UnitNumber from '@/components/UnitNumber'
import Pagination from '@/components/Pagination'
import AutoScrollList from '@/components/AutoScrollList'
import { getCaseList } from '@/api/page'

export default {
    name: "PartyBuildDialog",
    components: { CrossSign, UnitNumber, AutoScrollList, Pagination },
    props: {
        modal: {
            type: Boolean,
            default: true,
        },
        params: {
            type: Object,
            default: ()=>{return {}}
        }
    },
    mixins: [visible],
    watch: {
        isVisible(val) {
            if (val) {
                this.page = {
                    total: 0,
                    limit: 25,
                    current: 1
                }
                this.getCaseList()
            }
        }
    },
    data() {
        return {
            dataList: [],

            page: {
                total: 0,
                limit: 25,
                current: 1
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
        getCaseList() {
            getCaseList({
                ...this.params,
                curpage: this.page.current,
                pagesize: this.page.limit
            }).then(res => {
                this.dataList = res.result || []
                this.page.total = res.total
            })
        },
        pageChange(page){
            console.log('pageChange',page)
            this.page.current = page
            this.getCaseList()
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
    width: 800px;
    left: calc(50% - 400px);
    min-height: 500px;
    box-sizing: border-box;
    z-index: 9999;
    border-radius: 16px;

    .m-dialog-main-inner {
        left: calc(50% - 400px);
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
            height: 680px;
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

            .table-paginaiton {
                flex-shrink: 0;
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #03193370;
            }
        }
    }

}
</style>