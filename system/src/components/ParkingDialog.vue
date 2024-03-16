
<template>
    <div>
        <div class="dialog-modal" v-if="isVisible && modal"></div>
        <transition name="dialog">
            <div class="m-dialog-main" v-if="isVisible">
                <dv-border-box-13>
                    <div class="m-dialog-main-inner">
                        <div class="dialog-head">
                            <div class="dialog-head-title">{{ baseData.parkname }}停车统计</div>
                            <cross-sign @click.native="closeThis" />
                        </div>
                        <div class="dialog-body">
                            <div class="top-info">
                                <div class="left-item">
                                    <div class="value">
                                        <img src="@/assets/stzl/stzl-back1.png" class="back" />
                                        <unit-number number-size="19" :number="baseData.parkspacetotal"></unit-number>
                                    </div>
                                    <div class="label">总车位数</div>
                                </div>
                                <div class="left-item">
                                    <div class="value">
                                        <img src="@/assets/stzl/stzl-back1.png" class="back" />
                                        <unit-number number-size="19" :number="baseData.remainingspace"></unit-number>
                                    </div>
                                    <div class="label">剩余车位数</div>
                                </div>
                                <div class="left-item">
                                    <div class="value">
                                        <img src="@/assets/stzl/stzl-back1.png" class="back" />
                                        <unit-number number-size="19" :number="baseData.totalin"></unit-number>
                                    </div>
                                    <div class="label">总进车辆数</div>
                                </div>
                                <div class="left-item">
                                    <div class="value">
                                        <img src="@/assets/stzl/stzl-back1.png" class="back" />
                                        <unit-number number-size="19" :number="baseData.totalout"></unit-number>
                                    </div>
                                    <div class="label">总出车辆数</div>
                                </div>
                                <div class="left-item">
                                    <div class="value">
                                        <img src="@/assets/stzl/stzl-back1.png" class="back" />
                                        <unit-number number-size="19" :number="baseData.total"></unit-number>
                                    </div>
                                    <div class="label">合计总次数</div>
                                </div>

                            </div>
                            <div class="chart-view">
                                <div class="chart-left">
                                    <spot-label label="使用情况"></spot-label>
                                    <div class="l-chart" id="countChart"></div>
                                </div>
                                <div class="chart-right">
                                    <spot-label label="进出量统计"></spot-label>
                                    <div class="r-chart" id="jcChart"></div>
                                </div>
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
import { getCompanyinout } from '@/api/home'
import SpotLabel from '@/components/SpotLabel'
import * as echarts from 'echarts'

export default {
    name: "PartyBuildDialog",
    components: { CrossSign, UnitNumber, SpotLabel },
    props: {
        modal: {
            type: Boolean,
            default: true,
        },
        baseData: {
            type: Object,
            default: {},
        }
    },
    mixins: [visible],
    watch: {
        isVisible(val) {
            if (val) {
                this.getJcChartData()
            }
        }
    },
    data() {
        return {

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
        //加载使用情况图表
        initCountChart(data) {
            let countChart = echarts.init(document.getElementById('countChart'));
            // 绘制图表
            countChart.setOption({
                tooltip: {
                    trigger: 'item'
                },
                series: [
                    {
                        type: 'pie',
                        radius: '50%',

                        data: data
                    }
                ]
            });
        },

        //获取全部信息
        getJcChartData() {
            getCompanyinout().then(res => {
                let root = res.result || []
                let xData = []
                let yData1 = []
                let yData2 = []
                root.forEach(r => {
                    xData.push(r.month)
                    yData1.push(r.green + this.baseData.totalout)
                    yData2.push(r.yellow + this.baseData.totalin)
                });
                this.initJcChart(xData, yData1, yData2)


                ///一起加载
                let data = [{
                    name: '已用车位',
                    value: this.baseData.parkspacetotal - this.baseData.remainingspace
                }, {
                    name: '剩余车位',
                    value: this.baseData.remainingspace
                }]
                this.initCountChart(data)
            })
        },

        //加载进出图表
        initJcChart(xData, yData1, yData2) {
            let jcChart = echarts.init(document.getElementById('jcChart'));
            // 绘制图表
            jcChart.setOption({
                grid: {
                    left: '20%',
                    right: '5%',
                    bottom: '20%',
                    top: '6%'
                },
                tooltip: {
                    trigger: 'axis',
                },
                xAxis: {
                    type: 'category',
                    boundaryGap: false,
                    splitLine: {
                        show: false
                    },
                    axisLine: {
                        show: false
                    },
                    axisTick: {
                        show: false
                    },
                    data: xData,
                    axisLabel: {
                        show: true,
                        textStyle: {
                            color: '#0855a2',
                            fontSize: 10
                        },
                        rotate: 45
                    }
                },
                yAxis: {
                    type: 'value',
                    splitLine: {
                        show: false
                    },
                    axisLabel: {
                        show: true,
                        textStyle: {
                            color: '#0855a2',
                            fontSize: 10
                        }
                    }
                },
                series: [
                    {
                        name: '进',
                        type: 'line',
                        smooth: true,
                        data: yData1,
                        itemStyle: {
                            normal: {
                                color: '#07cf86'
                            }
                        }
                    },
                    {
                        name: '出',
                        type: 'line',
                        smooth: true,
                        data: yData2,
                        itemStyle: {
                            normal: {
                                color: '#0f71ce'
                            }
                        }
                    },
                ]
            });
        },
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
    top: 20%;
    width: 600px;
    left: calc(50% - 300px);
    min-height: 300px;
    box-sizing: border-box;
    z-index: 9999;
    border-radius: 16px;

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
            min-height: 300px;
            display: flex;
            flex-direction: column;

            .top-info {
                display: flex;
                background: #03274980;
                padding: 5px;
                flex-shrink: 0;

                .left-item {
                    flex: 1;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;

                    .value {
                        width: 100%;
                        height: 36px;
                        position: relative;
                        color: #0BFFA6;
                        display: flex;
                        align-items: center;
                        justify-content: center;

                        .back {
                            position: absolute;
                            top: 0;
                            left: 0;
                            width: 100%;
                            height: 100%;
                        }
                    }


                    .label {
                        margin-top: 3px;
                        color: #ffffff;
                        text-align: center;
                        font-size: 14px;
                    }
                }

                .left-item+.left-item {
                    margin-left: 8px;
                }

            }

            .chart-view {
                margin-top: 10px;
                margin-bottom: 10px;
                width: 100%;
                height: 270px;
                display: flex;

                .chart-left {
                    width: 260px;
                    height: 100%;
                    flex-shrink: 0;
                    background: #01183569;
                    padding: 5px;

                    .l-chart {
                        width: 100%;
                        margin-top: 8px;
                        height: calc(100% - 20px);
                    }
                }

                .chart-right {
                    margin-left: 10px;
                    flex: 1;
                    height: 100%;
                    background: #01183569;
                    padding: 5px;

                    .r-chart {
                        width: 100%;
                        margin-top: 8px;
                        height: calc(100% - 20px);
                    }
                }
            }
        }
    }

}
</style>