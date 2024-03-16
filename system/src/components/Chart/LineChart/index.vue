<!--
*@line-chart
*@author yuge
*@date 2023/4/27 14:05
-->
<template>
  <div class="line-chart-main">
    <div class="line-chart" ref="mChart"></div>
  </div>
</template>

<script>
import * as echarts from 'echarts'
import {getRandomColor} from "@/utils/gyy-util";

/**
 * 使用 只可以数据显示 目前未加入事件
 */
export default {
  name: "line-chart",
  props: {
    //标题
    title: {type: String, default: null},
    //显示数据
    showData: {
      type: Object, default: () => ({
        xData:['2018', '2019', '2020', '2021', '2022', '2023'],
        data:[500, 260, 130, 130, 125, 150]
      })
    },
    // 主题 light or dark
    theme: {type: String, default: 'light'},
  },
  components: {},
  data() {
    return {
      mChart: null,
    }
  },
  watch: {
    showData() {
      this.initChart()
    }
  },
  mounted() {
    this.mChart = echarts.init(this.$refs.mChart);
    this.initChart()
  },
  methods: {
    initChart() {
      const showData = this.showData

      if (!showData){
        console.log('无数据')
        return
      }

      const {xData, data} = showData
      let xAxisStyle = {}
      if (showData.xAxisStyle) {
        xAxisStyle = showData.xAxisStyle
      }
      const option = {
        backgroundColor: '#00265f50',
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow'
          }
        },
        grid: {
          top: '10',
          right: '5%',
          left: '11%',
          bottom: '20%'
        },
        xAxis: [{
          type: 'category',
          data: xData,
          axisLine: {
            lineStyle: {
              color: '#0173e620'
            }
          },
          axisLabel: {
            margin: 10,
            color: '#0173e6',
            textStyle: {
              fontSize: 12
            },
          },
        }],
        yAxis: [{
          minInterval:1,
          axisLabel: {
            formatter: '{value}',
            color: '#0173e6',
          },
          axisLine: {
            show: false,
            lineStyle: {
              color: '#0173e620'
            }
          },
          splitLine: {
            lineStyle: {
              color: '#0173e620'
            }
          }
        }],
        series: [{
          type: 'bar',
          data: data,
          barWidth: '8px',
          itemStyle: {
            normal: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                offset: 0,
                color: 'rgba(0,244,255,1)' // 0% 处的颜色
              }, {
                offset: 1,
                color: 'rgba(0,77,167,1)' // 100% 处的颜色
              }], false),
              barBorderRadius: [30, 30, 30, 30],
              shadowColor: 'rgba(0,160,221,.5)',
              shadowBlur: 4,
            }
          },
          label: {
            normal: {
              show: false,
              lineHeight: 30,
              width: 80,
              height: 30,
              backgroundColor: 'rgba(0,160,221,0.1)',
              borderRadius: 200,
              position: ['-8', '-60'],
              distance: 1,
              formatter: [
                '    {d|●}',
                ' {a|{c}}     \n',
                '    {b|}'
              ].join(','),
              rich: {
                d: {
                  color: '#3CDDCF',
                },
                a: {
                  color: '#fff',
                  align: 'center',
                },
                b: {
                  width: 1,
                  height: 30,
                  borderWidth: 1,
                  borderColor: '#234e6c',
                  align: 'left'
                },
              }
            }
          }
        }]
      };

      // 使用刚指定的配置项和数据显示图表。
      this.mChart.setOption(option);
    },
  },
  destroyed() {
  }
}
</script>

<style scoped lang="scss">
.line-chart-main {
  height: 100%;
  width: 100%;
  position: relative;
  .line-chart{
    height: 100%;
    width: 100%;
    z-index: 1;
  }
  .text{
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    color: rgba(255, 255, 255, 0.83);
    backdrop-filter: blur(7px);
  }
}
</style>
