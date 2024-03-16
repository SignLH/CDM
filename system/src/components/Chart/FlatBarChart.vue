<!--
*@flat-bar-chart
*@author yuge
*@date 2023/6/26 16:21
-->
<template>
  <div class="flat-bar-chart-main" ref="chartDiv"></div>
</template>

<script>
import * as echarts from 'echarts'

export default {
  name: "FlatBarChart",
  props: {
    //显示数据
    showData: {
      type: Object, default: () => ({
        //['高风险', '中风险', '低风险']
        dataList: [30, 20, 50],
        // itemNameList: ['高风险', '中风险', '低风险'],
        // colorList: ['#ff000090', '#ffb80090', '#1089E790',],
      })
    },
  },
  watch: {
    showData: {
      handler(v) {
        v&&this.initChart(v)
      },
      deep: true,
    }
  },
  components: {},
  data() {
    return {
      mChart: null
    }
  },
  mounted() {
    try {
      this.mChart = echarts.init(this.$refs.chartDiv)
    } catch (e) {
    }
    this.showData&&this.initChart(this.showData)
  },
  methods: {
    initChart(chartData) {
      // const dataList = [30,20,50]
      // const border = [100,100,100]
      // const itemNameList = ['高风险','中风险','低风险']
      // const colorList = ['#ff0000','#ffb800',  '#1089E7',];
      console.log(chartData)

      const data = chartData.dataList
      const barWidth = 15

      const colorList = [];
      colorList[2] = new echarts.graphic.LinearGradient(
        0, 0, 1, 0,
        [
          {offset: 0, color: '#0275f250'},
          {offset: 1, color: '#0275f2'}
        ]
      )
      colorList[1] = new echarts.graphic.LinearGradient(
        0, 0, 1, 0,
        [
          {offset: 0, color: '#f5cf0050'},
          {offset: 1, color: '#f5cf00'}
        ]
      )
      colorList[0] = new echarts.graphic.LinearGradient(
        0, 0, 1, 0,
        [
          {offset: 0, color: '#ff000050'},
          {offset: 1, color: '#ff0000'}
        ]
      )
      const option = {
        grid: {
          top: '4%',
          left: '70',
          right: '50',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'value',
          splitLine: {
            show: false,
          },
          offset: 10,
          axisTick: {
            show: false,
          },
          axisLine: {
            show: false,
          },
          axisLabel: {
            show: false,
          },
        },
        yAxis: {
          type: 'category',
          offset: '60',
          axisLine: {
            show: false,
          },
          axisTick: {
            show: false,
          },
          axisLabel: {
            inside: true,
            align: 'left',
            fontSize: 14,
            color: '#fff',
            // verticalAlign: 'top'
          },
          data: ['高风险', '中风险', '低风险']
        },
        series: [
          {
            name: '2011年',
            type: 'bar',
            // type: 'pictorialBar',
            barWidth,
            // symbol: 'rect',
            // symbolRepeat: 'fixed',
            // symbolMargin: '40%',
            // symbolClip: true,
            // symbolOffset: [5, 0],
            // stack: '12',
            // symbolSize: [4, 15],
            // symbolBoundingData: 100,
            itemStyle: {
              normal: {
                color(params) {
                  // build a color map as your need.
                  // var colorList = [
                  //   '#25defd','#ebc701','#00ddc2'
                  // ];
                  return colorList[params.dataIndex]
                },
                barBorderRadius: [20, 20, 20, 20],
              }
            },
            label: {
              show: false,
              position: 'right',
              offset: [5, 0],
              fontSize: 16,
              color: '#fff',
              formatter: '{c}'
            },
            z: 9999,
            data: data
          },
          {
            name: '2011年',
            type: 'bar',
            // type: 'pictorialBar',
            barWidth: barWidth,
            padding:[2,0],
            // symbolOffset: [16, 6],
            // animationDuration: 0,
            // symbolRepeat: 'fixed',
            // symbolMargin: '40%',
            // symbol: 'rect',
            // symbolSize: [4, 20],
            // symbolBoundingData: 100,
            barGap: '-100%',
            z: 99,
            itemStyle: {
              normal: {
                color: 'rgba(3,15,59,0.38)'
              }
            },
            label: {
              show: true,
              position: 'right',
              // distance: 20,
              offset: [5, 0],
              fontSize: 16,
              color: '#fff',
              formatter(params){
                return data[params.dataIndex]
              }
            },
            data: [100, 100, 100]
          },
          {
            // 分隔
            type: 'pictorialBar',
            itemStyle: {
              normal: {
                color: '#100c29',
              },
              barBorderRadius: [20, 20, 20, 20],
            },
            symbolRepeat: 'fixed',
            symbolMargin: 5,
            symbol: 'rect',
            symbolClip: true,
            symbolSize: [4, barWidth],
            symbolPosition: 'start',
            symbolOffset: [-4, 0],
            // symbolBoundingData: this.total,
            data,
            width: barWidth,
            z: 0,
            zlevel: 1,
          },
        ]
      };

      this.mChart.setOption(option)
    }
  },
  destroyed() {
  }
}
</script>

<style scoped lang="scss">
.flat-bar-chart-main {
  height: 100%;
  width: 100%;
}
</style>
