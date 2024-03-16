<template>
  <div class="container">
    <div ref="chart"></div>
  </div>
</template>

<script>
import {mapGetters} from 'vuex'
import * as echarts from 'echarts'

export default {
  name: 'LeftBarChart',
  props: {
    dataList: {
      type: Array,
      default: () => {
        return []
      }
    }
  },
  computed:{
  },
  mounted() {
    this.chart = echarts.init(this.$refs.chart)
    this.renderChart()
  },
  data() {
    return {
      chart: null,
      isDialogShow:false,
      entity:{}
    }
  },
  watch:{
  },
  methods: {
    renderChart() {

      var data = [{
        "name": "简单",
        "value": 4
      }, {
        "name": "中等",
        "value": 6
      }, {
        "name": "困难",
        "value": 7
      },];
      var xData = [],
          yData = [];
      var min = 50;
      data.map(function(a, b) {
        xData.push(a.name);
        if (a.value === 0) {
          yData.push(a.value + min);
        } else {
          yData.push(a.value);
        }
      });
      const option = {
        color: ['#3398DB'],
        // tooltip: {
        //   trigger: 'axis',
        //   axisPointer: {
        //     type: 'line',
        //     lineStyle: {
        //       opacity: 0
        //     }
        //   },
        // },
        legend: {
          data: ['直接访问', '背景'],
          show: false
        },
        grid: {
          left: '5%',
          right: '5%',
          bottom: '5%',
          top: '7%',
          height: '85%',
          containLabel: true,
          z: 22
        },
        xAxis: [{
          type: 'category',
          gridIndex: 0,
          data: xData,
          axisTick: {
            alignWithLabel: true
          },
          axisLine: {
            lineStyle: {
              color: '#0c3b71'
            }
          },
          axisLabel: {
            show: true,
            color: 'rgb(170,170,170)',
            fontSize:16
          }
        }],
        yAxis: [{
          type: 'value',
          gridIndex: 0,
          splitLine: {
            show: false
          },
          axisTick: {
            show: false
          },
          axisLine: {
            lineStyle: {
              color: '#0c3b71'
            }
          },
          axisLabel: {
            color: 'rgb(170,170,170)',
            formatter: '{value} '
          }
        },
          {
            type: 'value',
            gridIndex: 0,
            splitNumber: 12,
            splitLine: {
              show: false
            },
            axisLine: {
              show: false
            },
            axisTick: {
              show: false
            },
            axisLabel: {
              show: false
            },
            splitArea: {
              show: true,
              areaStyle: {
                color: ['rgba(250,250,250,0.0)', 'rgba(250,250,250,0.05)']
              }
            }
          }
        ],
        series: [{
          type: 'bar',
          barWidth: '30%',
          xAxisIndex: 0,
          yAxisIndex: 0,
          itemStyle: {
            normal: {
              barBorderRadius: 30,
              color: new echarts.graphic.LinearGradient(
                  0, 0, 0, 1, [{
                    offset: 0,
                    color: '#00feff'
                  },
                    {
                      offset: 0.5,
                      color: '#027eff'
                    },
                    {
                      offset: 1,
                      color: '#0286ff'
                    }
                  ]
              )
            }
          },
          data: yData,
          zlevel: 11

        },
          {
            type: 'bar',
            barWidth: '50%',
            xAxisIndex: 0,
            yAxisIndex: 1,
            barGap: '-135%',
            data: [100, 100, 100, 100, 100, 100, 100],
            itemStyle: {
              normal: {
                color: 'rgba(255,255,255,0.1)'
              }
            },
            zlevel: 9
          },

        ]
      };



      this.chart.setOption(option)
    },

    destroy() {
    }
  }
}
</script>

<style lang="scss" scoped>
div {
  height: 100%;
  width: 100%;
}
</style>
