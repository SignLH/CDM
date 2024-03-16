<!--
*@line-chart
*@author yuge
*@date 2023/4/27 14:05
-->
<template>
  <div class="line-chart-main">
    <div class="line-chart" ref="mChart"></div>
    <div class="text" style="z-index: 50" v-show="isNull&&false">
      无数据
    </div>
  </div>
</template>

<script>
import * as echarts from 'echarts'
import {getRandomColor} from "@/utils/gyy-util";

/**
 * 使用 只可以数据显示 目前未加入事件
 */
export default {
  name: "bar-chart",
  props: {
    //标题
    title: {type: String, default: null},
    //显示数据
    showData: {
      type: Object, default: () => ({
        colorList: ["#ff2f00", '#73DDFF', '#fe9a8b', '#F56948', '#9E87FF'],
        xData: ['一月', '二月', '三月', '四月', '五月', '六月'],
        data: [10, 20, 30, 12, 15, 4]
      })
    },
    // 主题 light or dark
    theme: {type: String, default: 'light'},
  },
  components: {},
  data() {
    return {
      mChart: null,
      isNull: true
    }
  },
  watch: {
    showData() {
      this.isNull = this.isNullData()
      this.initChart()
    }
  },
  mounted() {
    this.mChart = echarts.init(this.$refs.mChart);
    this.initChart()
  },
  methods: {
    isNullData() {
      if (!this.showData) {
        this.mChart.clear();
        return true
      }
      const seriesList = this.showData.seriesList
      if (seriesList == null || seriesList.length === 0) {
        this.mChart.clear();
        return true
      }
      return false
    },
    initChart() {
      const showData = this.showData
      // if (this.isNullData()) {
      //   this.mChart.setOption({
      //     backgroundColor: '#fff',
      //     title: {
      //       text: this.title,
      //       textStyle: {
      //         fontSize: 15,
      //         fontWeight: 600
      //       },
      //       left: 'center',
      //       top: '0%'
      //     },
      //   });
      //   return
      // }
      const data = showData.data;
      const titlename = showData.xData;
      const valdata = showData.data;
      const myColor = ['#1089E7', '#F57474', '#56D0E3', '#F8B448', '#8B78F6', '#a40cf6'];
      const option = {
        xAxis: {
          show: false
        },
        grid: {
          top: '2%',
          height: "90%",
        },
        yAxis: [
          {
            show: true,
            data: titlename,
            inverse: true,
            axisLine: {
              show: false
            },
            splitLine: {
              show: false
            },
            axisTick: {
              show: false
            },
            axisLabel: {
              color: '#fff',
              // formatter: function (value, index) {
              //   return [
              //     '{lg|' + (index + 1) + '}' + '{title|' + value + '} '
              //   ].join('\n')
              // },
              rich: {
                lg: {
                  backgroundColor: '#339911',
                  color: '#fff',
                  borderRadius: 15,
                  // padding: 5,
                  align: 'center',
                  width: 15,
                  height: 15
                },
              }
            },


          },
          {
            show: true,
            inverse: true,
            data: valdata,
            axisLabel: {
              textStyle: {
                fontSize: 12,
                color: '#fff',
                formatter: '{c}万元',
              },
            },
            axisLine: {
              show: false
            },
            splitLine: {
              show: false
            },
            axisTick: {
              show: false
            },

          }
          ],
        series: [
          {
            name: '条',
            type: 'bar',
            yAxisIndex: 0,
            data: data,
            barWidth: 12,
            itemStyle: {
              normal: {
                barBorderRadius: 30,
                color: function (params) {
                  const num = myColor.length;
                  return myColor[params.dataIndex % num]
                },
              }
            },
            label: {
              normal: {
                show: true,
                formatter: '{c}',
                fontSize: 12,
                color: '#fff',
              }
            },
          },
          // {
          //   name: '框',
          //   type: 'bar',
          //   yAxisIndex: 1,
          //   data: valdata.map(e=>100),
          //   barWidth: 15,
          //   itemStyle: {
          //     normal: {
          //       color: 'none',
          //       borderColor: '#00c1de',
          //       borderWidth: 3,
          //       barBorderRadius: 15,
          //     }
          //   }
          // },
        ]
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

  .line-chart {
    height: 100%;
    width: 100%;
    z-index: 1;
  }

  .text {
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
