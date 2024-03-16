<template>
  <div class="container">
    <div ref="chart"></div>
  </div>
</template>

<script>
import {mapGetters} from 'vuex'
import * as echarts from 'echarts'

export default {
  name: 'StudentChart2',
  props: {
    dataList: {
      type: Array,
      default: () => {
        return []
      }
    }
  },
  computed: {},
  mounted() {
    this.chart = echarts.init(this.$refs.chart)
    this.renderChart()
  },
  data() {
    return {
      chart: null,
      isDialogShow: false,
      entity: {}
    }
  },
  watch: {},
  methods: {
    renderChart() {

      let data = [5, 11, 4, 6, 2, 4]
      let indicator = [{
        name: '作答超时',
        max: 15
      },
        {
          name: '部分正确',
          max: 15
        },
        {
          name: '语法错误',
          max: 15
        },
        {
          name: '运行超时',
          max: 15
        },
        {
          name: '逻辑错误',
          max: 15
        },
        {
          name: '编译错误',
          max: 15
        },
      ]

      function getV(v) {
        let res = []
        let n = 6
        while (n--) {
          res.push(15 - v)
        }
        return res
      }

      const option = {
        title: {
          text: "学生ID：20232002的错误表现画像",
          top: "top",
          left: "20",
          textStyle: {
            color: 'rgba(255,255,255,0.9)',
            fontSize: 16,
            fontWeight: 600,
          },
        },
        radar: {
          center: ['50%', '50%'],
          radius: '75%',
          name: {
            formatter: function (name, indicator) {
              let arr;
              arr = [
                '{a|' + name + '}'
              ]

              return arr.join('\n')
            },
            textStyle: {
              rich: { //根据文字的组设置格式
                a: {
                  color: '#BCDCFF',
                  fontSize: 14,
                  fontWeight: 600,
                  fontFamily: 'Source Han Sans CN',
                },
                // b:{
                //     fontSize:14,
                //     verticalAlign:'top',
                //     width:57,
                //     color:'#8E88FE',
                //     fontWeight:600,
                //     align:'center'
                // },
              }

            }
          },
          nameGap: 0,
          indicator: indicator,
          splitLine: {
            show: false

          },
          splitArea: {
            show: false
          },
          axisLine: {
            show: false
          }
        },
        series: [{
          // name: '家庭融合包',
          type: 'radar',
          data: [data],
          // value:14,
          label: {
            show: true,
            formatter: function (params) {
              console.log(params)
              return params.value + '个';
            },
            color: '#8E88FE',
            // position:[-20,-10,-10,-10],
            align: 'right',
            distance: 10,
          },
          symbolSize: [6, 6],
          lineStyle: { //边缘颜色
            width: 0
          },
          itemStyle: {
            borderWidth: 1,
            color: '#fff',
            borderColor: '#F2F063',
          },
          areaStyle: {
            color: '#7D77F1',
            opacity: 0.6
          }
        },
          {
            type: 'radar',
            data: [
              getV(0),
            ],
            symbol: 'none',
            lineStyle: {
              width: 0
            },
            itemStyle: {
              color: '#4175F5'
            },

            areaStyle: {
              color: '#4175F5',
              opacity: 0.06
            }
          },
          {
            type: 'radar',
            data: [
              getV(3),
            ],

            symbol: 'none',
            lineStyle: {
              width: 0
            },
            itemStyle: {
              color: '#2C72C8'
            },

            areaStyle: {
              color: '#2C72C8',
              opacity: 0.12
            }
          },
          {
            type: 'radar',
            data: [
              getV(6),
            ],

            symbol: 'none',
            lineStyle: {
              width: 0
            },
            itemStyle: {
              color: '#4175F5'
            },

            areaStyle: {
              color: '#4175F5',
              opacity: 0.18
            }
          },
          {
            type: 'radar',
            data: [
              getV(9),
            ],

            symbol: 'none',
            lineStyle: {
              width: 0
            },
            itemStyle: {
              color: '#4175F5'
            },

            areaStyle: {
              color: '#4175F5',
              opacity: 0.19
            }
          },
          {
            type: 'radar',
            data: [
              getV(12),
            ],

            symbol: 'none',
            lineStyle: {
              width: 0
            },
            itemStyle: {
              color: '#4175F5'
            },

            areaStyle: {
              color: '#4175F5',
              opacity: 0.17
            }
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
