<template>
  <div class="hot-word-main" ref="main" id="chart-panel"></div>
</template>

<script>
import { deepClone } from '@/utils'
import * as echarts from 'echarts'

export default {
  name: 'HotWordChart',
  data() {
    return {
      chart: null,
      option: {
        //提示框
        tooltip: {
          trigger: 'item',
          extraCssText: 'color:white;background: rgba(8,18,18,0.70);border-radius: 4px;',
          formatter: function(params) {
            if (params['data']['value'] !== '') {
              return '<div>' + params['data']['value'] + '</div>'
            }
          }
        },
        animationEasingUpdate: 'bounceIn',
        series: [
          {
            type: 'graph',
            layout: 'force',
            force: {
              repulsion: 150,
              edgeLength: 10
            },
            // 	itemStyle:{
            // 	    opacity:1
            // 	},
            label: {
              show: true,
              fontWeight: '1000',
              fontSize: 14,
              align: 'center',
              verticalAlign: 'center',
              rich: {
                b: {
                  color: '#ffffff',
                  fontSize: 14,
                  // lineHeight: 30,
                  textBorderColor: 'transparent',
                  textBorderWidth: 0
                }
              },
              formatter(params) {
                let name = params.name
                if (name.length >= 5) {
                  name = name.slice(0, 4) + '\n' + name.slice(4)
                }
                return name
              }
            },
            data: []
          }
        ]
      }
    }
  },
  props: {
    //必须要有序
    wordData: {
      type: Array,
      default: () => {
        return [
          {
            name: '未带安全帽',
            value: 3000,
            searchObj: { type: '1' }
          },
          {
            name: '噪音',
            value: 1500,
            searchObj: { type: '1' }
          },
          {
            name: '车辆进出场异常',
            value: 1000,
            searchObj: { type: '1' }
          },
          {
            name: 'PM',
            value: 800,
            searchObj: { type: '1' }
          },
          {
            name: '重点岗位',
            value: 402,
            searchObj: { type: '1' }
          }
        ]
      }
    }
  },
  mounted() {
    this.initChart()
    this.renderChart()
  },
  watch: {
    wordData(val) {
      this.renderChart()
    }
  },
  methods: {
    initChart() {
      // 基于准备好的dom，初始化echarts实例
      this.chart = echarts.init(this.$refs.main)
    },
    renderChart() {
      let dataMini = {
        name: 'PM2.5',
        value: 50,
        symbol: 'circle',
        label: {
          normal: {
            textStyle: {
              color: '#fff',
              fontSize: 14
            }
          }
        },
        itemStyle: {
          // opacity: 0.5,
          // borderColor:'#000',
          color: {
            type: 'radial',
            x: 0.5,
            y: 0.5,
            r: 0.5,
            colorStops: [
              {
                offset: 0,
                color: '#99A4FF' // 0% 处的颜色
              },
              {
                offset: 1,
                color: '#4A56E2' // 100% 处的颜色
              }
            ],
            global: false // 缺省为 false
          }
        },
        symbolSize: 45,
        draggable: true
      }
      let wordData = this.wordData
      let dataCom = []
      //5个热词
      for (let i = 1; i <= wordData.length; i++) {
        let data = wordData[i - 1]
        dataMini.label.normal.textStyle.fontSize = this.getFontSize(i)
        dataMini.symbolSize = this.getBubbleSize(i)
        dataMini.itemStyle.color.colorStops[0].color = this.getColor(i)[0]
        dataMini.itemStyle.color.colorStops[1].color = this.getColor(i)[1]

        //用户数据
        dataMini.name = data.name
        dataMini.value = data.value
        dataMini.searchObj = data.searchObj
        dataCom.push(deepClone(dataMini))
      }
      this.option.series[0].data = dataCom

      this.chart.setOption(this.option)

      this.registerClickHotWord()
    },
    registerClickHotWord() {
      this.chart.on('click', (params) => {
        // console.log('hotword===='+params)
        this.$emit('clickHotWord', params)
      })
    },
    //第几个热词size
    getFontSize(index) {
      switch (index) {
        case 1:
          return 20
        case 2:
          return 16
        case 3:
          return 14
        case 4:
          return 12
        case 5:
          return 10
      }
    },
    getBubbleSize(index) {
      switch (index) {
        case 1:
          return 90
        case 2:
          return 80
        case 3:
          return 70
        case 4:
          return 60
        case 5:
          return 50
      }
    },
    getColor(index) {
      switch (index) {
        //返回渐变颜色 深色渐变浅色
        case 1:
          return ['#3F9FE3', '#146CCB']
        case 2:
          return ['#F3B58B', '#FF773D']
        case 3:
          return ['#F7D48F', '#EB9121']
        case 4:
          return ['#70CA64', '#2B8B21']
        case 5:
          return ['#99A4FF', '#4A56E2']
      }
    }
  }
}
</script>

<style lang="scss" scoped>
.hot-word-main {
  width: 100%;
  height: 100%;
}
</style>
