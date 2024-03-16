<!--
*@pie-chart
*@author yuge
*@date 2023/4/27 14:05
-->
<template>
  <div class="pie-chart-main" ref="mChart"></div>
</template>

<script>
import * as echarts from 'echarts'

/**
 * 使用 只可以数据显示 目前未加入事件
 */
export default {
  name: "pie-chart",
  props: {
    //标题
    title: {type: String, default: ''},
    //显示数据
    showData: {
      type: Object, default: () => ({
        colorList: ['#afa3f595', '#00d48895', '#3feed495', '#3bafff95', '#f1bb4c95', '#afffff95'],
        dataList: [
          {value: 17, name: '烟雾传感器',},
          {value: 23, name: '火焰传感器'},
          {value: 27, name: '一氧化碳传感器'},
          {value: 33, name: '震动传感器'},
          {value: 9, name: '雨滴传感器'},
          {value: 9, name: '蜂鸣器传感器'}
        ],
      })
    },
    // 主题 light or dark
    theme: {type: String, default: 'light'},
    tooltip:{
      type: Object, default: ()=>({position: 'right'})
    }
  },
  components: {},
  data() {
    return {
      mChart: null
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
      if (!showData) {
        return
      }
      const {colorList, dataList} = showData

      var scaleData = [
        {
          name: '工程建设',
          value: 10,
        },
        {
          name: '产权交易',
          value: 10,
        },
        {
          name: '土地交易',
          value: 30,
        },
        {
          name: '其他交易',
          value: 10,
        },
        {
          name: '土地交易',
          value: 10,
        },
        {
          name: '其他交易',
          value: 10,
        },
      ];
// 随机颜色
      var rich = {
        white: {
          color: '#ddd',
          align: 'center',
          padding: [3, 0],
        },
      };
      var placeHolderStyle = {
        normal: {
          label: {
            show: false,
          },
          labelLine: {
            show: false,
          },
          color: 'rgba(0, 0, 0, 0)',
          borderColor: 'rgba(0, 0, 0, 0)',
          borderWidth: 0,
        },
      };
      function bg2() {
        return '#' + Math.floor(Math.random() * 0xffffff).toString(16);
      }

      function rancolors(len) {
        var color = [];
        for (var i = 0; i <= len; i++) {
          var sjys = bg2();
          color.push(sjys);
          for (var i = 0; i < color.length; i++) {
            if ((color[i] = sjys)) {
              color[i] = bg2();
            }
          }
        }
        return color;
      }
      var data = [];
      var color = rancolors(7);
      for (var i = 0; i < scaleData.length; i++) {
        data.push(
          {
            value: scaleData[i].value,
            name: scaleData[i].name,
            itemStyle: {
              normal: {
                borderWidth: 50,
                shadowBlur: 100,
                borderColor: color[i],
                shadowColor: color[i],
              },
            },
          },
          {
            value: 50,
            name: '',
            itemStyle: placeHolderStyle,
          }
        );
      }
      var seriesObj = [
        {
          name: '',
          type: 'pie',
          clockWise: false,
          radius: [50, 60],
          hoverAnimation: false,
          itemStyle: {
            normal: {
              label: {
                show: false,
                position: 'outside',
                color: '#ddd',
                formatter: function (params) {
                  var percent = 0;
                  var total = 0;
                  for (var i = 0; i < scaleData.length; i++) {
                    total += scaleData[i].value;
                  }
                  percent = ((params.value / total) * 100).toFixed(0);
                  if (params.name !== '') {
                    return params.name + '\n{white|' + '占比' + percent + '%}';
                  } else {
                    return '';
                  }
                },
                rich: rich,
              },
              labelLine: {
                length: 30,
                length2: 100,
                show: true,
                color: '#00ffff',
              },
            },
          },
          data: data,
        },
      ];
      const option = {
        tooltip: {
          show: false,
        },
        legend: {
          show: false,
        },
        toolbox: {
          show: false,
        },
        series: seriesObj,
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
.pie-chart-main {
  height: 100%;
  width: 100%;
}
</style>
