<template>
  <div class="container">
    <div ref="chart"></div>
  </div>
</template>

<script>
import {mapGetters} from 'vuex'
import * as echarts from 'echarts'

export default {
  name: 'LeftPieChart',
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
      const echarts = this.$echarts

      var data = {
        dataSex: [{
          name: '名称1',
          value: '112',
        }, {
          value: '202',
        }],
        dataAge: [{
          name: '预警学生',
          value: '45'
        }, {
          name: '题目预警',
          value: '17'
        }]
      };
      const option = {
        grid: {
          left: '20%',
          top: '20%',
          right: '20%',
          bottom: '20%',
        },
        color: ['#2AC9FD', '#76FBC0', '#35C96E', '#FCC708', '#48B188', '#5957C2', '#4A5D73'],
        series: [
          {
            color: ['#2AC9FD', '#76FBC0', '#35C96E', '#FCC708', '#48B188', '#5957C2'],
            type: 'pie',
            radius: ['32', '70'],
            data: data.dataAge,
          },
          {
            type: 'pie',
            radius: ['79', '80'],
            data: [100],
            label: {
              show: false
            }
          },
          {
            type: 'pie',
            radius: ['27', '28'],
            data: [100],
            label: {
              show: false
            }
          }
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
