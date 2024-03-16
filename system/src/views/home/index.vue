<template>
  <div class="page-view">
    <!--    left-->
    <div class="page-left-view">
      <div class="zhtx-view">
        <title-frame title="知识预警总览">
          <div class="flex-around">
            <city-image-block type="tech" title="预警学生" value="45" unit="个"/>
            <city-image-block type="tech" title="题目预警" value="17" unit="个"/>
          </div>
          <left-pie-chart style="height: calc(100% - 80px)"/>
        </title-frame>
      </div>
      <div class="cstz-view">
        <title-frame title="预警问题难度">
          <left-bar-chart/>
        </title-frame>
      </div>
      <div class="gcjg-view">
        <title-frame title="知识预警通知设置">
          <notify-setting/>
        </title-frame>
      </div>
    </div>
    <!--    right-->
    <div class="page-right-view">
      <div class="djyl-view">
        <title-frame title="数据总览">
          <div class="flex-around">
            <city-image-block type="people" title="学生总数" value="246" unit="个"/>
            <city-image-block type="people" title="问题总数" value="50" unit="个"/>
          </div>
        </title-frame>
      </div>
      <div class="stzl-view">
        <title-frame title="问题难度">
          <div class="flex-around">
            <city-image-block type="leaves" title="简单题" value="12" unit="个"/>
            <city-image-block type="leaves" title="中等题" value="28" unit="个"/>
            <city-image-block type="leaves" title="困难题" value="10" unit="个"/>
          </div>
        </title-frame>
      </div>
      <div class="jjfz-view">
        <title-frame title="预警问题原因画像">
          <hot-word-chart :word-data="hotWordData"/>
        </title-frame>
      </div>
    </div>
  </div>
</template>

<script>
import TitleFrame from '@/components/TitleFrame.vue'
import SpotLabel from '@/components/SpotLabel'
import UnitNumber from '@/components/UnitNumber'
import SolitaryElderlyDialog from '@/components/SolitaryElderlyDialog.vue'
import CaseInfoDialog from '@/components/CaseInfoDialog'
import ParkingDialog from '@/components/ParkingDialog'
import VideoMonitor from "@/components/VideoMonitor"
import {mapGetters} from 'vuex'
import AutoScrollList from '@/components/AutoScrollList'
import AutoHorizontalScroll from '@/components/AutoHorizontalScroll'
import AutoScrollTable from "@/components/auto-scroll-table.vue";
import IconDataBlock from "@/components/IconDataBlock.vue";
import HorizontalTab from "@/components/HorizontalTab.vue";
import PieChart from "@/components/Chart/PieChart/index.vue";
import BarChart from "@/components/Chart/BarChart/index.vue";
import CityImageBlock from "@/components/CityImageBlock/index.vue";
import LeftPieChart from "@/views/home/components/LeftPieChart.vue";
import LeftBarChart from "@/views/home/components/LeftBarChart.vue";
import NotifySetting from "@/components/NotifySetting/index.vue";
import HotWordChart from "@/components/HotWordChart/index.vue";

export default {
  components: {
    HotWordChart,
    NotifySetting,
    LeftBarChart,
    LeftPieChart,
    CityImageBlock,
    BarChart,
    PieChart,
    HorizontalTab,
    IconDataBlock,
    AutoScrollTable,
    TitleFrame,
    AutoScrollList,
    SpotLabel,
    UnitNumber,
    VideoMonitor,
    SolitaryElderlyDialog,
    AutoHorizontalScroll,
    CaseInfoDialog,
    ParkingDialog
  },

  data() {
    return {
      hotWordData: [
        {name: '作答超时', value: 31},
        {name: '编译错误', value: 31},
        {name: '运行超时', value: 31},
        {name: '逻辑错误', value: 31},
        {name: '语法错误', value: 31},
      ],
    }
  },
  computed: {
  },
  watch: {},
  filters: {
  },
  mounted() {
  },
}
</script>

<style lang="scss" scoped>
.page-left-view {
  position: absolute;
  top: 5px;
  //left: -530px;
  left: 0;
  width: 400px;
  height: calc(100% - 5px);

  backdrop-filter: blur(10px);

  display: flex;
  flex-direction: column;

  .zhtx-view {
    height: 300px;
    width: 100%;
    position: relative;
  }

  .cstz-view {
    margin: 10px 0;
    flex: 1;
    width: 100%;
    position: relative;
  }

  .gcjg-view {
    height: 180px;
    width: 100%;
    position: relative;
  }
}

.page-right-view {
  position: absolute;
  top: 5px;
  right: 0;
  width: 400px;
  height: calc(100% - 5px);

  backdrop-filter: blur(10px);
  display: flex;
  flex-direction: column;

  .djyl-view {
    height: 220px;
    width: 100%;
    position: relative;
  }

  .stzl-view {
    margin-top: 10px;
    height: 220px;
    width: 100%;
    position: relative;
  }

  .shzl-view {
    margin-top: 10px;
    height: 170px;
    width: 100%;
    position: relative;
  }

  .jjfz-view {
    margin-top: 10px;
    flex: 1;
    width: 100%;
    position: relative;
  }

  .zdjk-view {
    margin-top: 10px;
    height: 170px;
    width: 100%;
    position: relative;
  }

}

</style>
