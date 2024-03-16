<!--
*@dy-gender-chart
*@author yuge
*@date 2023/6/26 16:21
-->
<template>
  <div class="dy-gender-chart-main">
    <div class="people-img">
      <img class="upper" src="./img/female.png" alt=""/>
      <img class="lower" src="./img/female-bottom.png" alt=""/>
    </div>
    <div class="data-gender">
      <img src="./img/data-left.png" alt=""/>
      <div class="data-inner">
        <unit-number :number="showData.female||0" numberSize="25" unit="%"/>
      </div>
    </div>
    <div class="data-gender">
      <img src="./img/data-right.png" alt=""/>
      <div class="data-inner">
        <unit-number :number="showData.male||0" numberSize="25" unit="%"/>
      </div>
    </div>
    <div class="people-img">
      <img class="upper" style="width: 60%" src="./img/male.png" alt=""/>
      <img class="lower" src="./img/male-bottom.png" alt=""/>
    </div>
  </div>
</template>

<script>
import UnitNumber from "@/components/UnitNumber.vue";

export default {
  name: "GenderPieChart",
  props: {
    //显示数据
    showData: {
      type: Object, default: () => ({
        male:70,
        female:30
      })
    },
  },
  watch:{
  },
  components: {UnitNumber},
  data() {
    return {
      mChart:null
    }
  },
  mounted() {
  },
  methods: {
  },
  destroyed() {
  }
}
</script>

<style scoped lang="scss">
.dy-gender-chart-main {
  height: 100%;
  width: 100%;
  display: flex;
  padding: 10px 0;

  .people-img{
    height: 100%;
    width: 23%;
    position: relative;
    display: flex;
    align-items: flex-end;
    justify-content:center;

    .upper{
      width: 70%;
      margin: auto 0;
      z-index: 50;
    }

    .lower{
      position: absolute;
      bottom: 7%;
      left: -15%;
      width: 130%;
    }
  }

  .data-gender{
    position: relative;
    width: 23%;
    img{
      position: absolute;
      top: 0;
      left: 0;
      height: 100%;
      width: 100%;
      object-fit: scale-down;
    }

    .data-inner{
      position: absolute;
      top: 0;
      left: 0;
      height: 100%;
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
    }

  }
  .data-gender+.data-gender{
    margin-left: 8px;
  }
}
</style>
