<!--
*@horizontal-tab
*@author yuge
*@date 2023/6/25 17:22
-->
<template>
  <div class="horizontal-tab-main">
    <div class="action-item" v-for="(item,i) in tabValue" :key="i" @click="tabChange(item,i)">
      <img v-if="selected===i" src="@/assets/gcjg/btn-choose.png" class="back" alt=""/>
      <img v-else src="@/assets/gcjg/btn.png" class="back" alt=""/>
      <span>{{ item.title }}</span>
    </div>
  </div>
</template>

<script>
export default {
  name: "horizontal-tab",
  props: {
    value: {},
    tabValue: {
      type: Array, default: () => [
        {title: 'tab1', id: 1},
        {title: 'tab2', id: 1},
      ]
    },
    //tab btn
    type:{type:String,default:'tab'}
  },
  components: {},
  data() {
    return {
      selected: -1,
    }
  },
  mounted() {
  },
  methods: {
    tabChange(item, index) {
      // less than 1, don`t select
      if (this.tabValue.length>1&&this.type==='tab') {
        this.selected = index
      }
      this.$emit('tabChange', {item, index})
    }
  },
  destroyed() {
  }
}
</script>

<style scoped lang="scss">
.horizontal-tab-main {
  display: flex;
  .action-item {
    padding: 0 20px;
    height: 22px;
    margin-left: 5px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    color: #ffffff;
    cursor: pointer;

    .back {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: -1;
    }

  }

  .action-item:hover{
    background:url("../assets/gcjg/btn-choose.png") no-repeat;
    background-size:  100% 100%;
  }
}
</style>
