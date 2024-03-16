<!--
*@DataVLoading
*@author yuge
*@date 2023/3/23 15:25
-->
<template>
  <div class="DataVLoading-main">
    <div class="data-loading-view" v-if="status !== 'success'">
      <dv-loading v-if="status === 'loading'">Loading...</dv-loading>
      <div class="loading-tip" v-if="status === 'error'">加载失败</div>
    </div>
  </div>
</template>

<script>
export default {
  name: "DataVLoading",
  props: {
    //success loading error
    status:{type:String,default:'success'}
  },
  components: {},
  data() {
    return {}
  },
  mounted() {
  },
  methods: {},
  destroyed() {
  }
}
</script>

<style scoped lang="scss">
.DataVLoading-main {
  .data-loading-view {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1;

    color: white;
    background-color: #000E28;
  }

  .data-loading-view::after {
    position: fixed;
    top: 0;
    left: 0;
    content: '';
    width: 100%;
    height: 100%;
    background-image: linear-gradient(0deg,
      #9FF7EA 0px,
      #9FF7EA 1px,
      transparent 1px,
      transparent 100px),
    linear-gradient(90deg,
        #9FF7EA 0px,
        #9FF7EA 1px,
        transparent 1px,
        transparent 100px);
    background-size: 50px 50px;
    opacity: 0.1;
    z-index: -1;
  }

}
</style>