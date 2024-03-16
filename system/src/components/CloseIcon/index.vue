<!--
*@CloseIcon
*@author yuge
*@date 2023/3/25 17:47
-->
<template>
  <div class="CloseIcon-main" :style="{height}">
    <img src="./img/close-normal.png" alt=""/>
  </div>
</template>

<script>
export default {
  name: "CloseIcon",
  props: {
    height:{type:String,default:'20px'}
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
.CloseIcon-main {
  position: absolute;
  top: 1px;
  right: 1px;

  width: auto;
  //background-color: white;

  img{
    height: 100%;
    //width: 16px;
    //width: auto;
  }
}
</style>