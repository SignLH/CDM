<!--
*@icon-data-block
*@author yuge
*@date 2023/6/25 16:16
-->
<template>
  <div class="icon-data-block-main" :class="{clickable:clickable,[styleType]:true}" @click="$emit('click')">
    <div class="value">
      <img :src="useIconBack" v-if="useIconBack" class="icon-back" alt=""/>
      <img :src="useIcon" class="icon-data-block" alt=""/>
      <div style="z-index: 50"><slot/></div>
    </div>
    <div class="label">{{ title||'未定义' }}</div>
  </div>
</template>

<script>
export default {
  name: "icon-data-block",
  props: {
    // 下部分标题
    title:{type:String},
    //主icon
    useIcon:{type:String,default:require('@/assets/stzl/stzl-back1.png')},
    //back icon
    useIconBack:{type:String},
    //是否可点击
    clickable: {type:Boolean},
    // leftRight
    styleType:{type:String,default: 'normal'}
  },
  components: {},
  data() {
    return {
    }
  },
  mounted() {
  },
  methods: {},
  destroyed() {
  }
}
</script>

<style scoped lang="scss">
.icon-data-block-main {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  position: relative;
  border-radius: 8px;
  overflow: hidden;
}
.clickable{
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
}
.clickable:hover{
  //background-color: rgba(255, 255, 255, 0.1);
  //backdrop-filter: blur(5px);
}

.icon-data-block-main.normal{
  .value {
    width: 100%;
    margin-top: 8px;
    height: 26px;
    color: #0BFFA6;
    display: flex;
    align-items: center;
    justify-content: center;

    .icon-data-block {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 60%;
      margin-top: 4px;
      object-fit: cover;
      z-index: -1;
      opacity: .8;
    }
  }

  .label {
    margin-top: 3px;
    color: #ffffff;
    text-align: center;
    font-size: 12px;
  }
}

.icon-data-block-main.leftRight{

  .label {
    margin-top: 3px;
    color: #ffffff;
    text-align: center;
    font-size: 14px;
  }

  .value {
    width: 100%;
    height: calc(90% - 18px);
    background: #01112ad2;
    border: 2px solid;
    border-image: linear-gradient(to right, #ffffff00, #30adcc96, #ffffff00) 2;
    padding-left: 45%;
    margin-top: 5px;
    position: relative;
    color: #0BFFA6;
    display: flex;
    align-items: center;
    justify-content: center;

    .icon-back {
      position: absolute;
      bottom: 4px;
      left: 8px;
      width: 50px;
      height: auto;
    }

    .icon-back2 {
      position: absolute;
      bottom: 1px;
      left: 6px;
      width: 54px;
      height: auto;
    }

    .icon-data-block {
      position: absolute;
      top: 3px;
      left: 20px;
      width: 27px;
      height: auto;
    }
  }
}

.icon-data-block-main.leftRight2{
  position: relative;
  .label {
    position: absolute;
    bottom: 10%;
    right: 4px;

    margin-top: 3px;
    color: #ffffff;
    text-align: center;
    font-size: 14px;
  }

  .value {
    width: 100%;
    height: calc(90% - 16px);
    padding-left: 45%;
    margin-top: 0;
    display: flex;
    justify-content: end;
    color: white;
  }

  .icon-data-block {
    position: absolute;
    top: 22%;
    left: 0;
    width: auto;
    height: 65%;
  }
}
</style>
