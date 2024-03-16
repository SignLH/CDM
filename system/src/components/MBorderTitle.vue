<!--
*@MBorderTitle
*@author GYY
*@date 2023/2/28 15:19
-->
<template>
  <div class="m-border-title-main" :style="{
    color:useConfig.color,
    border: `0.5px ${useConfig.borderShadow} solid`,
    'box-shadow': `inset 0 0 25px 0 ${useConfig.color}`,
  }" :title="title">
    {{ title }}
  </div>
</template>

<script>
export default {
  name: "MBorderTitle",
  props: {
    title: {type: String, default: '测试治理'},
    // info success warn error
    type: {type: String, default: 'info'},
  },
  data() {
    return {
      colorList: [
        //info
        {color: '#0db4f2', borderShadow: '#0091FF'},
        //success
        {color: '#0ce28c', borderShadow: '#00965d'},
        //warn
        {color: '#f4b148', borderShadow: '#bb7000'},
        //error
        {color: '#e32c00', borderShadow: '#8c1f00'},
      ],
      useConfig: {color: '#0db4f2', borderShadow: '#0091FF'}
    }
  },
  watch: {
    type:{
      handler(newValue, _) {
        const configList=['info','success','warn','error']
        this.useConfig=this.colorList[configList.findIndex(e=>e===newValue)]
      },
      immediate: true
    }
  },
  components: {},
  mounted() {
  },
  methods: {},
}
</script>

<style lang='scss' scoped>
.m-border-title-main {
  width: 100%;
  height: 100%;
  position: relative;

  display: flex;
  justify-content: center;
  align-items: center;
  text-align: center;
  color: #0db4f2;
  border: 0.5px #0091FF solid;
  box-shadow: inset 0 0 25px 0 #0091FF;

  //font-size: 21px;
  //clip-path: polygon(0px 0px, 150px 0px, 150px 35px, 135px 50px, 0px 50px);
}

</style>
