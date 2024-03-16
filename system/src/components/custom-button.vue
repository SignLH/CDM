<!--
*@custom-button
*@author GYY
*@date 2023/3/20 10:17
-->
<template>
  <div class="custom-button-main" :style="{
    color:useConfig.color,
    border: `0.5px ${useConfig.borderShadow} solid`,
    'box-shadow': `inset 0 0 15px 0 ${useConfig.color}`,
  }">
    <slot/>
  </div>
</template>

<script>
export default {
  name: "custom-button",
  props: {
    type:{type:String}
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
        const v= newValue||'info'
        // console.log(v,newValue)
        this.useConfig=this.colorList[configList.findIndex(e=>e===v)]
      },
      immediate: true,
    }
  },
  components: {},
  mounted() {
  },
  methods: {},
}
</script>

<style lang='scss' scoped>
.custom-button-main{
  height: 100%;
  width: 100%;

  display: flex;
  justify-content: center;
  align-items: center;
  text-align: center;
  color: #0db4f2;
  border: 0.5px #0091FF solid;
  box-shadow: inset 0 0 15px 0 #0091FF;
  border-radius: 2px;
  cursor: pointer;
  font-size: 14px;
  padding: 4px 8px;
}

.custom-button-main:hover{
  height: 100%;
  width: 100%;
  background-color: rgba(0, 0, 0, 0.10);
}
</style>
