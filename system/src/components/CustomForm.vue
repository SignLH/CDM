<!--
*@custom-form
*@author yuge
*@date 2023/6/28 10:07
-->
<template>
  <div class="custom-form-main">
    <dv-decoration-6 class="decoration" v-if="decoration"/>

    <div v-for="(item,i) in formData" :key="i">
      <div class="form-item" :style="{height:item.height}"  v-if="!(nullHided&&item.type!=='slot'&&!item.value)">
        <div class="title" :style="{width:titleWidth}">{{ item.title }}</div>
        <template v-if="item.type==='slot'">
          <slot :name="item.value"/>
        </template>
        <template v-else>
          <div class="value">{{ item.value }}</div>
        </template>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "custom-form",
  props: {
    formData: {
      type: Array,
      default: () => [
        {title: 'title1', value: 'value1',height:'25%'},
        {title: 'title2', value: 'value2',height:'25%'},
        {title: 'title3', value: 'value3',height:'25%'},
        {title: 'title4', value: 'value4',height:'25%'},
      ]
    },
    titleWidth:{
      type:String,
    },
    decoration:{
      type:Boolean,
    },
    nullHided:{
      type:Boolean,
    },
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
@mixin center{
  display: flex;
  align-items: center;
}

.custom-form-main {
  height: 100%;
  width: 100%;
  color: white;
  position: relative;
  overflow: auto;
  //background-color: #000e1c90;
  border-radius: 10px;
  padding: 4px;

  .decoration {
    width: 50%;
    height: 10%;
    position: absolute;
    top: 2px;
    right: 2px;
    overflow: hidden;
  }

  ::v-deep(.border-box-content){
    height: auto;
  }

  .form-item {
    display: flex;
    padding:4px;
    //height: 50px;

    .title{
      font-size: 17px;
      @include center;
      font-weight: bold;
    }
    .value{
      flex:1;
      font-size: 17px;
      @include center;
      margin-left: 4px;
    }
  }

  .form-item+.form-item{
    border-top: #003D7E 2px dashed;
  }
}
</style>
