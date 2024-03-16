<!--
*@index
*@author yuge
*@date 2024/1/26 12:10
-->
<template>
  <div class="notify-setting-main">
    <div class="item-main" v-for="(item,index) in notifyList" :key="item.name">
      <div class="left">
        <img src="./img/notify.svg" alt="">
        <div>
          <div class="title">{{item.name}}</div>
          <div class="status">状态：{{item.checked?'正常':'关闭'}}</div>
        </div>
      </div>
      <div class="right">
        <el-switch size="large" v-model="item.checked" active-color="#49C04B" inactive-color="#00000050">
        </el-switch>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'NotifySetting',
  props: {
    data:{
      type: Object,
      default: () => {
        return {
          notifyList:[
            {name:'微信', checked:true},
            {name:'邮箱', checked:true},
            {name:'短信', checked:true},
            {name:'企业微信' ,checked:true},
          ]
        }
      }
    }
  },
  components: {},
  data() {
    return {
      notifyList:[
        {name:'微信', checked:true},
        {name:'邮箱', checked:true},
        {name:'短信', checked:true},
        {name:'企业微信' ,checked:true},
      ]
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
.notify-setting-main {
  height: 100%;
  overflow: auto;
  color: white;
  &::-webkit-scrollbar {
    display: none;
  }
  .item-main{
    display: flex;
    margin-top: 8px;
    background: rgba(0, 0, 0, 0.20);
    border-radius: 8px;
    padding: 8px 32px;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
    .left{
      display: flex;
      img{
        margin-right: 4px;
      }
      .title{
        font-weight: bold;
      }
      .status{
        color: #ffffff90;
        //line-height: 18px;
      }
    }
    .right{

    }
  }
}
</style>
