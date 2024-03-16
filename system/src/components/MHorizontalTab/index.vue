<template>
  <div class="m-tab-main" :style="{borderRadius}">
    <div v-for="(item,index) in tabValue" @click="changeTab(item,index)" :key="index"
         class="m-tab-child" @mouseover="hoverThis(index)" @mouseout="isHover=-1"
         :class="{'m-tab-child-selected':index===defaultIndex,'m-tab-child-hover':index===isHover}">

        <div v-if="!isHaveSubtitle">{{ item.title }}</div>
        <div v-else >
<!--          <div style="display: flex;flex-direction: column;-->
<!--        justify-content: center;align-items: center;"-->
<!--          >-->
<!--            <div>{{ item.title }}</div>-->
<!--            <div class="m-tab-subtitle">{{ item.subtitle }}</div>-->
<!--          </div>-->
          {{ item.title }} (<span class="number-font">{{item.subtitle}}</span>)
        </div>

    </div>
  </div>

</template>

<script>
export default {
  name: 'MHorizontalTab',
  data() {
    return {
      defaultIndex: 0,
      borderRadius: '10.5px',
      isHover:-1,
    }
  },
  created() {
    if (!this.isHaveSubtitle) {
      this.borderRadius = '10.5px'
    } else {
      this.borderRadius = '24px'
    }
  },
  props: {
    tabValue: {
      type: Array,
      default: () => {
        return [
          //有subtitle 会有第二行
          { title: '测试1', subtitle: 'subtitle', value: '1' },
          { title: '测试2', subtitle: 'subtitle', value: '2' },
          { title: '全部', subtitle: 'subtitle', value: '2' }
        ]
      }
    },
    //是否有副标题
    isHaveSubtitle: {type: Boolean,default: false},

    currentIndex: {type: Number,default: 0},
  },
  watch:{
    currentIndex:{
      handler(newV,_){
        this.defaultIndex=newV
      },
      immediate:true
    }
  },
  methods: {
    changeTab(item, index) {
      this.defaultIndex = index
      this.$emit('changeTab', item)
    },
    hoverThis(index){
        this.isHover=index
    }
  }
}
</script>

<style scoped lang="scss">

//tab 按键组
@function linear-gradient-main-transparency($c1,$c2) {
  @return linear-gradient(180deg, rgba(0, 140, 198, $c1) 0%, rgba(42, 244, 218, $c2) 100%);
}

.m-tab-main {
  display: flex;
  background-image: linear-gradient-main-transparency(0.4,0.4);
  font-size: 12px;
  color: #FFFFFF;
  height: 100%;
  width: 100%;
  border-radius: 24px;
}

.m-tab-child {
  cursor: pointer;
  flex: 1;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: space-around;
}

.m-tab-child-selected {
  border-radius: 24px;
  background-image: linear-gradient-main-transparency(0.7,0.7);
}

.m-tab-child-hover {
  border-radius: 24px;
  background-image: linear-gradient-main-transparency(0.1,0.1);
}

.m-tab-subtitle{
  font-size: 12px;
  margin-top: 4%;
}
</style>
