<!--
*@SelectMenu 下拉菜单
*@author GYY
*@date 2023/2/17 9:49
-->
<template>
  <div class="select-menu-main">
    <div class="select-title" @click="isShowPanel=!isShowPanel">
      {{ currentName }}
      <img src="./icon/drop-icon.png" alt="" :class="{'img-rotate':isShowPanel}"/>
    </div>
    <transition name="drop">
      <div class="select-menu-panel" v-show="isShowPanel">
        <div v-for="(item,i) in dataList" :key="i" @click="change(item)"
             class="cell" :class="{'cell-selected':currentName===item.name}">
          {{ item.name }}
        </div>
      </div>
    </transition>
  </div>
</template>

<script>
export default {
  name: "SelectMenu",
  props: {
    dataList: {
      type: Array, default: () => {
        return [
          {name: '1920*1080', value: '1920*1080'},
          {name: '3840*1080', value: '3840*1080'}
        ]
      }
    },
    currentName: {
      type: [String,Number], default: "1920*1080"
    },
  },
  data() {
    return {
      isShowPanel: false,
    }
  },
  watch: {},
  components: {},
  mounted() {

  },
  methods: {
    change(item) {
      this.$emit('change',item)
      this.isShowPanel=false
    }
  },
}
</script>

<style lang='scss' scoped>
/* 设置持续时间和动画函数 */
.drop-enter-active, .drop-leave-active {
  transition: all .5s ease;
}

.drop-enter, .drop-leave-to {
  transform: translateY(-5px);
  opacity: 0;
}

.select-menu-main {
  height: 100%;
  width: 100%;
  font-size: 14px;
  position: relative;

  .select-title {
    height: 100%;
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;

    box-sizing: border-box;
    padding: 0 8px;

    border: 1px #0099ff solid;
    border-radius: 16px;
    img {
      height: 50%;
    }

    .img-rotate {
      transform: rotate(180deg);
    }
  }

  .select-menu-panel {
    position: absolute;
    top: calc(100% + 5px);
    left: 0;

    width: 100%;
    border: 1px #0099ff solid;
    border-radius: 16px;
    background-color: #0a5fa4;
    z-index: 49;

    .cell {
      line-height: 20px;
      cursor: pointer;

      box-sizing: border-box;
      padding: 0 5px;
    }

    .cell-selected {
      background-color: rgba(0, 140, 198);
      border-radius: 16px;
    }
  }

}
</style>