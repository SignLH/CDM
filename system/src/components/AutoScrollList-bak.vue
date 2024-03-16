<template>
  <div ref="scrollMain" class="auto-scroll-list-main" @click="autoScrollClick($event)" :key="keyValue"
       @mouseover="mEnter" @mouseleave="mLeave">
    <div ref="scrollItemBox" class="seamless-warp-box">
      <slot></slot>
    </div>
    <!-- <div v-html="copyHtml" class="seamless-warp-box"></div> -->
  </div>
</template>

<script>

export default {
  name: 'AutoScrollList-bak',
  props: {
    listData: {type: Array},
    isPause: {type: Boolean, default: false},
    keyValue: {type: [String, Number], default: ''},
    //滚动的速度
    speed: {type: Number, default: 80}
  },
  data() {
    return {
      scrollTop: 0, //列表滚动高度
      copyHtml: null,
      scrollInterval: null,
    }
  },
  watch: {
    isPause(newValue, _) {
      if (newValue) {
        this.mEnter()
      } else {
        this.mLeave()
      }
    },
    listData: {
      deep: true,
      immediate: false,
      handler(newValue, _) {
        this.mEnter()
        this.initScroll()
      }
    },
    //刷新高度为0
    keyValue(newV, _) {
      this.scrollTop = 0
    }
  },
  mounted() {
    this.initScroll()
  },
  methods: {
    initScroll() {
      this.$nextTick(() => {
        // this.copyHtml = this.$refs.scrollItemBox.innerHTML
        this.startScroll()
      })
    },
    // 鼠标移入停止滚动
    mEnter() {
      if (this.scrollInterval) {
        clearInterval(this.scrollInterval)
        this.scrollInterval = null
      }
    },
    // 鼠标移出继续滚动
    mLeave() {
      this.startScroll()
    },
    // 开始滚动
    startScroll() {
      if (this.scrollInterval) {
        return
      }
      //在当前位置开始滚动
      this.scrollTop = this.$refs.scrollMain.scrollTop
      this.scrollInterval = setInterval(this.scroll, this.speed)
    },
    // 滚动处理方法
    scroll() {
      this.scrollTop = this.scrollTop + 1
      this.$refs.scrollMain.scrollTop = this.scrollTop

      const clientHeight = this.$refs.scrollMain.clientHeight;
      const scrollTop = this.scrollTop;
      const scrollHeight = this.$refs.scrollMain.scrollHeight;
      const scrollEndDelay = 30  //像素
      if (clientHeight + scrollTop >= scrollHeight + scrollEndDelay) {
        this.scrollTop = 0
      }

    },
    autoScrollClick(e) {
      let index = e.target.dataset.index
      if (!index) {
        return
      }
      //默认是number 自己改
      this.$emit('autoScrollClick', Number(index))
    }
  },
  destroyed() {
    this.mEnter()
  }
}
</script>

<style lang="scss" scoped>
.auto-scroll-list-main {
  width: 100%;
  height: 100%;
  overflow: auto;
}

.auto-scroll-list-main::-webkit-scrollbar {
  display: none
}

.seamless-warp-box {
  width: 100%;
}
</style>
