<template>
  <img class="close" :src="require(isHover?'./img/close-red.svg':'./img/close.svg')"
       @mouseenter="isHover=true"  @mouseleave="isHover=false"
       :style="{width:size,height:size}" @click="handlerClick" alt=""/>
</template>

<script>
export default {
  name: 'CrossSign',
  data() {
    return {
      isHover: false
    }
  },
  props: {
    size: {
      type: String,
      default: '16px'
    }
  },
  methods: {
    handlerClick() {
      this.$emit('clickExit')
    }
  }
}
</script>

<style scoped lang="scss">
.close{
  cursor: pointer;
}
</style>
